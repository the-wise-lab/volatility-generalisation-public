from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_integrated_posterior_predictive_check.py"
    spec = importlib.util.spec_from_file_location("run_integrated_posterior_predictive_check", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_placeholder_fit(result_dir: Path) -> None:
    """Create the two files that mark a model folder as a finished fit.

    Model discovery only checks that ``metadata.json`` and ``posterior.nc``
    exist, so both files are left empty. Do not use this for a test that loads
    or reads the fit.

    Args:
        result_dir: Folder for one model, created if it does not exist.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metadata.json").touch()
    (result_dir / "posterior.nc").touch()


def test_choice_prob_from_log_likelihood_inverts_observed_choice_likelihood() -> None:
    module = _load_script_module()

    log_likelihood = np.log(np.array([0.2, 0.6, 0.9], dtype=float))
    choices = np.array([1.0, 0.0, 1.0], dtype=float)

    reconstructed = module._choice_prob_from_log_likelihood(log_likelihood, choices)

    np.testing.assert_allclose(reconstructed, np.array([0.2, 0.4, 0.9], dtype=float))


def test_trialwise_condition_ppc_aggregates_by_condition_and_trial() -> None:
    module = _load_script_module()

    choices = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    condition_labels = ["StableV1", "StableV1", "VolatileV2"]
    p_choose_0 = np.array(
        [
            [[0.9, 0.2, 0.8], [0.6, 0.3, 0.7], [0.4, 0.5, 0.1]],
            [[0.7, 0.4, 0.9], [0.5, 0.2, 0.6], [0.2, 0.7, 0.3]],
        ],
        dtype=float,
    )
    log_likelihood = np.log(
        np.where(choices[None, None, :, :] > 0.5, p_choose_0[None, :, :, :], 1.0 - p_choose_0[None, :, :, :])
    )
    idata = az.from_dict(log_likelihood={"obs": log_likelihood})
    data = {"choices": choices, "condition_labels": condition_labels}

    seed = 123
    trialwise_df, summary_df = module._trialwise_condition_ppc(idata, data, "demo_model", seed=seed)

    assert len(trialwise_df) == 6
    assert list(trialwise_df["condition"].unique()) == ["StableV1", "VolatileV2"]
    assert list(trialwise_df[trialwise_df["condition"] == "StableV1"]["trial"]) == [0, 1, 2]

    expected_trial_rows = []
    expected_summary_rows = []
    conditions = ["StableV1", "VolatileV2"]
    rng = np.random.default_rng(seed)
    for condition in conditions:
        mask = np.array(condition_labels) == condition
        observed = choices[mask].mean(axis=0)
        condition_probabilities = p_choose_0[:, mask, :]
        predicted_by_draw_trial = condition_probabilities.mean(axis=1)
        posterior_predictive_by_draw_trial = rng.binomial(1, condition_probabilities).mean(axis=1)
        predicted = predicted_by_draw_trial.mean(axis=0)
        predicted_by_subject_trial = condition_probabilities.mean(axis=0) >= 0.5
        predictive_accuracy = (predicted_by_subject_trial == choices[mask]).mean()
        lower = np.percentile(posterior_predictive_by_draw_trial, 2.5, axis=0)
        upper = np.percentile(posterior_predictive_by_draw_trial, 97.5, axis=0)
        for trial in range(choices.shape[1]):
            expected_trial_rows.append(
                {
                    "model": "demo_model",
                    "condition": condition,
                    "trial": trial,
                    "n_subjects": int(mask.sum()),
                    "observed_choice_prob": float(observed[trial]),
                    "predicted_choice_prob": float(predicted[trial]),
                    "predicted_choice_prob_ci_lower": float(lower[trial]),
                    "predicted_choice_prob_ci_upper": float(upper[trial]),
                    "n_draws_used": 2,
                }
            )
        residuals = predicted - observed
        expected_summary_rows.append(
            {
                "model": "demo_model",
                "condition": condition,
                "n_subjects": int(mask.sum()),
                "n_trials": 3,
                "n_draws_used": 2,
                "observed_choice_prob_mean": float(observed.mean()),
                "predicted_choice_prob_mean": float(predicted.mean()),
                "predictive_accuracy": float(predictive_accuracy),
                "mae": float(np.abs(residuals).mean()),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "mean_interval_width": float(np.mean(upper - lower)),
            }
        )

    pd.testing.assert_frame_equal(
        trialwise_df.reset_index(drop=True),
        pd.DataFrame(expected_trial_rows),
        check_like=False,
    )
    pd.testing.assert_frame_equal(
        summary_df.reset_index(drop=True),
        pd.DataFrame(expected_summary_rows),
        check_like=False,
    )


def test_resolve_models_uses_available_fits_by_default(tmp_path: Path) -> None:
    module = _load_script_module()
    results_dir = tmp_path / "results"
    _write_placeholder_fit(results_dir / "dual_lr_single_value")
    _write_placeholder_fit(results_dir / "custom_model")
    (results_dir / "incomplete_model").mkdir(parents=True, exist_ok=True)
    (results_dir / "incomplete_model" / "metadata.json").write_text("{}", encoding="utf-8")

    assert module._resolve_models(results_dir, None) == ["custom_model", "dual_lr_single_value"]


@pytest.mark.parametrize("missing_file", ["metadata.json", "posterior.nc"])
def test_resolve_models_errors_for_incomplete_known_model_folder(
    tmp_path: Path, missing_file: str
) -> None:
    module = _load_script_module()
    results_dir = tmp_path / "results"
    _write_placeholder_fit(results_dir / "dual_lr_single_value")
    _write_placeholder_fit(results_dir / "rescorla_wagner_single_value")
    (results_dir / "rescorla_wagner_single_value" / missing_file).unlink()

    with pytest.raises(FileNotFoundError) as exc:
        module._resolve_models(results_dir, None)

    assert str(results_dir / "rescorla_wagner_single_value") in str(exc.value)


def test_resolve_models_ignores_folders_that_are_not_models(tmp_path: Path) -> None:
    module = _load_script_module()
    results_dir = tmp_path / "results"
    _write_placeholder_fit(results_dir / "dual_lr_single_value")
    (results_dir / "model_comparison" / "tables").mkdir(parents=True)
    (results_dir / "posterior_predictive").mkdir()

    assert module._resolve_models(results_dir, None) == ["dual_lr_single_value"]


def test_resolve_output_dir_defaults_to_posterior_predictive() -> None:
    module = _load_script_module()

    assert module._resolve_output_dir(Path("results/integrated_model_selection"), None) == Path(
        "results/integrated_model_selection/posterior_predictive"
    )


def test_models_do_not_change_output_namespace(tmp_path: Path) -> None:
    module = _load_script_module()
    results_dir = tmp_path / "results"
    _write_placeholder_fit(results_dir / "dual_lr_single_value")

    assert module._resolve_models(results_dir, ["dual_lr_single_value"]) == [
        "dual_lr_single_value"
    ]
    assert module._resolve_output_dir(Path("results/integrated_model_selection"), None) == Path(
        "results/integrated_model_selection/posterior_predictive"
    )


def test_nonempty_output_dir_requires_overwrite(tmp_path: Path) -> None:
    module = _load_script_module()
    output_dir = tmp_path / "posterior_predictive"
    output_dir.mkdir()
    (output_dir / "existing.csv").write_text("x\n", encoding="utf-8")

    with np.testing.assert_raises(RuntimeError):
        module._prepare_output_dir(output_dir, overwrite=False)

    module._prepare_output_dir(output_dir, overwrite=True)
    assert output_dir.exists()


def test_resolve_models_errors_for_missing_explicit_model(tmp_path: Path) -> None:
    module = _load_script_module()
    results_dir = tmp_path / "results"
    _write_placeholder_fit(results_dir / "dual_lr_single_value")

    with np.testing.assert_raises(FileNotFoundError) as exc:
        module._resolve_models(results_dir, ["missing_model"])

    assert "missing_model" in str(exc.exception)


@pytest.mark.rendering
def test_main_writes_ppc_tables_and_figures(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    results_dir = tmp_path / "results" / "integrated_model_selection"
    data_dir = tmp_path / "data" / "prepared"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir = results_dir / "dual_lr_single_value"
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "dual_lr_single_value",
        "conditions": ["StableV1", "VolatileV2"],
        "subject_ids": ["s1", "s2", "t1"],
        "subject_labels": ["StableV1__s1", "StableV1__s2", "VolatileV2__t1"],
        "condition_labels": ["StableV1", "StableV1", "VolatileV2"],
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    posterior = np.log(
        np.array(
            [[[[0.8, 0.4], [0.6, 0.3], [0.2, 0.7]], [[0.7, 0.5], [0.5, 0.2], [0.3, 0.6]]]],
            dtype=float,
        )
    )
    az.from_dict(log_likelihood={"obs": posterior}).to_netcdf(model_dir / "posterior.nc")

    np.savez(
        data_dir / "StableV1.npz",
        outcomes_0=np.array([[1.0, 0.0], [0.0, 1.0]]),
        outcomes_1=np.array([[0.0, 1.0], [1.0, 0.0]]),
        choices=np.array([[1.0, 0.0], [0.0, 1.0]]),
        subject_ids=np.array(["s1", "s2"]),
    )
    np.savez(
        data_dir / "VolatileV2.npz",
        outcomes_0=np.array([[1.0, 1.0]]),
        outcomes_1=np.array([[0.0, 0.0]]),
        choices=np.array([[1.0, 0.0]]),
        subject_ids=np.array(["t1"]),
    )

    output_dir = tmp_path / "compiled"
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            results_dir=str(results_dir),
            output_dir=str(output_dir),
            models=["dual_lr_single_value"],
            data_dir=str(data_dir),
            n_draws=2,
            accuracy_warning_threshold=0.55,
            seed=7,
            overwrite=False,
        ),
    )

    module.main()

    assert (output_dir / "tables" / "posterior_predictive_trialwise.csv").exists()
    assert (output_dir / "tables" / "posterior_predictive_summary.csv").exists()
    assert any(output_dir.glob("figures/*posterior_predictive*.png"))
