import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volatility_generalisation.modelling_code import parameter_recovery
from volatility_generalisation.modelling_code.data_generation import (
    DEFAULT_MODEL_BOUNDS_CONFIG,
    generate_synthetic_data_single_model,
)
from volatility_generalisation.modelling_code.data_preparation import (
    OUTCOME_CODING_KEY,
    OUTCOME_CODING_REWARD_SAFE,
)
from volatility_generalisation.modelling_code.hierarchical_bayes import (
    _normalize_fit_data,
)
from volatility_generalisation.modelling_code.parameter_recovery import (
    build_parameter_recovery_dataframe,
    load_prepared_schedule,
    pack_posterior_parameter_samples,
    pack_true_parameter_array,
    run_parameter_recovery,
    summarize_parameter_recovery,
)

# Reward-coded outcomes (1 = safe) for two participants over 8 trials. The
# pattern is uneven, so a flipped copy can't be mistaken for the original.
REWARD_OUTCOMES_0 = np.array(
    [
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    ]
)
REWARD_OUTCOMES_1 = 1.0 - REWARD_OUTCOMES_0


def _write_prepared_file(path, outcome_coding=OUTCOME_CODING_REWARD_SAFE):
    """Write a small prepared condition file, like the ones prepare_data.py makes.

    Args:
        path: Where to write the ``.npz`` file.
        outcome_coding: Value of the file's ``outcome_coding`` entry, or None to
            leave the entry out (an older, explosion-coded file).

    Returns:
        The path written.
    """
    arrays = {
        "outcomes_0": REWARD_OUTCOMES_0,
        "outcomes_1": REWARD_OUTCOMES_1,
        "choices": np.ones_like(REWARD_OUTCOMES_0),
        "subject_ids": np.array(["a", "b"]),
    }
    if outcome_coding is not None:
        arrays[OUTCOME_CODING_KEY] = outcome_coding
    np.savez(path, **arrays)
    return path


def test_load_prepared_schedule_keeps_reward_coded_outcomes(tmp_path):
    """A reward-coded file gives back the first participant's outcomes unchanged."""
    schedule = load_prepared_schedule(_write_prepared_file(tmp_path / "c.npz"))

    np.testing.assert_array_equal(schedule["outcomes_0"], REWARD_OUTCOMES_0[0])
    np.testing.assert_array_equal(schedule["outcomes_1"], REWARD_OUTCOMES_1[0])


def test_load_prepared_schedule_flips_a_file_without_outcome_coding(tmp_path):
    """A file without ``outcome_coding`` is treated as explosion-coded."""
    path = _write_prepared_file(tmp_path / "c.npz", outcome_coding=None)

    schedule = load_prepared_schedule(path)

    np.testing.assert_array_equal(schedule["outcomes_0"], 1.0 - REWARD_OUTCOMES_0[0])
    np.testing.assert_array_equal(schedule["outcomes_1"], 1.0 - REWARD_OUTCOMES_1[0])


def test_simulated_outcomes_reach_the_fit_unchanged(tmp_path):
    """The fit sees the same outcomes that the choices were simulated from."""
    schedule = load_prepared_schedule(_write_prepared_file(tmp_path / "c.npz"))

    synthetic = generate_synthetic_data_single_model(
        "rescorla_wagner_single_value", 3, 8, schedule, rng_seed=0
    )
    fitted = _normalize_fit_data(synthetic["data_for_fit"])

    expected_0 = np.tile(REWARD_OUTCOMES_0[0], (3, 1))
    np.testing.assert_array_equal(synthetic["data_for_fit"]["outcomes_0"], expected_0)
    np.testing.assert_array_equal(fitted["outcomes_0"], expected_0)
    np.testing.assert_array_equal(fitted["outcomes_1"], 1.0 - expected_0)


def test_run_parameter_recovery_fits_the_simulated_outcomes(tmp_path, monkeypatch):
    """The recovery run passes the model the outcomes it simulated from."""
    _write_prepared_file(tmp_path / "StableV1.npz")
    seen = {}

    class StopAfterFit(Exception):
        """Raised by the stand-in fit, to end the run once the data is seen."""

    def record_fit(data, **kwargs):
        """Keep the data the fit would use, then stop the run.

        Args:
            data: The data passed to the fit.
            **kwargs: The fit settings, ignored.

        Raises:
            StopAfterFit: Always.
        """
        seen.update(_normalize_fit_data(data))
        raise StopAfterFit

    monkeypatch.setattr(parameter_recovery, "fit_mcmc", record_fit)
    with pytest.raises(StopAfterFit):
        run_parameter_recovery(
            models=["rescorla_wagner_single_value"],
            conditions=["StableV1"],
            data_dir=tmp_path,
            output_dir=tmp_path / "out",
            n_subjects=3,
            num_warmup=1,
            num_samples=1,
            num_chains=1,
            seed=0,
            progress_bar=False,
        )

    expected_0 = np.tile(REWARD_OUTCOMES_0[0], (3, 1))
    np.testing.assert_array_equal(seen["outcomes_0"], expected_0)
    np.testing.assert_array_equal(seen["outcomes_1"], 1.0 - expected_0)


@pytest.mark.parametrize("shift_one_chain", [False, True])
def test_run_parameter_recovery_records_settings_and_convergence(
    tmp_path, monkeypatch, shift_one_chain
):
    """Each recovery run saves its settings file and hash, and its convergence.

    The stand-in fit edits the bounds file, to check that the hash is of the
    file as the fit read it. It returns four chains. With ``shift_one_chain``,
    one chain of the first parameter sits far from the others, so the fit does
    not converge; both models must still write all their outputs.
    """
    _write_prepared_file(tmp_path / "StableV1.npz")
    config_path = tmp_path / "model_bounds.toml"
    config_path.write_bytes(DEFAULT_MODEL_BOUNDS_CONFIG.read_bytes())
    model_names = (
        ["rescorla_wagner_single_value", "dual_lr_single_value"]
        if shift_one_chain
        else ["rescorla_wagner_single_value"]
    )
    original_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    class FakeMCMC:
        """Stand-in for a finished fit: four chains of 500 draws, 3 participants."""

        num_chains = 4

        def __init__(self, model_name):
            """Store the model to make samples for.

            Args:
                model_name: Name of the fitted model.
            """
            self.model_name = model_name

        def get_samples(self, group_by_chain=False):
            """Return posterior samples for every parameter.

            Args:
                group_by_chain: Whether to keep the chain axis.

            Returns:
                Dictionary from site name to an array of shape (4, 500, 3), or
                (2000, 3) without the chain axis.
            """
            rng = np.random.default_rng(0)
            samples = {}
            for i, name in enumerate(
                parameter_recovery.get_parameter_names(self.model_name, config_path)
            ):
                values = 0.5 + 0.01 * rng.standard_normal((4, 500, 3))
                if shift_one_chain and i == 0:
                    values[0] += 1.0
                samples[f"{self.model_name}_{name}_subj"] = (
                    values if group_by_chain else values.reshape(-1, 3)
                )
            return samples

        def get_extra_fields(self):
            """Return the extra fields of the fit.

            Returns:
                Dictionary with no divergent transitions.
            """
            return {"diverging": np.zeros((4, 500), dtype=bool)}

    def fit_and_edit_settings(model_names, **_kwargs):
        """Edit the bounds file during the fit, then return a finished fit.

        Args:
            model_names: Name of the model being fitted.
            **_kwargs: The other fit settings, ignored.

        Returns:
            A stand-in finished fit.
        """
        with open(config_path, "a", encoding="utf-8") as handle:
            handle.write("# edited during the fit\n")
        return FakeMCMC(model_names)

    monkeypatch.setattr(parameter_recovery, "fit_mcmc", fit_and_edit_settings)

    results = run_parameter_recovery(
        models=model_names,
        conditions=["StableV1"],
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        n_subjects=3,
        num_warmup=1,
        num_samples=1,
        num_chains=4,
        seed=7,
        config_path=config_path,
        progress_bar=False,
    )

    assert len(results) == len(model_names)
    for model_idx, (result, model_name) in enumerate(zip(results, model_names)):
        run_dir = Path(result["output_dir"])
        for name in (
            "parameter_recovery.csv",
            "parameter_recovery_summary.csv",
            "run_metadata.json",
            "diagnostics.json",
        ):
            assert (run_dir / name).exists()
        # The standalone runs no longer draw figures. They are made from the
        # CSV files by scripts/make_recovery_figures.py.
        assert not list(run_dir.glob("*.svg"))
        run_metadata = json.loads(
            (run_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        diagnostics = json.loads((run_dir / "diagnostics.json").read_text("utf-8"))
        summary = pd.read_csv(run_dir / "parameter_recovery_summary.csv")
        first_site = f"{model_name}_{summary['parameter'][0]}_subj"

        expected_converged = not shift_one_chain
        assert result["converged"] is expected_converged
        assert diagnostics["converged"] is expected_converged
        assert list(summary["converged"]) == [expected_converged] * len(summary)
        assert run_metadata.pop("max_rhat") == max(diagnostics["rhat"].values())
        assert run_metadata.pop("min_ess") == min(diagnostics["ess"].values())
        assert run_metadata == {
            "condition": "StableV1",
            "model": model_name,
            "n_subjects": 3,
            "seed": 7 + model_idx,
            "converged": expected_converged,
            "divergences": 0,
            "config_path": str(config_path),
            "config_sha256": original_sha256,
        }
        assert list(summary["rhat"]) == [
            diagnostics["rhat"][f"{model_name}_{p}_subj"] for p in summary["parameter"]
        ]
        if shift_one_chain:
            assert diagnostics["rhat"][first_site] > 1.01


def test_simulation_requires_real_outcomes():
    """Simulating without an outcome sequence is an error, not made-up outcomes."""
    missing = "missing 1 required positional argument: 'external_outcomes'"
    with pytest.raises(TypeError, match=missing):
        generate_synthetic_data_single_model("rescorla_wagner_single_value", 2, 8)


def test_pack_true_parameter_array_preserves_parameter_order():
    true_params = {
        "temperature": np.array([1.0, 2.0]),
        "learning_rate": np.array([0.2, 0.8]),
    }

    packed = pack_true_parameter_array(true_params, ["learning_rate", "temperature"])

    assert packed.shape == (2, 2)
    np.testing.assert_allclose(packed[:, 0], [0.2, 0.8])
    np.testing.assert_allclose(packed[:, 1], [1.0, 2.0])


def test_pack_posterior_parameter_samples_returns_sample_subject_parameter_array():
    posterior_samples = {
        "rescorla_wagner_single_value_learning_rate_subj": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "rescorla_wagner_single_value_temperature_subj": np.array([[1.0, 1.5], [2.0, 2.5]]),
    }

    packed = pack_posterior_parameter_samples(
        posterior_samples,
        "rescorla_wagner_single_value",
        ["learning_rate", "temperature"],
    )

    assert packed.shape == (2, 2, 2)
    np.testing.assert_allclose(packed[:, :, 0], posterior_samples["rescorla_wagner_single_value_learning_rate_subj"])
    np.testing.assert_allclose(packed[:, :, 1], posterior_samples["rescorla_wagner_single_value_temperature_subj"])


def test_summarize_parameter_recovery_computes_expected_metrics():
    true_values = np.array([[0.0, 1.0], [1.0, 2.0]])
    recovered_samples = np.array(
        [
            [[0.2, 0.5], [0.8, 2.5]],
            [[0.2, 0.5], [0.8, 2.5]],
        ]
    )

    summary = summarize_parameter_recovery(
        "VolatileV1",
        "rescorla_wagner_single_value",
        ["learning_rate", "temperature"],
        true_values,
        recovered_samples,
    )

    assert list(summary["parameter"]) == ["learning_rate", "temperature"]
    np.testing.assert_allclose(summary["rmse"], [0.2, 0.5])
    np.testing.assert_allclose(summary["bias"], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(summary["pearson_r"], [1.0, 1.0])


def test_build_parameter_recovery_dataframe_contains_true_recovered_and_error_columns():
    true_values = np.array([[0.1, 1.0], [0.3, 2.0]])
    recovered_samples = np.array(
        [
            [[0.2, 1.2], [0.4, 2.4]],
            [[0.2, 1.2], [0.4, 2.4]],
        ]
    )

    frame = build_parameter_recovery_dataframe(
        "StableV1",
        "rescorla_wagner_single_value",
        ["learning_rate", "temperature"],
        true_values,
        recovered_samples,
    )

    assert list(frame.columns) == [
        "condition",
        "model",
        "subject_idx",
        "true_learning_rate",
        "recovered_learning_rate",
        "error_learning_rate",
        "true_temperature",
        "recovered_temperature",
        "error_temperature",
    ]
    np.testing.assert_allclose(frame["error_learning_rate"], [0.1, 0.1])
