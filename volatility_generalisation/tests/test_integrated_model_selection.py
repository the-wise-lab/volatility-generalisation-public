from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from volatility_generalisation.modelling_code.condition_metadata import build_context_regressors


import jax

from volatility_generalisation.modelling_code.integrated_model_selection import (
    _resolve_parameter_priors,
    fit_integrated_mcmc,
    fit_integrated_svi,
    get_default_conditions,
    integrated_samples_to_idata,
    load_integrated_dataset,
    select_pathfinder_init_params,
    svi_to_init_params,
)


def _load_compile_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compile_integrated_model_selection.py"
    spec = importlib.util.spec_from_file_location("compile_integrated_model_selection", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_context_regressors_matches_expected_codes() -> None:
    regressors = build_context_regressors(["StableV1", "VolatileV2", "VolatileV3_2", "StableV4_2"])

    expected = np.asarray(
        [
            [-0.5, 0.5, -0.25],
            [0.5, -0.5, -0.25],
            [0.5, 0.5, 0.25],
            [-0.5, -0.5, 0.25],
        ]
    )
    np.testing.assert_allclose(regressors, expected)


def test_build_context_regressors_rejects_first_block_conditions() -> None:
    with np.testing.assert_raises(ValueError):
        build_context_regressors(["VolatileV1"])


def test_get_default_conditions_uses_second_block_design(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[fitting]\nconditions=['VolatileV1']\n", encoding="utf-8")

    assert get_default_conditions(config_path) == [
        "StableV1",
        "VolatileV2",
        "VolatileV3_2",
        "StableV4_2",
    ]


def test_load_integrated_dataset_concatenates_subjects_across_conditions(tmp_path: Path) -> None:
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()

    np.savez(
        data_dir / "StableV1.npz",
        outcomes_0=np.arange(12).reshape(3, 4),
        outcomes_1=np.arange(12, 24).reshape(3, 4),
        choices=np.ones((3, 4)),
        subject_ids=np.asarray(["s1", "s2", "s3"]),
    )
    np.savez(
        data_dir / "VolatileV2.npz",
        outcomes_0=np.arange(8).reshape(2, 4),
        outcomes_1=np.arange(8, 16).reshape(2, 4),
        choices=np.zeros((2, 4)),
        subject_ids=np.asarray(["t1", "t2"]),
    )

    data = load_integrated_dataset(["StableV1", "VolatileV2"], data_dir=data_dir)

    assert data["choices"].shape == (5, 4)
    assert data["subject_ids"] == ["s1", "s2", "s3", "t1", "t2"]
    assert data["subject_labels"] == [
        "StableV1__s1",
        "StableV1__s2",
        "StableV1__s3",
        "VolatileV2__t1",
        "VolatileV2__t2",
    ]
    assert data["condition_labels"] == ["StableV1", "StableV1", "StableV1", "VolatileV2", "VolatileV2"]
    np.testing.assert_allclose(
        np.asarray(data["context"]),
        np.asarray(
            [
                [-0.5, 0.5, -0.25],
                [-0.5, 0.5, -0.25],
                [-0.5, 0.5, -0.25],
                [0.5, -0.5, -0.25],
                [0.5, -0.5, -0.25],
            ]
        ),
    )


def test_resolve_parameter_priors_defaults_context_scale_to_mu_scale() -> None:
    prior_spec = _resolve_parameter_priors({"prior_mu_sigma": 0.45})

    assert prior_spec["prior_mu_mean"] == 0.0
    assert prior_spec["prior_mu_sigma"] == 0.45
    assert prior_spec["prior_beta_sigma"] == 0.45
    assert prior_spec["prior_sigma_scale"] == 1.0
    assert prior_spec["prior_offset_sigma"] == 0.5


def test_resolve_parameter_priors_uses_explicit_context_scale() -> None:
    prior_spec = _resolve_parameter_priors(
        {
            "prior_mu_mean": 0.2,
            "prior_mu_sigma": 0.45,
            "prior_beta_sigma": 0.15,
            "prior_sigma_scale": 0.5,
            "prior_offset_sigma": 0.35,
        }
    )

    assert prior_spec == {
        "prior_mu_mean": 0.2,
        "prior_mu_sigma": 0.45,
        "prior_beta_sigma": 0.15,
        "prior_sigma_scale": 0.5,
        "prior_offset_sigma": 0.35,
    }


# --- outcome coding normalisation ---


def _explosion_coded_condition_file(condition_path: Path) -> None:
    """Write an explosion-coded .npz file without outcome-coding metadata."""
    np.savez(
        condition_path,
        outcomes_0=np.ones((2, 4)),
        outcomes_1=np.zeros((2, 4)),
        choices=np.ones((2, 4)),
        subject_ids=np.asarray(["s1", "s2"]),
    )


def _new_condition_file(condition_path: Path) -> None:
    """Write a reward-coded .npz file with outcome-coding metadata."""
    np.savez(
        condition_path,
        outcomes_0=np.ones((2, 4)),
        outcomes_1=np.zeros((2, 4)),
        choices=np.ones((2, 4)),
        subject_ids=np.asarray(["s1", "s2"]),
        outcome_coding="reward_safe",
    )


def test_load_integrated_dataset_file_without_metadata_is_converted(
    tmp_path: Path,
) -> None:
    """An .npz without metadata is converted from explosion to reward coding."""
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    _explosion_coded_condition_file(data_dir / "StableV1.npz")

    data = load_integrated_dataset(["StableV1"], data_dir=data_dir)

    # Files without metadata are explosion-coded: 1 = explosion, 0 = safe
    # After conversion to reward coding: 1 -> 0 (explosion), 0 -> 1 (safe)
    # Input had outcomes_0=1, outcomes_1=0; after flip should be 0, 1
    assert np.allclose(np.asarray(data["outcomes_0"]), 0.0)
    assert np.allclose(np.asarray(data["outcomes_1"]), 1.0)
    # Choices unchanged
    assert np.allclose(np.asarray(data["choices"]), 1.0)


def test_load_integrated_dataset_new_file_not_double_flipped(tmp_path: Path) -> None:
    """New metadata-tagged reward-coded files are not flipped again."""
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    _new_condition_file(data_dir / "StableV1.npz")

    data = load_integrated_dataset(["StableV1"], data_dir=data_dir)

    # Already reward-coded: 1 = safe, 0 = explosion
    assert np.allclose(np.asarray(data["outcomes_0"]), 1.0)
    assert np.allclose(np.asarray(data["outcomes_1"]), 0.0)
    assert np.allclose(np.asarray(data["choices"]), 1.0)


# --- log-likelihood ---


def test_integrated_samples_to_idata_uses_log_lik_from_posterior_fallback(monkeypatch) -> None:
    class FakeMCMC:
        def get_samples(self, group_by_chain: bool = False):
            assert group_by_chain is True
            return {
                "log_lik": np.zeros((1, 2, 4)),
                "dual_lr_single_value_temperature_mu": np.zeros((1, 2)),
            }

    def fake_from_numpyro(mcmc, log_likelihood=None):
        assert log_likelihood is False
        return az.InferenceData(posterior=xr.Dataset({"dummy": (("chain", "draw"), np.zeros((1, 2)))}))

    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.az.from_numpyro",
        fake_from_numpyro,
    )

    idata = integrated_samples_to_idata(FakeMCMC())

    assert idata.log_likelihood["obs"].shape == (1, 2, 4)


def test_integrated_samples_to_idata_with_post_hoc_log_lik(monkeypatch) -> None:
    class FakeMCMC:
        def get_samples(self, group_by_chain: bool = False):
            assert group_by_chain is True
            return {"dual_lr_single_value_temperature_subj": np.zeros((1, 2, 4))}

    def fake_compute(*_args, **_kwargs):
        return np.zeros((1, 2, 4, 10))

    def fake_from_numpyro(mcmc, log_likelihood=None):
        assert log_likelihood is False
        return az.InferenceData(posterior=xr.Dataset({"dummy": (("chain", "draw"), np.zeros((1, 2)))}))

    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.compute_trial_log_likelihood",
        fake_compute,
    )
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.az.from_numpyro",
        fake_from_numpyro,
    )

    data = {"outcomes_0": np.zeros((4, 10)), "outcomes_1": np.zeros((4, 10)), "choices": np.zeros((4, 10))}
    idata = integrated_samples_to_idata(FakeMCMC(), model_name="dual_lr_single_value", data=data)

    assert idata.log_likelihood["obs"].shape == (1, 2, 4, 10)


def test_compute_trial_log_likelihood_uses_reward_coded_outcomes(monkeypatch) -> None:
    def fake_run_single_subject_scan(model_name, params, outcomes_0, outcomes_1, choices):
        return outcomes_0

    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.run_single_subject_scan",
        fake_run_single_subject_scan,
    )

    class FakeMCMC:
        def get_samples(self, group_by_chain: bool = False):
            assert group_by_chain is True
            return {
                "rescorla_wagner_single_value_temperature_subj": np.zeros((1, 1, 1)),
            }

    data = {
        "outcomes_0": np.array([[1.0, 0.0]]),
        "outcomes_1": np.array([[0.0, 1.0]]),
        "choices": np.array([[1.0, 0.0]]),
    }

    from volatility_generalisation.modelling_code.integrated_model_selection import (
        compute_trial_log_likelihood,
    )

    result = compute_trial_log_likelihood(FakeMCMC(), "rescorla_wagner_single_value", data)

    assert result.shape == (1, 1, 1, 2)
    assert np.all(result > -1e-3)


def test_resolve_models_returns_explicit_models_unchanged() -> None:
    from volatility_generalisation.modelling_code.integrated_model_selection import (
        get_default_models,
    )

    module = _load_compile_module()
    default_names = get_default_models()
    explicit = ["rescorla_wagner_single_value"]

    result = module._resolve_models(default_names, explicit)
    assert result == explicit


def test_resolve_models_returns_all_default_models() -> None:
    module = _load_compile_module()
    default_names = [
        "rescorla_wagner_single_value",
        "dual_lr_single_value",
    ]

    result = module._resolve_models(default_names, None)

    assert result == ["rescorla_wagner_single_value", "dual_lr_single_value"]


def test_resolve_output_dir_uses_explicit_path() -> None:
    module = _load_compile_module()

    result = module._resolve_output_dir(Path("/results"), Path("/custom"))

    assert result == Path("/custom")


def test_resolve_output_dir_uses_model_comparison_by_default() -> None:
    module = _load_compile_module()

    result = module._resolve_output_dir(Path("/results"), None)

    assert result == Path("/results/model_comparison")


def test_nonempty_output_dir_requires_overwrite(tmp_path: Path) -> None:
    module = _load_compile_module()
    output_dir = tmp_path / "model_comparison"
    output_dir.mkdir()
    (output_dir / "existing.csv").write_text("x\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-empty"):
        module._prepare_output_dir(output_dir, overwrite=False)

    module._prepare_output_dir(output_dir, overwrite=True)
    assert output_dir.exists()


def test_save_pareto_table_uses_precomputed_loo_rows(tmp_path: Path) -> None:
    module = _load_compile_module()

    loo_rows = [
        {
            "model": "model_a",
            "n_pareto_k_gt_0_5": 2,
            "n_pareto_k_gt_0_7": 1,
            "max_pareto_k": 0.85,
        },
        {
            "model": "model_b",
            "n_pareto_k_gt_0_5": 0,
            "n_pareto_k_gt_0_7": 0,
            "max_pareto_k": 0.30,
        },
    ]

    module._save_pareto_table(loo_rows, tmp_path)

    saved = pd.read_csv(tmp_path / "pareto_k_table.csv")
    assert list(saved["model"]) == ["model_a", "model_b"]
    assert list(saved["pareto_k_gt_0_7"]) == [1, 0]
    assert list(saved["pareto_k_max"]) == [0.85, 0.30]


def test_mask_initial_trials_returns_original_when_n_initial_zero() -> None:
    module = _load_compile_module()
    ll_data = xr.DataArray(
        np.arange(24).reshape(1, 2, 12),
        dims=["chain", "draw", "trial"],
    )
    idata = az.InferenceData(
        log_likelihood=xr.Dataset({"obs": ll_data})
    )
    masked = module._mask_initial_trials(idata, 0)
    assert masked is idata
    assert masked.log_likelihood["obs"].shape == (1, 2, 12)


def test_mask_initial_trials_drops_first_trial() -> None:
    module = _load_compile_module()
    values = np.arange(24).reshape(1, 2, 12).astype(float)
    ll_data = xr.DataArray(values, dims=["chain", "draw", "trial"])
    idata = az.InferenceData(
        log_likelihood=xr.Dataset({"obs": ll_data})
    )
    masked = module._mask_initial_trials(idata, 1)
    assert masked.log_likelihood["obs"].shape == (1, 2, 11)
    # Original is unchanged
    assert idata.log_likelihood["obs"].shape == (1, 2, 12)
    # First trial (value 0) should be gone, second trial (value 1) is now first
    np.testing.assert_array_equal(
        masked.log_likelihood["obs"].values[0, 0, :],
        values[0, 0, 1:],
    )


def test_mask_initial_trials_drops_multiple_trials() -> None:
    module = _load_compile_module()
    ll_data = xr.DataArray(
        np.zeros((1, 1, 20)), dims=["chain", "draw", "trial"]
    )
    idata = az.InferenceData(
        log_likelihood=xr.Dataset({"obs": ll_data})
    )
    masked = module._mask_initial_trials(idata, 3)
    assert masked.log_likelihood["obs"].shape == (1, 1, 17)


def test_mask_initial_trials_raises_when_n_initial_ge_n_trials() -> None:
    module = _load_compile_module()
    ll_data = xr.DataArray(
        np.zeros((1, 1, 5)), dims=["chain", "draw", "trial"]
    )
    idata = az.InferenceData(
        log_likelihood=xr.Dataset({"obs": ll_data})
    )
    with pytest.raises(ValueError, match="must be less than"):
        module._mask_initial_trials(idata, 5)


def test_mask_initial_trials_raises_when_n_initial_exceeds_n_trials() -> None:
    module = _load_compile_module()
    ll_data = xr.DataArray(
        np.zeros((1, 1, 5)), dims=["chain", "draw", "trial"]
    )
    idata = az.InferenceData(
        log_likelihood=xr.Dataset({"obs": ll_data})
    )
    with pytest.raises(ValueError, match="must be less than"):
        module._mask_initial_trials(idata, 10)


def test_model_selection_loo_row_exposes_reliability_fields(monkeypatch) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compile_integrated_model_selection.py"
    spec = importlib.util.spec_from_file_location("compile_integrated_model_selection", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeParetoK:
        values = np.asarray([0.2, 0.8])

    class FakeLooResult:
        elpd_loo = 12.5
        se = 0.5
        p_loo = 3.0
        pareto_k = FakeParetoK()

    monkeypatch.setattr(module.az, "loo", lambda *_args, **_kwargs: FakeLooResult())

    row = module.build_loo_row(
        "dual_lr_single_value",
        {"diagnostics": {"converged": True, "divergences": 2}},
        FakeLooResult(),
    )

    assert row["converged"] is True
    assert row["divergences"] == 2
    assert row["n_pareto_k_gt_0_5"] == 1
    assert row["n_pareto_k_gt_0_7"] == 1
    assert row["max_pareto_k"] == 0.8
    assert row["n_trials_used"] == 2
    assert row["n_initial_trials_dropped"] == 0


def test_build_loo_row_exposes_trial_count_fields(monkeypatch) -> None:
    module = _load_compile_module()

    class FakeParetoK:
        values = np.asarray([0.2, 0.5, 0.9])

    class FakeLooResult:
        elpd_loo = -100.0
        se = 5.0
        p_loo = 10.0
        pareto_k = FakeParetoK()

    row = module.build_loo_row(
        "test_model",
        {"diagnostics": {"converged": True, "divergences": 0}},
        FakeLooResult(),
        n_initial_trials_dropped=3,
    )

    assert row["n_trials_used"] == 3
    assert row["n_initial_trials_dropped"] == 3


def test_model_selection_compile_reports_missing_expected_model(tmp_path: Path, monkeypatch) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compile_integrated_model_selection.py"
    spec = importlib.util.spec_from_file_location("compile_integrated_model_selection", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeParetoK:
        values = np.asarray([0.2])

    class FakeLooResult:
        elpd_loo = 1.5
        se = 0.1
        p_loo = 0.3
        pareto_k = FakeParetoK()

    monkeypatch.setattr(module.az, "loo", lambda *_args, **_kwargs: FakeLooResult())
    monkeypatch.setattr(
        module,
        "load_fit",
        lambda result_dir: (
            {"diagnostics": {"converged": True, "divergences": 0}},
            SimpleNamespace(result_dir=result_dir),
        )
        if result_dir.name == "model_a"
        else (_ for _ in ()).throw(FileNotFoundError(f"missing fit for {result_dir.name}")),
    )
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["model_a", "missing_model"])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=tmp_path,
            output_dir=tmp_path,
            models=None,
            include_waic=False,
            loo_drop_initial_trials=0,
            overwrite=False,
        ),
    )

    with np.testing.assert_raises(FileNotFoundError) as error:
        module.main()

    assert "missing_model" in str(error.exception)


def test_model_selection_compile_compares_precomputed_loo_results(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_compile_module()

    class FakeParetoK:
        def __init__(self, values) -> None:
            self.values = np.asarray(values)

    class FakeLooResult:
        def __init__(self, elpd_loo: float, se: float, p_loo: float, pareto_values) -> None:
            self.elpd_loo = elpd_loo
            self.se = se
            self.p_loo = p_loo
            self.pareto_k = FakeParetoK(pareto_values)

    idata_by_model = {
        "model_a": SimpleNamespace(name="idata_a"),
        "model_b": SimpleNamespace(name="idata_b"),
    }
    loo_by_model = {
        "model_a": FakeLooResult(10.0, 0.2, 1.5, [0.1, 0.2]),
        "model_b": FakeLooResult(9.5, 0.3, 1.6, [0.3, 0.4]),
    }

    monkeypatch.setattr(
        module,
        "load_fit",
        lambda result_dir: (
            {"diagnostics": {"converged": True, "divergences": 0}},
            idata_by_model[result_dir.name],
        ),
    )
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["model_a", "model_b"])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=tmp_path,
            output_dir=tmp_path,
            models=None,
            include_waic=False,
            loo_drop_initial_trials=0,
            overwrite=False,
        ),
    )
    monkeypatch.setattr(
        module.az,
        "loo",
        lambda idata, **_kwargs: loo_by_model[
            next(model_name for model_name, model_idata in idata_by_model.items() if model_idata is idata)
        ],
    )

    compare_calls = []

    def fake_compare(compare_dict, **kwargs):
        compare_calls.append((compare_dict, kwargs))
        assert compare_dict == loo_by_model
        assert all(not hasattr(value, "name") for value in compare_dict.values())
        return pd.DataFrame(
            {
                "rank": [0, 1],
                "elpd_loo": [10.0, 9.5],
                "p_loo": [1.5, 1.6],
                "weight": [0.7, 0.3],
            },
            index=["model_a", "model_b"],
        )

    monkeypatch.setattr(module.az, "compare", fake_compare)

    module.main()

    assert len(compare_calls) == 1


# --- select_pathfinder_init_params ---


def test_select_pathfinder_init_params_multi_chain() -> None:
    rng = np.random.default_rng(0)
    samples = {
        "mu": rng.normal(size=(10,)),
        "offset": rng.normal(size=(10, 5)),
    }
    result = select_pathfinder_init_params(samples, num_chains=4, seed=42)
    assert list(result.keys()) == ["mu", "offset"]
    assert result["mu"].shape == (4,)
    assert result["offset"].shape == (4, 5)


def test_select_pathfinder_init_params_single_chain() -> None:
    rng = np.random.default_rng(0)
    samples = {"mu": rng.normal(size=(5,))}
    result = select_pathfinder_init_params(samples, num_chains=1, seed=42)
    assert result["mu"].shape == (1,)


def test_select_pathfinder_init_params_reuses_with_replacement() -> None:
    rng = np.random.default_rng(0)
    samples = {"mu": rng.normal(size=(2,))}
    result = select_pathfinder_init_params(samples, num_chains=4, seed=42)
    assert result["mu"].shape == (4,)  # samples with replacement


def test_select_pathfinder_init_params_empty_samples() -> None:
    result = select_pathfinder_init_params({}, num_chains=3, seed=0)
    assert result == {}


def test_select_pathfinder_init_params_deterministic_seed() -> None:
    rng = np.random.default_rng(0)
    samples = {"x": rng.normal(size=(10,))}
    r1 = select_pathfinder_init_params(samples, num_chains=3, seed=123)
    r2 = select_pathfinder_init_params(samples, num_chains=3, seed=123)
    np.testing.assert_equal(r1["x"], r2["x"])


# --- fit_integrated_mcmc init_params forwarding ---


class _FakeKernel:
    def __init__(self, *args, **kwargs):
        pass


class _RecordingMCMC:
    def __init__(self, *args, **kwargs):
        self.recorded_init_params = None

    def run(self, rng_key, **kwargs):
        self.recorded_init_params = kwargs.pop("init_params", None)

    def get_samples(self, group_by_chain=False):
        return {}

    def get_extra_fields(self):
        return {}

    @property
    def num_chains(self):
        return 1


def _small_data() -> dict:
    return {
        "choices": np.zeros((3, 5)),
        "outcomes_0": np.zeros((3, 5)),
        "outcomes_1": np.zeros((3, 5)),
        "context": np.zeros((3, 3)),
    }


def test_fit_integrated_mcmc_default_passes_no_init_params(monkeypatch) -> None:
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.NUTS",
        _FakeKernel,
    )
    recorder = _RecordingMCMC()
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.MCMC",
        lambda *a, **kw: recorder,
    )

    fit_integrated_mcmc(
        data=_small_data(),
        model_name="dual_lr_single_value",
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=0,
    )

    assert recorder.recorded_init_params is None


def test_fit_integrated_mcmc_forwards_init_params(monkeypatch) -> None:
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.NUTS",
        _FakeKernel,
    )
    recorder = _RecordingMCMC()
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.MCMC",
        lambda *a, **kw: recorder,
    )

    dummy_init = {"mu": np.array([0.0])}
    fit_integrated_mcmc(
        data=_small_data(),
        model_name="dual_lr_single_value",
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=0,
        init_params=dummy_init,
    )

    assert recorder.recorded_init_params is dummy_init


# --- svi_to_init_params ---


_JITTER_TEST_MODEL = "dual_lr_single_value"
_JITTER_TEST_PARAMETERS = ["temperature", "learning_rate_pos", "learning_rate_neg"]
# The group location of the positive learning rate that the fitted guide
# "reports". Its sampler-scale value is the same number (see below).
_SUPPLIED_GROUP_LOCATION = 0.3
_SUPPLIED_SUBJECT_SIGMA = 2.0
_SUPPLIED_SUBJECT_OFFSETS = np.array([0.1, -0.2, 0.3])


def _guide_values_for_every_latent_site() -> dict:
    """Give one posterior draw for every latent site of the test model.

    The names are the real latent sites of ``dual_lr_single_value``: for each
    of its three parameters, a group location, three volatility weights, a
    spread between participants and one offset per participant (three
    participants in the small test data).

    Returns:
        Dictionary from site name to the value of one posterior draw.
    """
    values = {}
    for parameter in _JITTER_TEST_PARAMETERS:
        prefix = f"{_JITTER_TEST_MODEL}_{parameter}"
        values[f"{prefix}_mu"] = np.asarray(0.0)
        values[f"{prefix}_beta_current_vol"] = np.asarray(0.1)
        values[f"{prefix}_beta_previous_vol"] = np.asarray(-0.1)
        values[f"{prefix}_beta_current_previous_interaction"] = np.asarray(0.05)
        values[f"{prefix}_subject_sigma"] = np.asarray(_SUPPLIED_SUBJECT_SIGMA)
        values[f"{prefix}_subject_offset_raw"] = _SUPPLIED_SUBJECT_OFFSETS
    values[f"{_JITTER_TEST_MODEL}_learning_rate_pos_mu"] = np.asarray(
        _SUPPLIED_GROUP_LOCATION
    )
    return values


def _fake_guide():
    """Create a minimal stand-in guide that returns the same draw for every chain.

    Returns:
        An object whose ``sample_posterior`` gives the values from
        ``_guide_values_for_every_latent_site``, repeated once for each draw
        asked for.
    """
    draw = _guide_values_for_every_latent_site()

    class _FakeGuide:
        def sample_posterior(self, rng_key, params, sample_shape=()):
            n = int(np.prod(sample_shape)) if sample_shape else 1
            return {
                name: jnp.broadcast_to(value, (n,) + value.shape)
                for name, value in draw.items()
            }

    return _FakeGuide()


def test_svi_to_init_params_jitter_produces_variation() -> None:
    """Without jitter the guide's values arrive unchanged; with jitter they move.

    The stand-in guide supplies a value for every latent site. The group
    location and the offsets are on the sampler's own scale already (their
    priors are Normal), and the spread between participants is a positive
    value that the sampler works with on the log scale. So with no jitter the
    result must hold the supplied location, the supplied offsets and the log of
    the supplied spread, for every chain.
    """
    expected_sites = set(_guide_values_for_every_latent_site())
    location_site = f"{_JITTER_TEST_MODEL}_learning_rate_pos_mu"
    sigma_site = f"{_JITTER_TEST_MODEL}_learning_rate_pos_subject_sigma"
    offset_site = f"{_JITTER_TEST_MODEL}_learning_rate_pos_subject_offset_raw"
    num_chains = 2

    def convert(jitter_scale):
        """Turn the stand-in guide into starting values, with the given jitter."""
        return svi_to_init_params(
            data=_small_data(),
            model_name=_JITTER_TEST_MODEL,
            guide=_fake_guide(),
            params={},
            num_chains=num_chains,
            seed=0,
            jitter_scale=jitter_scale,
        )

    result_no_jitter = convert(0.0)
    result_with_jitter = convert(0.05)

    for result in (result_no_jitter, result_with_jitter):
        # Every latent site is present, each with a leading chain axis.
        assert set(result) == expected_sites
        for site, value in result.items():
            assert value.shape[0] == num_chains, f"{site} has shape {value.shape}"
        assert result[offset_site].shape == (num_chains, 3)

    # No jitter: the supplied values reach the result, in every chain.
    np.testing.assert_allclose(
        result_no_jitter[location_site], [_SUPPLIED_GROUP_LOCATION] * num_chains,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result_no_jitter[offset_site],
        np.tile(_SUPPLIED_SUBJECT_OFFSETS, (num_chains, 1)),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result_no_jitter[sigma_site],
        [np.log(_SUPPLIED_SUBJECT_SIGMA)] * num_chains,
        rtol=1e-5,
    )

    # Small jitter (standard deviation 0.05): the location moves away from the
    # supplied value but stays close to it, and each chain moves differently so
    # the chains start apart.
    shift = result_with_jitter[location_site] - result_no_jitter[location_site]
    assert np.max(np.abs(shift)) > 1e-3
    assert np.max(np.abs(shift)) < 0.5
    assert shift[0] != shift[1]


def test_reparam_fit_integrated_mcmc_runs(monkeypatch) -> None:
    class _FakeKernel:
        def __init__(self, *args, **kwargs): pass

    class _RecordingMCMC:
        def __init__(self, *args, **kwargs):
            self.recorded_model_name = None
        def run(self, rng_key, **kwargs):
            self.recorded_model_name = kwargs.get("model_name")
        def get_samples(self, group_by_chain=False): return {}
        def get_extra_fields(self): return {}
        @property
        def num_chains(self): return 1

    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.NUTS",
        _FakeKernel,
    )
    recorder = _RecordingMCMC()
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.integrated_model_selection.MCMC",
        lambda *a, **kw: recorder,
    )
    data = {
        "choices": np.zeros((3, 5)),
        "outcomes_0": np.zeros((3, 5)),
        "outcomes_1": np.zeros((3, 5)),
        "context": np.zeros((3, 3)),
    }
    fit_integrated_mcmc(
        data=data,
        model_name="leaky_beta_asymmetric_single_value",
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=0,
        config_path=Path(__file__).resolve().parents[2] / "config" / "model_bounds.toml",
    )
    assert recorder.recorded_model_name == "leaky_beta_asymmetric_single_value"


def _load_selection_fit_module():
    """Load ``scripts/run_integrated_model_selection.py`` as a module.

    Returns:
        The loaded script module.
    """
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_integrated_model_selection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_integrated_model_selection", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_summary_reports_the_95_percent_interval_and_convergence(
    tmp_path: Path,
) -> None:
    """The group summary has the 95% equal-tailed interval, not a 94% HDI.

    The draws are skewed and come from two chains, so an HDI, or an interval
    from one chain only, would differ from the percentiles of the pooled draws.

    Args:
        tmp_path: Folder in which the summary is saved.
    """
    draws = np.random.default_rng(1).exponential(size=(2, 300))
    idata = az.from_dict(posterior={"model_mu": draws, "model_x_subj": draws})

    _load_selection_fit_module()._write_group_summaries(idata, "model", tmp_path)

    summary = pd.read_csv(tmp_path / "group_parameter_summary.csv", index_col=0)
    assert list(summary.index) == ["model_mu"]
    expected_columns = {"mean", "sd", "ci_lower", "ci_upper"}
    expected_columns |= {"mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"}
    assert expected_columns <= set(summary.columns)
    assert not any(column.startswith("hdi_") for column in summary.columns)
    # The table is unrounded, so the interval ends match to full precision.
    lower, upper = np.percentile(draws.ravel(), [2.5, 97.5])
    assert summary.loc["model_mu", "ci_lower"] == pytest.approx(lower)
    assert summary.loc["model_mu", "ci_upper"] == pytest.approx(upper)


def test_model_selection_main_records_the_settings_file(
    tmp_path: Path, monkeypatch
) -> None:
    """The saved metadata names the bounds file the fit used, with its hash.

    The stand-in fit edits the file, to check that the hash is of the file as
    the fit read it, not as it was when the results were saved.
    """
    module = _load_selection_fit_module()
    config_path = tmp_path / "model_bounds.toml"
    config_path.write_text("[common]\n", encoding="utf-8")
    args = SimpleNamespace(
        config="scripts/config.toml",
        model_config=str(config_path),
        model="rescorla_wagner_single_value",
        conditions=["StableV1", "VolatileV2"],
        data_dir=str(tmp_path / "prepared"),
        output_dir=str(tmp_path / "results"),
        pathfinder_init=False,
        svi_init=False,
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        seed=0,
        target_accept_prob=0.95,
        max_tree_depth=10,
        thinning=1,
        loo_drop_initial_trials=0,
    )
    data = {
        "conditions": ["StableV1", "VolatileV2"],
        "subject_ids": ["s1", "s2"],
        "subject_labels": ["Subject 1", "Subject 2"],
        "condition_labels": ["StableV1", "VolatileV2"],
        "choices": np.zeros((2, 4)),
        "context": np.zeros((2, 3)),
    }

    class FakeMCMC:
        """Stand-in for a finished fit."""

        def get_samples(self, group_by_chain: bool = True):
            """Return one posterior sample.

            Args:
                group_by_chain: Ignored.

            Returns:
                Dictionary with one site.
            """
            return {"rescorla_wagner_single_value_alpha": np.zeros((1, 1))}

    fake_idata = SimpleNamespace(log_likelihood={"obs": np.zeros((1, 1, 2, 4))})
    fake_loo = SimpleNamespace(elpd_loo=1.0, p_loo=0.2, se=0.1)

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_integrated_dataset", lambda **_kwargs: data)
    def fit_and_edit_settings(**_kwargs):
        """Edit the bounds file during the fit, then return a finished fit.

        Args:
            **_kwargs: The fit settings, ignored.

        Returns:
            A stand-in finished fit.
        """
        config_path.write_text("[common]\nedited = true\n", encoding="utf-8")
        return FakeMCMC()

    monkeypatch.setattr(module, "fit_integrated_mcmc", fit_and_edit_settings)
    monkeypatch.setattr(
        module, "diagnose_mcmc", lambda *_args, **_kwargs: {"converged": True}
    )
    monkeypatch.setattr(
        module, "integrated_samples_to_idata", lambda *_args, **_kwargs: fake_idata
    )
    monkeypatch.setattr(module.az, "loo", lambda *_args, **_kwargs: fake_loo)
    monkeypatch.setattr(module.az, "to_netcdf", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_write_group_summaries", lambda *_args: None)
    monkeypatch.setattr(module, "_write_subject_summaries", lambda *_args: None)
    monkeypatch.setattr(module, "_write_posterior_summary_plot", lambda *_args: None)

    module.main()

    metadata_path = (
        tmp_path / "results" / "rescorla_wagner_single_value" / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["config_path"] == str(config_path)
    assert metadata["config_sha256"] == hashlib.sha256(b"[common]\n").hexdigest()
