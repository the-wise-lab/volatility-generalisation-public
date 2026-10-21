from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from volatility_generalisation.modelling_code.data_generation import load_parameter_bounds
from volatility_generalisation.modelling_code.integrated_model_recovery import (
    build_integrated_group_effect_recovery,
    build_integrated_parameter_recovery,
    build_recovery_job_grid,
    build_recovery_matrices,
    compute_recovery_seed,
    rank_recovery_fits,
    select_recovery_winners,
    simulate_integrated_dataset,
    validate_recovery_fit_grid,
)


GROUP_EFFECTS = (
    "mu",
    "beta_current_vol",
    "beta_previous_vol",
    "beta_current_previous_interaction",
    "subject_sigma",
)


def _write_condition(path: Path, subject_ids: list[str], choices_value: float) -> None:
    n_subjects = len(subject_ids)
    np.savez(
        path,
        outcomes_0=np.tile(np.asarray([[1.0, 0.0, 1.0, 0.0]]), (n_subjects, 1)),
        outcomes_1=np.tile(np.asarray([[0.0, 1.0, 0.0, 1.0]]), (n_subjects, 1)),
        choices=np.full((n_subjects, 4), choices_value),
        subject_ids=np.asarray(subject_ids),
    )


def _load_compile_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compile_integrated_model_recovery.py"
    spec = importlib.util.spec_from_file_location("compile_integrated_model_recovery", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_recovery_fit_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_integrated_model_recovery_fit.py"
    spec = importlib.util.spec_from_file_location("run_integrated_model_recovery_fit", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_recovery_idata(n_trials: int = 4) -> az.InferenceData:
    values = np.arange(1 * 2 * 2 * n_trials, dtype=float).reshape(1, 2, 2, n_trials)
    ll_data = xr.DataArray(values, dims=["chain", "draw", "subject", "trial"])
    return az.InferenceData(log_likelihood=xr.Dataset({"obs": ll_data}))


def _default_group_effects(model: str) -> tuple[dict, dict]:
    """Make true group-level effects and matching posterior sites for a model.

    Args:
        model: Model name. Every parameter in its bounds gets all five effects.

    Returns:
        The ``hierarchy_parameters`` entry, and the posterior sites, each with
        two draws around the true value.
    """
    hierarchy, sites = {}, {}
    for parameter in load_parameter_bounds(model):
        hierarchy[parameter] = {"subject_offset_raw": [0.0, 0.0]}
        for index, effect in enumerate(GROUP_EFFECTS):
            value = 0.1 * (index + 1)
            hierarchy[parameter][effect] = value
            site = f"{model}_{parameter}_{effect}"
            sites[site] = np.asarray([[value - 0.5, value + 0.5]])
    return hierarchy, sites


def _write_fit_artifacts(
    run_dir: Path,
    *,
    replicate: int,
    simulated_model: str,
    fitted_model: str,
    elpd_loo: float,
    converged: bool,
    n_bad_pareto_k: int,
    max_pareto_k: float,
    simulation_true_parameters: dict[str, list[float]] | None = None,
    hierarchy_parameters: dict[str, dict] | None = None,
    posterior_sites: dict[str, np.ndarray] | None = None,
    posterior_log_likelihood: np.ndarray | None = None,
    summary_updates: dict[str, object] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if (
        simulation_true_parameters is not None
        and hierarchy_parameters is None
        and simulated_model == fitted_model
    ):
        # A model fitted to its own data also has true group-level effects, and
        # posterior sites for them.
        hierarchy_parameters, group_sites = _default_group_effects(fitted_model)
        posterior_sites = {**(posterior_sites or {}), **group_sites}
    fit_summary = {
        "replicate": replicate,
        "simulated_model": simulated_model,
        "fitted_model": fitted_model,
        "elpd_loo": elpd_loo,
        "se_loo": 0.1,
        "p_loo": 1.0,
        "loo_rank_within_dataset": None,
        "n_bad_pareto_k": n_bad_pareto_k,
        "max_pareto_k": max_pareto_k,
        "converged": converged,
        "fit_duration": 0.1,
        "pareto_k_threshold": 0.7,
        "output_path": str(run_dir),
        "posterior_path": str(run_dir / "posterior.nc"),
    }
    if summary_updates is not None:
        fit_summary.update(summary_updates)
    with open(run_dir / "fit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(fit_summary, handle, indent=2)

    if simulation_true_parameters is not None:
        subject_count = len(next(iter(simulation_true_parameters.values())))
        simulation_metadata = {
            "simulated_model": simulated_model,
            "simulation_seed": 123,
            "conditions": ["StableV1", "VolatileV2"],
            "hierarchy_parameters": hierarchy_parameters or {},
            "true_parameters": simulation_true_parameters,
        }
        dataset_metadata = {
            "replicate": replicate,
            "simulated_model": simulated_model,
            "fitted_model": fitted_model,
            "conditions": ["StableV1", "VolatileV2"],
            "n_subjects": subject_count,
            "n_conditions": 2,
            "n_trials": 4,
            "subject_ids": ["s1", "s2"],
            "subject_labels": ["Subject 1", "Subject 2"],
            "condition_labels": ["StableV1", "VolatileV2"],
            "context": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "simulation_seed": 123,
            "choice_coding": "1 means option 0 chosen; 0 means option 1 chosen",
            "output_path": str(run_dir),
        }
        with open(run_dir / "simulation_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(simulation_metadata, handle, indent=2)
        with open(run_dir / "dataset_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(dataset_metadata, handle, indent=2)

    log_likelihood = posterior_log_likelihood
    if log_likelihood is None:
        log_likelihood = np.asarray(
            [[[[0.0, 0.1, 0.2, 0.3], [0.05, 0.15, 0.25, 0.35]], [[0.02, 0.12, 0.22, 0.32], [0.07, 0.17, 0.27, 0.37]]]],
            dtype=float,
        )

    log_likelihood_ds = xr.Dataset(
        {"obs": xr.DataArray(log_likelihood, dims=["chain", "draw", "subject", "trial"])},
    )
    if posterior_sites is not None:
        first_site = next(iter(posterior_sites.values()))
        posterior = xr.Dataset(
            {
                site_name: (("chain", "draw", "subject")[: site_values.ndim], site_values)
                for site_name, site_values in posterior_sites.items()
            },
            coords={
                "chain": np.arange(first_site.shape[0]),
                "draw": np.arange(first_site.shape[1]),
                "subject": [0, 1],
            },
        )
    else:
        posterior = xr.Dataset(
            {
                "dummy_posterior": xr.DataArray(
                    np.zeros((log_likelihood_ds["obs"].sizes["chain"], log_likelihood_ds["obs"].sizes["draw"])),
                    dims=["chain", "draw"],
                )
            },
            coords={
                "chain": np.arange(log_likelihood_ds["obs"].sizes["chain"]),
                "draw": np.arange(log_likelihood_ds["obs"].sizes["draw"]),
            },
        )
    az.to_netcdf(az.InferenceData(posterior=posterior, log_likelihood=log_likelihood_ds), run_dir / "posterior.nc")

    return run_dir


def _build_ranking_frame(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        with open(run_dir / "fit_summary.json", "r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["summary_path"] = str(run_dir / "fit_summary.json")
        rows.append(row)
    return rank_recovery_fits(pd.DataFrame(rows))


def _make_posterior_log_likelihood(base: float) -> np.ndarray:
    return base + np.arange(1 * 2 * 2 * 4, dtype=float).reshape(1, 2, 2, 4) * 0.01


def test_load_log_likelihood_idata_reads_only_log_likelihood_group(tmp_path: Path) -> None:
    module = _load_compile_module()
    posterior_path = tmp_path / "posterior.nc"
    values = np.arange(1 * 2 * 2 * 4, dtype=float).reshape(1, 2, 2, 4)
    az.to_netcdf(
        az.InferenceData(
            log_likelihood=xr.Dataset({"obs": xr.DataArray(values, dims=["chain", "draw", "subject", "trial"])}),
        ),
        posterior_path,
    )

    idata = module._load_log_likelihood_idata(posterior_path)

    assert idata.groups() == ["log_likelihood"]
    np.testing.assert_array_equal(idata.log_likelihood["obs"].values, values)


def test_resolve_output_dir_defaults_to_summary() -> None:
    module = _load_compile_module()

    assert module._resolve_output_dir(Path("results/integrated_model_recovery"), None) == Path(
        "results/integrated_model_recovery/summary"
    )


def test_nonempty_output_dir_requires_overwrite(tmp_path: Path) -> None:
    module = _load_compile_module()
    output_dir = tmp_path / "summary"
    output_dir.mkdir()
    (output_dir / "existing.csv").write_text("x\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-empty"):
        module._prepare_output_dir(output_dir, overwrite=False)

    module._prepare_output_dir(output_dir, overwrite=True)
    assert output_dir.exists()


@pytest.mark.parametrize("loo_workers", [0, -1])
def test_compile_integrated_model_recovery_rejects_invalid_loo_workers(
    tmp_path: Path,
    monkeypatch,
    loo_workers: int,
) -> None:
    module = _load_compile_module()
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=tmp_path / "results",
            output_dir=tmp_path / "compiled",
            models=["rescorla_wagner_single_value"],
            replicates=1,
            pareto_k_threshold=0.7,
            loo_drop_initial_trials=1,
            loo_source="auto",
            loo_workers=loo_workers,
            overwrite=False,
        ),
    )

    with pytest.raises(ValueError, match="--loo-workers must be at least 1"):
        module.main()

    assert not (tmp_path / "compiled").exists()


def test_compile_integrated_model_recovery_parallel_worker_failure_includes_row_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "compiled"

    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=1.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=2.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
    )

    module = _load_compile_module()
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["rescorla_wagner_single_value", "dual_lr_single_value"])

    class FakeExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, task):
            future = concurrent.futures.Future()
            row_index, row, n_initial_trials_dropped, pareto_k_threshold = task
            if row["fitted_model"] == "dual_lr_single_value":
                future.set_exception(ValueError("boom"))
            else:
                future.set_result(fn((row_index, row, n_initial_trials_dropped, pareto_k_threshold)))
            return future

    monkeypatch.setattr(module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=results_dir,
            output_dir=output_dir,
            models=["rescorla_wagner_single_value", "dual_lr_single_value"],
            replicates=1,
            pareto_k_threshold=0.7,
            loo_drop_initial_trials=1,
            loo_source="posterior",
            loo_workers=2,
            overwrite=False,
        ),
    )

    with pytest.raises(RuntimeError, match=r"replicate=0, simulated_model=rescorla_wagner_single_value, fitted_model=dual_lr_single_value"):
        module.main()


def test_mask_initial_trials_returns_original_when_n_initial_zero() -> None:
    module = _load_recovery_fit_module()
    idata = _build_recovery_idata()

    masked = module._mask_initial_trials(idata, 0)

    assert masked is idata
    assert masked.log_likelihood["obs"].shape == (1, 2, 2, 4)


def test_mask_initial_trials_drops_first_trial() -> None:
    module = _load_recovery_fit_module()
    values = np.arange(1 * 2 * 2 * 4, dtype=float).reshape(1, 2, 2, 4)
    idata = az.InferenceData(
        log_likelihood=xr.Dataset({"obs": xr.DataArray(values, dims=["chain", "draw", "subject", "trial"])}),
    )

    masked = module._mask_initial_trials(idata, 1)

    assert masked.log_likelihood["obs"].shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(masked.log_likelihood["obs"].values, values[..., 1:])
    np.testing.assert_array_equal(idata.log_likelihood["obs"].values, values)


def test_mask_initial_trials_raises_when_n_initial_ge_n_trials() -> None:
    module = _load_recovery_fit_module()
    idata = _build_recovery_idata(n_trials=5)

    with pytest.raises(ValueError, match="must be less than"):
        module._mask_initial_trials(idata, 5)


def test_mask_initial_trials_raises_when_n_initial_is_negative() -> None:
    module = _load_recovery_fit_module()
    idata = _build_recovery_idata()

    with pytest.raises(ValueError, match="must be non-negative"):
        module._mask_initial_trials(idata, -1)


@pytest.mark.parametrize("n_initial, match", [(-1, "must be non-negative"), (4, "must be less than")])
def test_compile_mask_initial_trials_rejects_invalid_drop_counts(
    n_initial: int,
    match: str,
) -> None:
    module = _load_compile_module()
    idata = _build_recovery_idata()

    with pytest.raises(ValueError, match=match):
        module._mask_initial_trials(idata, n_initial)


@pytest.mark.parametrize(("loo_drop_initial_trials", "expected_trials_used"), [(1, 3), (0, 4)])
def test_main_masks_trials_before_loo_and_records_counts(
    tmp_path: Path,
    monkeypatch,
    loo_drop_initial_trials: int,
    expected_trials_used: int,
) -> None:
    module = _load_recovery_fit_module()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname = 'temp'\n", encoding="utf-8")
    (project_dir / "config").mkdir()
    (project_dir / "config" / "model_bounds.toml").write_text(
        "[common]\n", encoding="utf-8"
    )
    output_dir = tmp_path / "results"
    fake_idata = _build_recovery_idata(n_trials=4)

    class FakeMCMC:
        def get_samples(self, group_by_chain: bool = True):
            return {"rescorla_wagner_single_value_alpha": np.zeros((1, 1))}

    fake_loo_result = SimpleNamespace(
        elpd_loo=1.0,
        se=0.1,
        p_loo=0.2,
        pareto_k=SimpleNamespace(values=np.asarray([0.1, 0.2, 0.3])),
    )
    captured: dict[str, object] = {}

    args = SimpleNamespace(
        config="scripts/config.toml",
        model_config="config/model_bounds.toml",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="dual_lr_single_value",
        models=["rescorla_wagner_single_value", "dual_lr_single_value"],
        conditions=["StableV1", "VolatileV2"],
        data_dir="data/prepared",
        output_dir=str(output_dir),
        n_subjects=2,
        base_seed=42,
        num_warmup=1,
        num_samples=1,
        num_chains=1,
            target_accept_prob=0.95,
            max_tree_depth=10,
            thinning=1,
            pareto_k_threshold=0.7,
            loo_drop_initial_trials=loo_drop_initial_trials,
            overwrite=False,
        )
    data = {
        "simulated_model": "rescorla_wagner_single_value",
        "simulation_seed": 123,
        "conditions": ["StableV1", "VolatileV2"],
        "hierarchy_parameters": {},
        "true_parameters": {"alpha": np.asarray([0.1, 0.2])},
        "choices": np.zeros((2, 4)),
        "subject_ids": ["s1", "s2"],
        "subject_labels": ["Subject 1", "Subject 2"],
        "condition_labels": ["StableV1", "VolatileV2"],
        "context": np.zeros((2, 3)),
    }

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "simulate_integrated_dataset", lambda **_kwargs: data)
    monkeypatch.setattr(module, "fit_integrated_mcmc", lambda **_kwargs: FakeMCMC())
    monkeypatch.setattr(module, "diagnose_mcmc", lambda *_args, **_kwargs: {"converged": True, "divergences": 0})
    monkeypatch.setattr(module, "integrated_samples_to_idata", lambda *_args, **_kwargs: fake_idata)
    monkeypatch.setattr(module.az, "to_netcdf", lambda *_args, **_kwargs: None)

    def fake_loo(idata, pointwise=True):
        captured["idata"] = idata
        captured["pointwise"] = pointwise
        return fake_loo_result

    monkeypatch.setattr(module.az, "loo", fake_loo)
    monkeypatch.chdir(project_dir)

    module.main()

    assert captured["pointwise"] is True
    assert captured["idata"].log_likelihood["obs"].sizes["trial"] == expected_trials_used

    run_dir = module.recovery_output_dir(
        output_dir,
        replicate=args.replicate,
        simulated_model=args.simulated_model,
        fitted_model=args.fitted_model,
    )
    with open(run_dir / "fit_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(run_dir / "diagnostics.json", "r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)

    assert summary["n_initial_trials_dropped"] == loo_drop_initial_trials
    assert summary["n_trials_used"] == expected_trials_used
    assert diagnostics["n_initial_trials_dropped"] == loo_drop_initial_trials
    assert diagnostics["n_trials_used"] == expected_trials_used

    with open(run_dir / "dataset_metadata.json", "r", encoding="utf-8") as handle:
        dataset_metadata = json.load(handle)
    assert dataset_metadata["config_path"] == "config/model_bounds.toml"
    assert (
        dataset_metadata["config_sha256"] == hashlib.sha256(b"[common]\n").hexdigest()
    )


def test_build_recovery_job_grid_covers_all_combinations() -> None:
    jobs = build_recovery_job_grid(2, ["rescorla_wagner_single_value", "dual_lr_single_value"], ["rescorla_wagner_single_value", "dual_lr_single_value"])

    assert len(jobs) == 8
    assert jobs[0] == {
        "replicate": 0,
        "simulated_model": "rescorla_wagner_single_value",
        "fitted_model": "rescorla_wagner_single_value",
    }
    assert jobs[-1] == {
        "replicate": 1,
        "simulated_model": "dual_lr_single_value",
        "fitted_model": "dual_lr_single_value",
    }


def test_compute_recovery_seed_is_deterministic() -> None:
    model_order = ["rescorla_wagner_single_value", "dual_lr_single_value"]

    seed_a = compute_recovery_seed(3, "dual_lr_single_value", model_order, base_seed=11)
    seed_b = compute_recovery_seed(3, "dual_lr_single_value", model_order, base_seed=11)
    seed_c = compute_recovery_seed(3, "rescorla_wagner_single_value", model_order, base_seed=11)

    assert seed_a == seed_b
    assert seed_a != seed_c


def test_simulate_integrated_dataset_is_reproducible_and_bounded(tmp_path: Path) -> None:
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    subject_ids = ["s1", "s2", "s3"]
    _write_condition(data_dir / "StableV1.npz", subject_ids, 0.0)
    _write_condition(data_dir / "VolatileV2.npz", subject_ids, 1.0)

    simulated_a = simulate_integrated_dataset(
        simulated_model="rescorla_wagner_single_value",
        conditions=["StableV1", "VolatileV2"],
        n_subjects=2,
        seed=123,
        data_dir=data_dir,
    )
    simulated_b = simulate_integrated_dataset(
        simulated_model="rescorla_wagner_single_value",
        conditions=["StableV1", "VolatileV2"],
        n_subjects=2,
        seed=123,
        data_dir=data_dir,
    )

    assert np.asarray(simulated_a["choices"]).shape == (2, 4)
    np.testing.assert_allclose(np.asarray(simulated_a["choices"]), np.asarray(simulated_b["choices"]))
    assert sorted(simulated_a["condition_labels"]) == ["StableV1", "VolatileV2"]

    learning_rates = simulated_a["true_parameters"]["learning_rate"]
    temperature = simulated_a["true_parameters"]["temperature"]
    temperature_bounds = load_parameter_bounds("rescorla_wagner_single_value")["temperature"]
    assert np.all((learning_rates >= 0.0) & (learning_rates <= 1.0))
    assert np.all(
        (temperature >= temperature_bounds["min"]) & (temperature <= temperature_bounds["max"])
    )
    assert set(np.unique(np.asarray(simulated_a["choices"]))).issubset({0.0, 1.0})


def test_select_recovery_winners_includes_unconverged_and_high_pareto_k_fits() -> None:
    loo_df = pd.DataFrame(
        [
            {
                "replicate": 0,
                "simulated_model": "rescorla_wagner_single_value",
                "fitted_model": "rescorla_wagner_single_value",
                "elpd_loo": 10.0,
                "converged": True,
                "n_bad_pareto_k": 0,
                "max_pareto_k": 0.2,
            },
            {
                "replicate": 0,
                "simulated_model": "rescorla_wagner_single_value",
                "fitted_model": "dual_lr_single_value",
                "elpd_loo": 11.0,
                "converged": False,
                "n_bad_pareto_k": 2,
                "max_pareto_k": 0.9,
            },
            {
                "replicate": 0,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "rescorla_wagner_single_value",
                "elpd_loo": 3.0,
                "converged": True,
                "n_bad_pareto_k": 1,
                "max_pareto_k": 0.8,
            },
            {
                "replicate": 0,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "dual_lr_single_value",
                "elpd_loo": 2.0,
                "converged": True,
                "n_bad_pareto_k": 0,
                "max_pareto_k": 0.2,
            },
        ]
    )

    ranked = rank_recovery_fits(loo_df)
    winners = select_recovery_winners(loo_df)

    assert "valid_for_recovery" not in ranked.columns
    assert ranked["loo_rank_within_dataset"].tolist() == [2, 1, 1, 2]
    # The highest-elpd fits win even though one did not converge and has
    # Pareto k 0.9, and the other has a Pareto k above 0.7.
    assert winners[
        ["simulated_model", "fitted_model", "converged", "max_pareto_k"]
    ].to_dict("records") == [
        {
            "simulated_model": "dual_lr_single_value",
            "fitted_model": "rescorla_wagner_single_value",
            "converged": True,
            "max_pareto_k": 0.8,
        },
        {
            "simulated_model": "rescorla_wagner_single_value",
            "fitted_model": "dual_lr_single_value",
            "converged": False,
            "max_pareto_k": 0.9,
        },
    ]


@pytest.mark.parametrize("bad_elpd_loo", [float("nan"), None, float("inf")])
def test_rank_recovery_fits_raises_naming_fit_with_non_finite_elpd_loo(
    bad_elpd_loo,
) -> None:
    loo_df = pd.DataFrame(
        [
            {
                "replicate": 3,
                "simulated_model": "rescorla_wagner_single_value",
                "fitted_model": "rescorla_wagner_single_value",
                "elpd_loo": 10.0,
            },
            {
                "replicate": 3,
                "simulated_model": "rescorla_wagner_single_value",
                "fitted_model": "dual_lr_single_value",
                "elpd_loo": bad_elpd_loo,
            },
        ]
    )

    with pytest.raises(ValueError) as error:
        rank_recovery_fits(loo_df)

    message = str(error.value)
    assert "replicate=3" in message
    assert "simulated_model=rescorla_wagner_single_value" in message
    assert "fitted_model=dual_lr_single_value" in message
    with pytest.raises(ValueError, match="fitted_model=dual_lr_single_value"):
        select_recovery_winners(loo_df)


def test_build_recovery_matrices_normalizes_rows() -> None:
    winners_df = pd.DataFrame(
        [
            {"simulated_model": "rescorla_wagner_single_value", "fitted_model": "rescorla_wagner_single_value"},
            {"simulated_model": "rescorla_wagner_single_value", "fitted_model": "dual_lr_single_value"},
            {"simulated_model": "dual_lr_single_value", "fitted_model": "dual_lr_single_value"},
        ]
    )

    confusion_df, inversion_df = build_recovery_matrices(
        winners_df,
        model_order=["rescorla_wagner_single_value", "dual_lr_single_value"],
    )

    np.testing.assert_allclose(confusion_df.sum(axis=1).values, np.asarray([1.0, 1.0]))
    assert confusion_df.loc["rescorla_wagner_single_value", "rescorla_wagner_single_value"] == 0.5
    assert inversion_df.loc["dual_lr_single_value", "dual_lr_single_value"] > 0.0


def test_build_integrated_parameter_recovery_uses_diagonal_fits_only(tmp_path: Path) -> None:
    diagonal_dir = _write_fit_artifacts(
        tmp_path / "replicate_000" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=12.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={
            "learning_rate": [0.1, 0.3],
            "temperature": [1.0, 1.2],
        },
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
            "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
        },
    )
    off_diagonal_dir = _write_fit_artifacts(
        tmp_path / "replicate_000" / "rescorla_wagner_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=5.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={
            "learning_rate": [0.9, 0.8],
            "temperature": [2.0, 1.8],
        },
        posterior_sites={
            "dual_lr_single_value_learning_rate_subj": np.asarray([[[0.9, 0.8], [0.9, 0.8]]]),
            "dual_lr_single_value_temperature_subj": np.asarray([[[2.0, 1.8], [2.0, 1.8]]]),
        },
    )

    ranked_df = _build_ranking_frame([diagonal_dir, off_diagonal_dir])
    rows_df, summary_df = build_integrated_parameter_recovery(ranked_df, results_dir=tmp_path)

    assert set(rows_df["simulated_model"]) == {"rescorla_wagner_single_value"}
    assert set(rows_df["fitted_model"]) == {"rescorla_wagner_single_value"}
    assert len(rows_df) == 4
    assert len(summary_df) == 2


def test_build_integrated_parameter_recovery_aligns_true_and_recovered_values(tmp_path: Path) -> None:
    run_dir = _write_fit_artifacts(
        tmp_path / "replicate_001" / "dual_lr_single_value" / "dual_lr_single_value",
        replicate=1,
        simulated_model="dual_lr_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=11.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.25,
        simulation_true_parameters={
            "alpha": [0.1, 0.3],
            "beta": [1.0, 1.2],
        },
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.3], [0.2, 0.4]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
        },
    )

    ranked_df = _build_ranking_frame([run_dir])
    rows_df, summary_df = build_integrated_parameter_recovery(ranked_df, results_dir=tmp_path)

    alpha_rows = rows_df[rows_df["parameter"] == "alpha"].sort_values("subject_index")
    beta_rows = rows_df[rows_df["parameter"] == "beta"].sort_values("subject_index")

    np.testing.assert_allclose(alpha_rows["true_value"].values, [0.1, 0.3])
    np.testing.assert_allclose(alpha_rows["recovered_mean"].values, [0.1, 0.35])
    np.testing.assert_allclose(alpha_rows["error"].values, [0.0, 0.05])
    np.testing.assert_allclose(beta_rows["true_value"].values, [1.0, 1.2])
    np.testing.assert_allclose(beta_rows["recovered_mean"].values, [1.2, 1.5])
    np.testing.assert_allclose(beta_rows["error"].values, [0.2, 0.3])

    assert set(["converged", "n_bad_pareto_k", "max_pareto_k"]).issubset(
        rows_df.columns
    )
    assert "valid_for_recovery" not in rows_df.columns
    assert "valid_for_recovery" not in summary_df.columns
    assert rows_df["converged"].all()
    assert (rows_df["n_bad_pareto_k"] == 0).all()
    assert np.isclose(rows_df["max_pareto_k"].iloc[0], 0.25)

    alpha_summary = summary_df[summary_df["parameter"] == "alpha"].iloc[0]
    assert np.isclose(alpha_summary["pearson_r"], 1.0)
    assert np.isclose(alpha_summary["rmse"], np.sqrt((0.0**2 + 0.05**2) / 2))
    assert np.isclose(alpha_summary["bias"], 0.025)


def test_build_integrated_parameter_recovery_keeps_unconverged_diagonal_fits_with_diagnostics(
    tmp_path: Path,
) -> None:
    run_dir = _write_fit_artifacts(
        tmp_path / "replicate_003" / "dual_lr_single_value" / "dual_lr_single_value",
        replicate=3,
        simulated_model="dual_lr_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=7.0,
        converged=False,
        n_bad_pareto_k=2,
        max_pareto_k=0.9,
        simulation_true_parameters={
            "alpha": [0.2, 0.4],
            "beta": [1.1, 1.3],
        },
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.2, 0.5], [0.3, 0.6]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.0, 1.2], [1.1, 1.4]]]),
        },
    )

    ranked_df = _build_ranking_frame([run_dir])
    rows_df, summary_df = build_integrated_parameter_recovery(ranked_df, results_dir=tmp_path)

    assert len(rows_df) == 4
    assert len(summary_df) == 2
    assert rows_df["converged"].eq(False).all()
    assert rows_df["n_bad_pareto_k"].eq(2).all()
    assert np.isclose(rows_df["max_pareto_k"].iloc[0], 0.9)
    assert summary_df["converged"].eq(False).all()
    assert summary_df["n_bad_pareto_k"].eq(2).all()
    assert np.isclose(summary_df["max_pareto_k"].iloc[0], 0.9)


@pytest.mark.parametrize(
    ("missing_artifact", "expected_message"),
    [
        ("dataset_metadata", "missing dataset_metadata.json"),
        ("posterior", "missing posterior.nc"),
    ],
)
def test_build_integrated_parameter_recovery_fails_on_partial_diagonal_artifacts(
    tmp_path: Path,
    missing_artifact: str,
    expected_message: str,
) -> None:
    run_dir = tmp_path / "replicate_004" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "fit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "replicate": 4,
                "simulated_model": "rescorla_wagner_single_value",
                "fitted_model": "rescorla_wagner_single_value",
                "elpd_loo": 9.0,
                "se_loo": 0.1,
                "p_loo": 1.0,
                "loo_rank_within_dataset": 1,
                "n_bad_pareto_k": 0,
                "max_pareto_k": 0.2,
                "converged": True,
                "fit_duration": 0.1,
                "pareto_k_threshold": 0.7,
                "output_path": str(run_dir),
                "posterior_path": str(run_dir / "posterior.nc"),
            },
            handle,
            indent=2,
        )
    with open(run_dir / "simulation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "simulated_model": "rescorla_wagner_single_value",
                "simulation_seed": 123,
                "conditions": ["StableV1", "VolatileV2"],
                "hierarchy_parameters": {},
                "true_parameters": {
                    "learning_rate": [0.1, 0.3],
                    "temperature": [1.0, 1.2],
                },
            },
            handle,
            indent=2,
        )
    if missing_artifact == "dataset_metadata":
        pass
    else:
        dataset_metadata = {
            "replicate": 4,
            "simulated_model": "rescorla_wagner_single_value",
            "fitted_model": "rescorla_wagner_single_value",
            "conditions": ["StableV1", "VolatileV2"],
            "n_subjects": 2,
            "n_conditions": 2,
            "n_trials": 4,
            "subject_ids": ["s1", "s2"],
            "subject_labels": ["Subject 1", "Subject 2"],
            "condition_labels": ["StableV1", "VolatileV2"],
            "context": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "simulation_seed": 123,
            "choice_coding": "1 means option 0 chosen; 0 means option 1 chosen",
            "output_path": str(run_dir),
        }
        with open(run_dir / "dataset_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(dataset_metadata, handle, indent=2)
    if missing_artifact == "posterior":
        pass

    ranked_df = _build_ranking_frame([run_dir])

    with pytest.raises(FileNotFoundError, match=expected_message):
        build_integrated_parameter_recovery(ranked_df, results_dir=tmp_path)


def test_build_integrated_parameter_recovery_fails_on_empty_diagonal_folder(
    tmp_path: Path,
) -> None:
    run_dir = _write_fit_artifacts(
        tmp_path / "replicate_000" / "dual_lr_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=7.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
    )
    (run_dir / "posterior.nc").unlink()
    ranked_df = _build_ranking_frame([run_dir])
    assert not (run_dir / "simulation_metadata.json").exists()
    assert not (run_dir / "dataset_metadata.json").exists()

    with pytest.raises(FileNotFoundError) as error:
        build_integrated_parameter_recovery(ranked_df, results_dir=tmp_path)

    assert str(run_dir) in str(error.value)


def test_build_integrated_parameter_recovery_fails_on_missing_posterior_site(tmp_path: Path) -> None:
    run_dir = _write_fit_artifacts(
        tmp_path / "replicate_002" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=2,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=8.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={
            "learning_rate": [0.1, 0.3],
            "temperature": [1.0, 1.2],
        },
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
        },
    )

    ranked_df = _build_ranking_frame([run_dir])

    with pytest.raises(KeyError, match="rescorla_wagner_single_value_temperature_subj"):
        build_integrated_parameter_recovery(ranked_df, results_dir=tmp_path)


def _write_group_effect_fit(
    results_dir: Path,
    *,
    drop_site: str | None = None,
    drop_parameter: str | None = None,
    converged: bool = True,
) -> pd.DataFrame:
    """Write a model's fit to its own data, with a different value per effect.

    Every effect of each parameter has its own true value and its own posterior,
    so an effect read from the wrong site, or paired with the wrong true value,
    changes the result. The posterior of an effect has two chains with two
    draws each. Chain 0 is centred on 0.05 above the true value (draws 1.0
    either side) and chain 1 is 2.0 higher, so reading only one chain gives a
    different mean and interval. A fit of the other model to the same data is
    also written, and must be ignored.

    Args:
        results_dir: Folder for the fits.
        drop_site: Name of a site (without the model and parameter) to leave
            out of the posterior of ``learning_rate``.
        drop_parameter: Name of a parameter to leave out of the saved true
            group-level values.
        converged: Whether the fit is recorded as converged.

    Returns:
        The ranked table of both fits.
    """
    model = "rescorla_wagner_single_value"
    hierarchy = {}
    sites = {}
    for offset, parameter in ((0.0, "learning_rate"), (1.0, "temperature")):
        if parameter == drop_parameter:
            continue
        hierarchy[parameter] = {"subject_offset_raw": [0.0, 0.0]}
        for index, effect in enumerate(GROUP_EFFECTS):
            true_value = offset + 0.1 * (index + 1)
            hierarchy[parameter][effect] = true_value
            if parameter == "learning_rate" and effect == drop_site:
                continue
            centre = true_value + 0.05
            site = f"{model}_{parameter}_{effect}"
            sites[site] = np.asarray(
                [[centre - 1.0, centre + 1.0], [centre + 1.0, centre + 3.0]]
            )
    diagonal = _write_fit_artifacts(
        results_dir / "replicate_000" / model / model,
        replicate=0,
        simulated_model=model,
        fitted_model=model,
        elpd_loo=2.0,
        converged=converged,
        n_bad_pareto_k=3,
        max_pareto_k=0.9,
        simulation_true_parameters={
            "learning_rate": [0.1, 0.3],
            "temperature": [1.0, 1.2],
        },
        hierarchy_parameters=hierarchy,
        posterior_sites=sites,
    )
    other = _write_fit_artifacts(
        results_dir / "replicate_000" / model / "dual_lr_single_value",
        replicate=0,
        simulated_model=model,
        fitted_model="dual_lr_single_value",
        elpd_loo=1.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
    )
    return _build_ranking_frame([diagonal, other])


def test_build_integrated_group_effect_recovery_pairs_each_effect_with_its_own_site(
    tmp_path: Path,
) -> None:
    ranked_df = _write_group_effect_fit(tmp_path, converged=False)

    effects = build_integrated_group_effect_recovery(ranked_df, results_dir=tmp_path)

    # Two parameters with five effects each, from the one fit of the model to
    # its own data. The fit of the other model is ignored.
    assert len(effects) == 10
    assert set(effects["simulated_model"]) == {"rescorla_wagner_single_value"}
    for row in effects.itertuples():
        offset = 0.0 if row.parameter == "learning_rate" else 1.0
        true_value = offset + 0.1 * (GROUP_EFFECTS.index(row.effect) + 1)
        assert row.true_value == pytest.approx(true_value)
        # All four draws are used: they are centre - 1, centre + 1, centre + 1
        # and centre + 3, so their mean is centre + 1 and their 2.5th and 97.5th
        # percentiles are centre - 0.85 and centre + 2.85.
        centre = true_value + 0.05
        assert row.posterior_mean == pytest.approx(centre + 1.0)
        assert row.eti95_lower == pytest.approx(centre - 0.85)
        assert row.eti95_upper == pytest.approx(centre + 2.85)
        # A fit that did not converge is kept, with its diagnostics.
        assert not row.converged
        assert (row.n_bad_pareto_k, row.max_pareto_k) == (3, 0.9)


@pytest.mark.parametrize(
    ("fit_options", "error", "match"),
    [
        (
            {"drop_site": "beta_previous_vol"},
            KeyError,
            "rescorla_wagner_single_value_learning_rate_beta_previous_vol",
        ),
        (
            {"drop_parameter": "temperature"},
            ValueError,
            r"replicate_000.*Missing: \['temperature'\]",
        ),
    ],
)
def test_build_integrated_group_effect_recovery_fails_on_incomplete_fit(
    tmp_path: Path, fit_options: dict, error: type, match: str
) -> None:
    ranked_df = _write_group_effect_fit(tmp_path, **fit_options)

    with pytest.raises(error, match=match):
        build_integrated_group_effect_recovery(ranked_df, results_dir=tmp_path)


def test_validate_recovery_fit_grid_reports_missing_summary(tmp_path: Path, monkeypatch) -> None:
    module = _load_compile_module()
    posterior_run = _write_fit_artifacts(
        tmp_path / "shared" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=1.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
    )

    loo_df = pd.DataFrame(
        [
            {
                "replicate": 0,
                "simulated_model": "rescorla_wagner_single_value",
                "fitted_model": "rescorla_wagner_single_value",
                "posterior_path": str(posterior_run / "posterior.nc"),
            },
            {
                "replicate": 0,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "rescorla_wagner_single_value",
                "posterior_path": str(posterior_run / "posterior.nc"),
            },
            {
                "replicate": 0,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "dual_lr_single_value",
                "posterior_path": str(posterior_run / "posterior.nc"),
            },
        ]
    )

    monkeypatch.setattr(module, "_load_rows", lambda _results_dir: loo_df.to_dict("records"))
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["rescorla_wagner_single_value", "dual_lr_single_value"])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=tmp_path,
            output_dir=None,
            models=None,
            replicates=2,
            pareto_k_threshold=0.7,
            loo_drop_initial_trials=1,
            loo_source="posterior",
            loo_workers=1,
            overwrite=False,
        ),
    )

    with pytest.raises(RuntimeError) as error:
        module.main()

    assert "replicate=1" in str(error.value)
    assert "simulated_model=rescorla_wagner_single_value" in str(error.value)


def test_validate_recovery_fit_grid_reports_replicate_with_no_fits() -> None:
    models = ["rescorla_wagner_single_value", "dual_lr_single_value"]
    loo_df = pd.DataFrame(
        [
            {"replicate": 0, "simulated_model": simulated, "fitted_model": fitted}
            for simulated in models
            for fitted in models
        ]
    )

    missing = validate_recovery_fit_grid(
        loo_df, model_order=models, expected_replicates=2
    )

    assert missing == [
        (1, simulated, fitted) for simulated in models for fitted in models
    ]
    assert (
        validate_recovery_fit_grid(loo_df, model_order=models, expected_replicates=1)
        == []
    )
    # An empty result set is not a reason to skip the check.
    assert (
        len(
            validate_recovery_fit_grid(
                pd.DataFrame(), model_order=models, expected_replicates=1
            )
        )
        == 4
    )


@pytest.mark.parametrize(
    ("extra_fit", "expected_text"),
    [
        (
            {
                "replicate": 1,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "dual_lr_single_value",
            },
            "not expected: (1, 'dual_lr_single_value', 'dual_lr_single_value')",
        ),
        (
            {
                "replicate": 0,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "other_model",
            },
            "not expected: (0, 'dual_lr_single_value', 'other_model')",
        ),
        (
            {
                "replicate": 0,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "dual_lr_single_value",
            },
            "more than one fit: (0, 'dual_lr_single_value', 'dual_lr_single_value')",
        ),
    ],
)
def test_validate_recovery_fit_grid_rejects_fits_outside_the_grid(
    extra_fit: dict[str, object], expected_text: str
) -> None:
    """A leftover replicate, an extra model or a repeated fit must not be ranked."""
    models = ["rescorla_wagner_single_value", "dual_lr_single_value"]
    rows = [
        {"replicate": 0, "simulated_model": simulated, "fitted_model": fitted}
        for simulated in models
        for fitted in models
    ]
    loo_df = pd.DataFrame(rows + [extra_fit])

    with pytest.raises(ValueError) as error:
        validate_recovery_fit_grid(loo_df, model_order=models, expected_replicates=1)

    assert expected_text in str(error.value)


def test_compile_integrated_model_recovery_requires_replicates_option(
    monkeypatch,
) -> None:
    module = _load_compile_module()
    monkeypatch.setattr(sys, "argv", ["compile_integrated_model_recovery.py"])

    with pytest.raises(SystemExit):
        module.parse_args()

    monkeypatch.setattr(
        sys, "argv", ["compile_integrated_model_recovery.py", "--replicates", "3"]
    )
    assert module.parse_args().replicates == 3


@pytest.mark.parametrize(("loo_drop_initial_trials", "expected_trials_used"), [(1, 3), (0, 4)])
def test_compile_integrated_model_recovery_masks_trials_and_records_counts(
    tmp_path: Path,
    monkeypatch,
    loo_drop_initial_trials: int,
    expected_trials_used: int,
) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "compiled"

    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=1.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"learning_rate": [0.1, 0.3], "temperature": [1.0, 1.2]},
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
            "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
        },
        posterior_log_likelihood=_make_posterior_log_likelihood(0.4),
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=9.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        posterior_log_likelihood=_make_posterior_log_likelihood(-0.4),
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "dual_lr_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=8.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        posterior_log_likelihood=_make_posterior_log_likelihood(-0.6),
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "dual_lr_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=2.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"alpha": [0.1, 0.3], "beta": [1.0, 1.2]},
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.5], [0.2, 0.5]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
        },
        posterior_log_likelihood=_make_posterior_log_likelihood(0.2),
    )

    module = _load_compile_module()
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["rescorla_wagner_single_value", "dual_lr_single_value"])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=results_dir,
            output_dir=output_dir,
            models=["rescorla_wagner_single_value", "dual_lr_single_value"],
            replicates=1,
            pareto_k_threshold=100.0,
            loo_drop_initial_trials=loo_drop_initial_trials,
            loo_source="posterior",
            loo_workers=1,
            overwrite=False,
        ),
        )

    module.main()

    loo_df = pd.read_csv(output_dir / "tables" / "model_recovery_loo.csv")
    target_row = loo_df[(loo_df["simulated_model"] == "rescorla_wagner_single_value") & (loo_df["fitted_model"] == "rescorla_wagner_single_value")].iloc[0]
    masked_idata = module._mask_initial_trials(az.from_netcdf(Path(target_row["posterior_path"])), loo_drop_initial_trials)
    expected_loo = az.loo(masked_idata, pointwise=True)
    assert target_row["elpd_loo"] == pytest.approx(float(expected_loo.elpd_loo))
    assert target_row["se_loo"] == pytest.approx(float(expected_loo.se))
    assert target_row["p_loo"] == pytest.approx(float(expected_loo.p_loo))
    assert set(loo_df["n_initial_trials_dropped"]) == {loo_drop_initial_trials}
    assert set(loo_df["n_trials_used"]) == {expected_trials_used}

    with open(output_dir / "model_recovery_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["loo_drop_initial_trials"] == loo_drop_initial_trials
    assert summary["n_trials_used"] == expected_trials_used
    assert summary["loo_workers"] == 1


def test_compile_integrated_model_recovery_recomputes_loo_from_posterior(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "compiled"

    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=1.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"learning_rate": [0.1, 0.3], "temperature": [1.0, 1.2]},
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
            "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
        },
        posterior_log_likelihood=_make_posterior_log_likelihood(0.6),
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=9.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"alpha": [0.1, 0.3], "beta": [1.0, 1.2]},
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.5], [0.2, 0.5]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
        },
        posterior_log_likelihood=_make_posterior_log_likelihood(-0.4),
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "dual_lr_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=8.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"learning_rate": [0.1, 0.3], "temperature": [1.0, 1.2]},
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
            "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
        },
        posterior_log_likelihood=_make_posterior_log_likelihood(-0.5),
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "dual_lr_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=2.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"alpha": [0.1, 0.3], "beta": [1.0, 1.2]},
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.5], [0.2, 0.5]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
        },
        posterior_log_likelihood=_make_posterior_log_likelihood(0.2),
    )

    module = _load_compile_module()
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["rescorla_wagner_single_value", "dual_lr_single_value"])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=results_dir,
            output_dir=output_dir,
            models=["rescorla_wagner_single_value", "dual_lr_single_value"],
            replicates=1,
            pareto_k_threshold=float("inf"),
            loo_drop_initial_trials=1,
            loo_source="posterior",
            loo_workers=1,
            overwrite=False,
        ),
        )

    module.main()

    loo_df = pd.read_csv(output_dir / "tables" / "model_recovery_loo.csv")
    winners_df = pd.read_csv(output_dir / "tables" / "model_recovery_winners.csv")

    for simulated_model in ["dual_lr_single_value", "rescorla_wagner_single_value"]:
        group = loo_df[loo_df["simulated_model"] == simulated_model].sort_values("elpd_loo", ascending=False)
        assert group.iloc[0]["fitted_model"] == simulated_model

    assert winners_df[["simulated_model", "fitted_model"]].to_dict("records") == [
        {"simulated_model": "dual_lr_single_value", "fitted_model": "dual_lr_single_value"},
        {"simulated_model": "rescorla_wagner_single_value", "fitted_model": "rescorla_wagner_single_value"},
    ]


def test_compile_integrated_model_recovery_parallel_matches_serial_recompute(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    serial_output_dir = tmp_path / "serial"
    parallel_output_dir = tmp_path / "parallel"

    run_specs = [
        ("rescorla_wagner_single_value", "rescorla_wagner_single_value", 0.6),
        ("rescorla_wagner_single_value", "dual_lr_single_value", -0.4),
        ("dual_lr_single_value", "rescorla_wagner_single_value", -0.5),
        ("dual_lr_single_value", "dual_lr_single_value", 0.2),
    ]
    for simulated_model, fitted_model, base in run_specs:
        if fitted_model == "rescorla_wagner_single_value":
            simulation_true_parameters = {"learning_rate": [0.1, 0.3], "temperature": [1.0, 1.2]}
            posterior_sites = {
                "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
                "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
            }
        else:
            simulation_true_parameters = {"alpha": [0.1, 0.3], "beta": [1.0, 1.2]}
            posterior_sites = {
                "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.5], [0.2, 0.5]]]),
                "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
            }
        _write_fit_artifacts(
            results_dir / "replicate_000" / simulated_model / fitted_model,
            replicate=0,
            simulated_model=simulated_model,
            fitted_model=fitted_model,
            elpd_loo=base + 1.0,
            converged=True,
            n_bad_pareto_k=0,
            max_pareto_k=0.2,
            simulation_true_parameters=simulation_true_parameters,
            posterior_sites=posterior_sites,
            summary_updates={"n_initial_trials_dropped": 0, "n_trials_used": 4},
            posterior_log_likelihood=_make_posterior_log_likelihood(base),
        )

    module = _load_compile_module()
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["rescorla_wagner_single_value", "dual_lr_single_value"])
    common_args = dict(
        config="scripts/config.toml",
        results_dir=results_dir,
        models=["rescorla_wagner_single_value", "dual_lr_single_value"],
        replicates=1,
        pareto_k_threshold=100.0,
        loo_drop_initial_trials=1,
        loo_source="auto",
        overwrite=False,
    )

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(output_dir=serial_output_dir, loo_workers=1, **common_args),
    )
    module.main()

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(output_dir=parallel_output_dir, loo_workers=2, **common_args),
    )
    module.main()

    serial_loo = pd.read_csv(serial_output_dir / "tables" / "model_recovery_loo.csv")
    parallel_loo = pd.read_csv(parallel_output_dir / "tables" / "model_recovery_loo.csv")
    sort_columns = ["replicate", "simulated_model", "fitted_model"]
    pd.testing.assert_frame_equal(
        serial_loo.sort_values(sort_columns).reset_index(drop=True),
        parallel_loo.sort_values(sort_columns).reset_index(drop=True),
    )

    serial_winners = pd.read_csv(serial_output_dir / "tables" / "model_recovery_winners.csv")
    parallel_winners = pd.read_csv(parallel_output_dir / "tables" / "model_recovery_winners.csv")
    pd.testing.assert_frame_equal(
        serial_winners.sort_values(sort_columns).reset_index(drop=True),
        parallel_winners.sort_values(sort_columns).reset_index(drop=True),
    )


def test_compile_integrated_model_recovery_auto_uses_fresh_summary_without_posteriors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "compiled"

    run_dirs = [
        _write_fit_artifacts(
            results_dir / "replicate_000" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
            replicate=0,
            simulated_model="rescorla_wagner_single_value",
            fitted_model="rescorla_wagner_single_value",
            elpd_loo=6.0,
            converged=True,
            n_bad_pareto_k=0,
            max_pareto_k=0.2,
            summary_updates={"n_initial_trials_dropped": 1, "n_trials_used": 3},
            simulation_true_parameters={
                "learning_rate": [0.1, 0.3],
                "temperature": [1.0, 1.2],
            },
            posterior_sites={
                "rescorla_wagner_single_value_learning_rate_subj": np.asarray(
                    [[[0.2, 0.4], [0.2, 0.6]]]
                ),
                "rescorla_wagner_single_value_temperature_subj": np.asarray(
                    [[[1.1, 1.4], [1.3, 1.6]]]
                ),
            },
        ),
        _write_fit_artifacts(
            results_dir / "replicate_000" / "rescorla_wagner_single_value" / "dual_lr_single_value",
            replicate=0,
            simulated_model="rescorla_wagner_single_value",
            fitted_model="dual_lr_single_value",
            elpd_loo=2.0,
            converged=True,
            n_bad_pareto_k=0,
            max_pareto_k=0.2,
            summary_updates={"n_initial_trials_dropped": 1, "n_trials_used": 3},
        ),
        _write_fit_artifacts(
            results_dir / "replicate_000" / "dual_lr_single_value" / "rescorla_wagner_single_value",
            replicate=0,
            simulated_model="dual_lr_single_value",
            fitted_model="rescorla_wagner_single_value",
            elpd_loo=1.0,
            converged=True,
            n_bad_pareto_k=0,
            max_pareto_k=0.2,
            summary_updates={"n_initial_trials_dropped": 1, "n_trials_used": 3},
        ),
        _write_fit_artifacts(
            results_dir / "replicate_000" / "dual_lr_single_value" / "dual_lr_single_value",
            replicate=0,
            simulated_model="dual_lr_single_value",
            fitted_model="dual_lr_single_value",
            elpd_loo=5.0,
            converged=True,
            n_bad_pareto_k=0,
            max_pareto_k=0.2,
            summary_updates={"n_initial_trials_dropped": 1, "n_trials_used": 3},
            simulation_true_parameters={"alpha": [0.1, 0.3], "beta": [1.0, 1.2]},
            posterior_sites={
                "dual_lr_single_value_alpha_subj": np.asarray(
                    [[[0.0, 0.5], [0.2, 0.5]]]
                ),
                "dual_lr_single_value_beta_subj": np.asarray(
                    [[[1.2, 1.4], [1.2, 1.6]]]
                ),
            },
        ),
    ]
    # Parameter recovery needs every fit of a model to its own data, so only
    # the fits to the other model lose their posteriors.
    for run_dir in (run_dirs[1], run_dirs[2]):
        (run_dir / "posterior.nc").unlink()

    module = _load_compile_module()
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["rescorla_wagner_single_value", "dual_lr_single_value"])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=results_dir,
            output_dir=output_dir,
            models=["rescorla_wagner_single_value", "dual_lr_single_value"],
            replicates=1,
            pareto_k_threshold=0.7,
            loo_drop_initial_trials=1,
            loo_source="auto",
            loo_workers=2,
            overwrite=False,
        ),
    )

    module.main()

    loo_df = pd.read_csv(output_dir / "tables" / "model_recovery_loo.csv")
    winners_df = pd.read_csv(output_dir / "tables" / "model_recovery_winners.csv")

    assert set(loo_df["n_initial_trials_dropped"]) == {1}
    assert set(loo_df["n_trials_used"]) == {3}
    assert winners_df[["simulated_model", "fitted_model"]].to_dict("records") == [
        {"simulated_model": "dual_lr_single_value", "fitted_model": "dual_lr_single_value"},
        {"simulated_model": "rescorla_wagner_single_value", "fitted_model": "rescorla_wagner_single_value"},
    ]


def test_compile_integrated_model_recovery_summary_mode_rejects_stale_summaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"

    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=6.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
    )

    module = _load_compile_module()
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=results_dir,
            output_dir=None,
            models=["rescorla_wagner_single_value"],
            replicates=1,
            pareto_k_threshold=0.7,
            loo_drop_initial_trials=1,
            loo_source="summary",
            loo_workers=1,
            overwrite=False,
        ),
    )

    with pytest.raises(ValueError, match="stale or incomplete"):
        module.main()


def test_compile_integrated_model_recovery_writes_parameter_recovery_outputs(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    output_dir = results_dir / "summary"

    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=12.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={
            "learning_rate": [0.1, 0.3],
            "temperature": [1.0, 1.2],
        },
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
            "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
        },
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=4.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"alpha": [0.1, 0.3], "beta": [1.0, 1.2]},
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.5], [0.2, 0.5]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
        },
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "dual_lr_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=3.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"learning_rate": [0.1, 0.3], "temperature": [1.0, 1.2]},
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
            "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
        },
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "dual_lr_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=11.0,
        converged=False,
        n_bad_pareto_k=2,
        max_pareto_k=0.9,
        simulation_true_parameters={
            "alpha": [0.1, 0.3],
            "beta": [1.0, 1.2],
        },
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.5], [0.2, 0.5]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
        },
    )

    module = _load_compile_module()
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["rescorla_wagner_single_value", "dual_lr_single_value"])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=results_dir,
            output_dir=None,
            models=["rescorla_wagner_single_value", "dual_lr_single_value"],
            replicates=1,
            pareto_k_threshold=0.7,
            loo_drop_initial_trials=1,
            loo_source="posterior",
            loo_workers=1,
            overwrite=False,
        ),
        )

    module.main()

    assert (output_dir / "tables" / "parameter_recovery.csv").exists()
    assert (output_dir / "tables" / "parameter_recovery_summary.csv").exists()
    assert (output_dir / "tables" / "group_effect_recovery.csv").exists()
    assert not (output_dir / "figures").exists()
    with open(output_dir / "model_recovery_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["n_parameter_recovery_rows"] == 8
    assert summary["n_parameter_recovery_summary_rows"] == 4
    # Five effects for each of the 2 + 3 parameters of the two models fitted to
    # their own data.
    assert summary["n_group_effect_rows"] == 25
    group_effects = pd.read_csv(output_dir / "tables" / "group_effect_recovery.csv")
    assert len(group_effects) == 25
    assert summary["n_total_datasets"] == 2
    assert summary["n_winner_datasets"] == 2
    assert "n_no_winner_datasets" not in summary
    # The dual-learning-rate winner did not converge. Pareto k is recomputed
    # from the posteriors, so its count is checked against the winners table.
    winners_df = pd.read_csv(output_dir / "tables" / "model_recovery_winners.csv")
    assert summary["n_winners_not_converged"] == 1
    assert summary["n_winners_with_high_pareto_k"] == int(
        (winners_df["n_bad_pareto_k"] > 0).sum()
    )
    assert summary["winner_rule"] == (
        "Highest elpd_loo among all fitted models for each simulated data set. "
        "Fits that did not converge or have Pareto k above the threshold are included; "
        "their convergence and Pareto k are recorded in model_recovery_loo.csv."
    )
    loo_columns = pd.read_csv(output_dir / "tables" / "model_recovery_loo.csv").columns
    assert {"converged", "n_bad_pareto_k", "max_pareto_k"}.issubset(loo_columns)
    assert "valid_for_recovery" not in loo_columns


def test_compile_integrated_model_recovery_writes_summary_with_all_fits(
    tmp_path: Path, monkeypatch
) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "compiled"

    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=12.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"learning_rate": [0.1, 0.3], "temperature": [1.0, 1.2]},
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
            "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
        },
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "rescorla_wagner_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="rescorla_wagner_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=4.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"alpha": [0.1, 0.3], "beta": [1.0, 1.2]},
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.5], [0.2, 0.5]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
        },
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "dual_lr_single_value" / "rescorla_wagner_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="rescorla_wagner_single_value",
        elpd_loo=3.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"learning_rate": [0.1, 0.3], "temperature": [1.0, 1.2]},
        posterior_sites={
            "rescorla_wagner_single_value_learning_rate_subj": np.asarray([[[0.2, 0.4], [0.2, 0.6]]]),
            "rescorla_wagner_single_value_temperature_subj": np.asarray([[[1.1, 1.4], [1.3, 1.6]]]),
        },
    )
    _write_fit_artifacts(
        results_dir / "replicate_000" / "dual_lr_single_value" / "dual_lr_single_value",
        replicate=0,
        simulated_model="dual_lr_single_value",
        fitted_model="dual_lr_single_value",
        elpd_loo=11.0,
        converged=True,
        n_bad_pareto_k=0,
        max_pareto_k=0.2,
        simulation_true_parameters={"alpha": [0.1, 0.3], "beta": [1.0, 1.2]},
        posterior_sites={
            "dual_lr_single_value_alpha_subj": np.asarray([[[0.0, 0.5], [0.2, 0.5]]]),
            "dual_lr_single_value_beta_subj": np.asarray([[[1.2, 1.4], [1.2, 1.6]]]),
        },
    )

    module = _load_compile_module()
    monkeypatch.setattr(module, "get_default_models", lambda _config: ["rescorla_wagner_single_value", "dual_lr_single_value"])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="scripts/config.toml",
            results_dir=results_dir,
            output_dir=output_dir,
            models=["rescorla_wagner_single_value", "dual_lr_single_value"],
            replicates=1,
            pareto_k_threshold=0.7,
            loo_drop_initial_trials=1,
            loo_source="posterior",
            loo_workers=1,
            overwrite=False,
        ),
    )

    module.main()

    assert (output_dir / "tables" / "parameter_recovery.csv").exists()
    assert (output_dir / "tables" / "parameter_recovery_summary.csv").exists()
    with open(output_dir / "model_recovery_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["n_parameter_recovery_rows"] == 8
    assert summary["n_parameter_recovery_summary_rows"] == 4
    assert summary["n_group_effect_rows"] == 25


def test_summarize_recovery_dataset_counts_reports_datasets_and_winners() -> None:
    module = _load_compile_module()

    ranked_df = pd.DataFrame(
        [
            {
                "replicate": 0,
                "simulated_model": "rescorla_wagner_single_value",
                "fitted_model": "rescorla_wagner_single_value",
                "loo_rank_within_dataset": 1,
            },
            {
                "replicate": 0,
                "simulated_model": "rescorla_wagner_single_value",
                "fitted_model": "dual_lr_single_value",
                "loo_rank_within_dataset": 2,
            },
            {
                "replicate": 0,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "rescorla_wagner_single_value",
                "loo_rank_within_dataset": 2,
            },
            {
                "replicate": 0,
                "simulated_model": "dual_lr_single_value",
                "fitted_model": "dual_lr_single_value",
                "loo_rank_within_dataset": 1,
            },
        ]
    )
    winners_df = ranked_df[ranked_df["loo_rank_within_dataset"] == 1]

    summary = module.summarize_recovery_dataset_counts(ranked_df, winners_df)

    assert summary == {"n_total_datasets": 2, "n_winner_datasets": 2}
