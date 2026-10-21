#!/usr/bin/env python3
"""Fit one candidate model to one integrated simulated dataset.

This is one step of the model-recovery analysis. It simulates a dataset from a
known model, fits a candidate model to it, and saves the leave-one-out
cross-validation (LOO) scores and diagnostics. Each SLURM array task runs it once
for one replicate, one simulated model and one fitted model. The results are later
gathered by compile_integrated_model_recovery.py. The results are saved even if
the chains did not converge, and the script still ends with exit code 0, so SLURM
reports the task as completed.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import xarray as xr


def _resolve_host_device_count() -> int:
    """Work out how many CPU cores the fit can use, on SLURM or on a local machine.

    Uses the number of CPUs SLURM gave to the task if that is set and valid.
    Otherwise uses the number of CPUs on this machine. The result is never
    below 1.

    Returns:
        Number of CPU cores to use for running MCMC chains.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    local_cpus = os.cpu_count()
    return max(1, local_cpus or 1)


HOST_DEVICE_COUNT = _resolve_host_device_count()
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import arviz as az
import jax
import numpy as np
import numpyro
from rich.console import Console

from volatility_generalisation.modelling_code.diagnostics import (
    diagnose_mcmc,
    print_chain_failure_summary,
)
from volatility_generalisation.modelling_code.data_generation import (
    settings_file_record,
)
from volatility_generalisation.modelling_code.integrated_model_recovery import (
    DEFAULT_PARETO_K_THRESHOLD,
    DEFAULT_RESULTS_DIR,
    compute_recovery_seed,
    recovery_output_dir,
    simulate_integrated_dataset,
)
from volatility_generalisation.modelling_code.integrated_model_selection import (
    fit_integrated_mcmc,
    get_default_conditions,
    get_default_models,
    integrated_samples_to_idata,
)

numpyro.set_platform("cpu")
numpyro.set_host_device_count(HOST_DEVICE_COUNT)
jax.config.update("jax_enable_x64", True)

job_tmp_dir = tempfile.mkdtemp(prefix="arviz_tmp_")
os.environ["ARVIZ_DATA_DIR"] = job_tmp_dir


def cleanup_arviz_tmp() -> None:
    """Delete this job's temporary ArviZ folder when the script exits."""
    shutil.rmtree(job_tmp_dir, ignore_errors=True)


atexit.register(cleanup_arviz_tmp)

console = Console()

DEFAULT_CONFIG = "scripts/config.toml"
DEFAULT_MODEL_CONFIG = "config/model_bounds.toml"
DEFAULT_DATA_DIR = "data/prepared"


def _mask_initial_trials(idata: az.InferenceData, n_initial: int) -> az.InferenceData:
    """Remove the first trials from the log-likelihood before computing LOO.

    By default the first trial is dropped (`--loo-drop-initial-trials`). The
    input is not changed; a copy is returned.

    Args:
        idata: Fitted model results that include the pointwise log-likelihood,
            with a "trial" dimension.
        n_initial: Number of trials to drop from the start of each participant's
            data. If 0, the input is returned unchanged.

    Returns:
        Model results whose log-likelihood starts after the first `n_initial`
        trials.

    Raises:
        ValueError: If `n_initial` is negative, or is not less than the total
            number of trials.
    """
    if n_initial < 0:
        raise ValueError(f"n_initial ({n_initial}) must be non-negative")
    if n_initial <= 0:
        return idata
    n_total = idata.log_likelihood["obs"].sizes["trial"]
    if n_initial >= n_total:
        raise ValueError(
            f"n_initial ({n_initial}) must be less than total trials ({n_total})"
        )
    log_lik_masked = {
        name: da.isel(trial=slice(n_initial, None))
        for name, da in idata.log_likelihood.data_vars.items()
    }
    new = idata.copy()
    new.log_likelihood = xr.Dataset(log_lik_masked)
    return new


def parse_args() -> argparse.Namespace:
    """Read the command-line options for one model-recovery fit.

    Returns:
        The command-line options, one attribute per option: the replicate
        number, the simulated and fitted model names, the data and output
        folders, the number of simulated participants, the seed, the MCMC
        settings and the LOO settings.
    """
    parser = argparse.ArgumentParser(description="Run one integrated model-recovery fit")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--simulated-model", required=True)
    parser.add_argument("--fitted-model", required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n-subjects", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--target-accept-prob", type=float, default=0.95)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--loo-drop-initial-trials", type=int, default=1)
    parser.add_argument("--pareto-k-threshold", type=float, default=DEFAULT_PARETO_K_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    """Simulate one dataset, fit one candidate model to it, and save the results.

    Simulates participants from the simulated model, using a seed worked out from
    the replicate and the simulated model. It then fits the fitted model with
    MCMC, checks the chains, and computes LOO scores after dropping the first
    `--loo-drop-initial-trials` trials (1 by default; 0 keeps all). It saves the
    posterior samples, the simulation details, a fit summary and the diagnostics
    to the output folder for this replicate and model pair. The results are saved
    even if the chains did not converge; the summary records whether they did.
    The script still ends with exit code 0, so SLURM reports the task as
    completed.
    """
    args = parse_args()
    settings_record = settings_file_record(args.model_config)
    model_order = args.models or get_default_models(args.config)
    conditions = args.conditions or get_default_conditions(args.config)
    simulation_seed = compute_recovery_seed(
        replicate=args.replicate,
        simulated_model=args.simulated_model,
        model_order=model_order,
        base_seed=args.base_seed,
    )

    output_dir = recovery_output_dir(
        Path(args.output_dir),
        replicate=args.replicate,
        simulated_model=args.simulated_model,
        fitted_model=args.fitted_model,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Replicate:[/cyan] {args.replicate}")
    console.print(f"[cyan]Simulated:[/cyan] {args.simulated_model}")
    console.print(f"[cyan]Fitted:[/cyan] {args.fitted_model}")
    console.print(f"[cyan]Seed:[/cyan] {simulation_seed}")

    data = simulate_integrated_dataset(
        simulated_model=args.simulated_model,
        conditions=conditions,
        n_subjects=args.n_subjects,
        seed=simulation_seed,
        data_dir=args.data_dir,
        config_path=args.model_config,
    )

    fit_start = time.time()
    mcmc = fit_integrated_mcmc(
        data=data,
        model_name=args.fitted_model,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=simulation_seed,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        thinning=args.thinning,
        progress_bar=True,
        config_path=args.model_config,
    )
    fit_duration = time.time() - fit_start

    posterior_samples = mcmc.get_samples(group_by_chain=True)
    diagnostics = diagnose_mcmc(
        mcmc,
        posterior_samples,
        param_names=[
            key
            for key in posterior_samples
            if key.startswith(f"{args.fitted_model}_") and not key.endswith("choice_prob")
        ],
        verbose=True,
    )
    if not diagnostics['converged']:
        print_chain_failure_summary(posterior_samples, diagnostics)
    idata = integrated_samples_to_idata(mcmc, model_name=args.fitted_model, data=data)
    idata_for_loo = _mask_initial_trials(idata, args.loo_drop_initial_trials)
    loo = az.loo(idata_for_loo, pointwise=True)
    pareto_k = np.asarray(loo.pareto_k.values)
    n_trials_used = int(idata_for_loo.log_likelihood["obs"].sizes["trial"])

    az.to_netcdf(idata, output_dir / "posterior.nc")

    dataset_metadata = {
        "replicate": args.replicate,
        "simulated_model": args.simulated_model,
        "fitted_model": args.fitted_model,
        "conditions": conditions,
        "n_subjects": args.n_subjects,
        "n_conditions": len(data["conditions"]),
        "n_trials": int(np.asarray(data["choices"]).shape[-1]),
        "subject_ids": data["subject_ids"],
        "subject_labels": data["subject_labels"],
        "condition_labels": data["condition_labels"],
        "context": np.asarray(data["context"]).tolist(),
        "simulation_seed": simulation_seed,
        "choice_coding": "1 means option 0 chosen; 0 means option 1 chosen",
        "output_path": str(output_dir),
        **settings_record,
    }
    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_metadata, handle, indent=2)

    simulation_metadata = {
        "simulated_model": data["simulated_model"],
        "simulation_seed": data["simulation_seed"],
        "conditions": data["conditions"],
        "hierarchy_parameters": data["hierarchy_parameters"],
        "true_parameters": {
            key: np.asarray(value).tolist() for key, value in data["true_parameters"].items()
        },
    }
    with open(output_dir / "simulation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(simulation_metadata, handle, indent=2)

    summary = {
        "replicate": args.replicate,
        "simulated_model": args.simulated_model,
        "fitted_model": args.fitted_model,
        "simulation_seed": simulation_seed,
        "n_subjects": args.n_subjects,
        "conditions": conditions,
        "elpd_loo": float(loo.elpd_loo),
        "se_loo": float(loo.se),
        "p_loo": float(loo.p_loo),
        "loo_rank_within_dataset": None,
        "n_bad_pareto_k": int((pareto_k > args.pareto_k_threshold).sum()),
        "max_pareto_k": float(pareto_k.max()),
        "n_initial_trials_dropped": int(args.loo_drop_initial_trials),
        "n_trials_used": n_trials_used,
        "converged": bool(diagnostics["converged"]),
        "fit_duration": float(fit_duration),
        "pareto_k_threshold": float(args.pareto_k_threshold),
        "output_path": str(output_dir),
        "posterior_path": str(output_dir / "posterior.nc"),
    }
    with open(output_dir / "fit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                **diagnostics,
                "n_initial_trials_dropped": int(args.loo_drop_initial_trials),
                "n_trials_used": n_trials_used,
                "loo": {
                    "elpd_loo": float(loo.elpd_loo),
                    "p_loo": float(loo.p_loo),
                    "se": float(loo.se),
                },
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
