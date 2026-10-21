#!/usr/bin/env python3
"""Fit one integrated hierarchical model across multiple conditions.

This is the fitting step of the model-selection analysis. It fits one candidate
model to the prepared data from all chosen conditions together, checks the
chains, computes leave-one-out cross-validation (LOO) scores, and saves the
results. Each SLURM array task runs it once for one candidate model. The results
are later compared across models by compile_integrated_model_selection.py. The
results are saved even if the chains did not converge, and the script still ends
with exit code 0, so SLURM reports the task as completed.
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
from typing import Any


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
import matplotlib
import numpy as np
import numpyro
import xarray as xr
import pandas as pd
from rich.console import Console

from volatility_generalisation.modelling_code.diagnostics import (
    diagnose_mcmc,
    print_chain_failure_summary,
)
from volatility_generalisation.modelling_code.data_generation import (
    settings_file_record,
)
from volatility_generalisation.modelling_code.data_preparation import (
    OUTCOME_CODING_KEY,
    OUTCOME_CODING_REWARD_SAFE,
)
from volatility_generalisation.modelling_code.integrated_model_selection import (
    fit_integrated_mcmc,
    fit_integrated_pathfinder_restarts,
    fit_integrated_svi,
    get_default_conditions,
    integrated_samples_to_idata,
    load_integrated_dataset,
    select_pathfinder_init_params,
    svi_to_init_params,
)
from volatility_generalisation.modelling_code.posterior_utils import posterior_interval

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
DEFAULT_OUTPUT_DIR = "results/integrated_model_selection"


def parse_args() -> argparse.Namespace:
    """Read the command-line options for one model-selection fit.

    Covers the candidate model, the conditions, the data and output folders,
    the MCMC settings, the optional Pathfinder or SVI starting-point search,
    and the number of initial trials to drop from the LOO calculation.

    Returns:
        The command-line options, one attribute per option.
    """
    parser = argparse.ArgumentParser(
        description="Run integrated hierarchical model selection for one candidate model."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Pipeline config TOML")
    parser.add_argument(
        "--model-config",
        default=DEFAULT_MODEL_CONFIG,
        help="Model bounds config TOML",
    )
    parser.add_argument("--model", required=True, help="Candidate model family to fit")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Prepared conditions to include in one integrated fit",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Prepared data directory")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for integrated fits",
    )
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accept-prob", type=float, default=0.95)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--thinning", type=int, default=1)

    # Pathfinder initialization (opt-in)
    parser.add_argument("--pathfinder-init", action="store_true",
                        help="Use Pathfinder to find a good starting point before MCMC")
    parser.add_argument("--pathfinder-restarts", type=int, default=1,
                        help="Number of Pathfinder restarts (picks best by ELBO)")
    parser.add_argument("--pathfinder-num-samples", type=int, default=200,
                        help="Internal Pathfinder L-BFGS samples")
    parser.add_argument("--pathfinder-posterior-samples", type=int, default=32,
                        help="Samples drawn from Pathfinder approximation for init selection")
    parser.add_argument("--pathfinder-ftol", type=float, default=1e-4,
                        help="Pathfinder convergence tolerance")
    parser.add_argument("--pathfinder-seed", type=int, default=None,
                        help="Random seed for Pathfinder (defaults to main seed)")

    # SVI initialization (opt-in)
    parser.add_argument("--svi-init", action="store_true",
                        help="Use SVI to find a good starting point before MCMC")
    parser.add_argument("--svi-steps", type=int, default=10000,
                        help="Max SVI optimisation steps")
    parser.add_argument("--svi-learning-rate", type=float, default=1e-3,
                        help="Peak SVI learning rate")
    parser.add_argument("--svi-guide-type", default="normal",
                        choices=["normal", "lowrank"],
                        help="SVI guide family")
    parser.add_argument("--svi-eval-freq", type=int, default=100,
                        help="Check convergence every N SVI steps")
    parser.add_argument("--svi-patience", type=int, default=20,
                        help="Early-stop after this many checks without improvement")
    parser.add_argument("--svi-min-delta", type=float, default=0.001,
                        help="Minimum loss change to count as improvement")
    parser.add_argument("--svi-jitter-scale", type=float, default=0.01,
                        help="Std of Gaussian jitter added to SVI samples for NUTS init")
    parser.add_argument("--svi-seed", type=int, default=None,
                        help="Random seed for SVI (defaults to main seed)")
    parser.add_argument(
        "--loo-drop-initial-trials", type=int, default=1,
        help="Drop N initial trials from LOO computation (default: 1). "
             "Set to 0 to include all trials.",
    )
    return parser.parse_args()


def _output_dir(base_output_dir: Path, model_name: str) -> Path:
    """Give the folder in which the results for one model are saved.

    Args:
        base_output_dir: Folder that holds the results for all models.
        model_name: Name of the fitted model.

    Returns:
        Path to the sub-folder named after the model.
    """
    return base_output_dir / model_name


def _posterior_mean_by_subject_session(idata: az.InferenceData, site_name: str) -> np.ndarray:
    """Average one parameter's posterior samples over all chains and draws.

    Args:
        idata: Fitted model results with a posterior group.
        site_name: Name of the parameter in the posterior, for example a
            subject-level parameter that has one value per participant.

    Returns:
        Array of the posterior mean, with the chain and draw dimensions removed
        (for a subject-level parameter, one value per participant).
    """
    values = idata.posterior[site_name].values
    return values.mean(axis=(0, 1))


def _write_group_summaries(idata: az.InferenceData, model_name: str, output_dir: Path) -> None:
    """Save a table that summarises the group-level parameters of a model.

    The table has the posterior mean, standard deviation, 95% equal-tailed
    interval (``ci_lower`` and ``ci_upper``) and convergence measures
    (``mcse_mean``, ``mcse_sd``, ``ess_bulk``, ``ess_tail`` and ``r_hat``) for
    each group-level parameter, and is saved as group_parameter_summary.csv.
    Nothing is saved if the model has no group-level parameters.

    Args:
        idata: Fitted model results with a posterior group.
        model_name: Name of the fitted model. Only parameters whose names start
            with it are summarised.
        output_dir: Folder in which to save the CSV file.
    """
    group_vars = [
        variable
        for variable in idata.posterior.data_vars
        if variable.startswith(f"{model_name}_")
        and not variable.endswith("_subj")
        and variable != "choice_prob"
    ]
    if not group_vars:
        return
    # ArviZ's default summary reports a 94% highest-density interval. Ask only
    # for its convergence measures, and add the mean, standard deviation and the
    # project's 95% equal-tailed interval. Each function receives the draws of
    # one parameter from all chains pooled together. Values are kept unrounded
    # (by default ArviZ rounds most columns to 3 decimals), so that interval
    # ends can be quoted.
    summary = az.summary(
        idata,
        var_names=group_vars,
        kind="diagnostics",
        round_to="none",
        stat_funcs={
            "mean": np.mean,
            "sd": lambda draws: np.std(draws, ddof=1),
            "ci_lower": lambda draws: posterior_interval(draws)[0],
            "ci_upper": lambda draws: posterior_interval(draws)[1],
        },
    )
    summary.to_csv(output_dir / "group_parameter_summary.csv")


def _write_subject_summaries(
    idata: az.InferenceData,
    model_name: str,
    subject_ids: list[str],
    subject_labels: list[str],
    condition_labels: list[str],
    output_dir: Path,
) -> None:
    """Save each participant's posterior mean for every subject-level parameter.

    The table has one row per parameter and participant, with the participant's
    condition, and is saved as subject_parameter_summary.csv. Nothing is saved
    if the model has no subject-level parameters.

    Args:
        idata: Fitted model results with a posterior group.
        model_name: Name of the fitted model. Only parameters whose names start
            with it and end in "_subj" are used.
        subject_ids: Participant identifiers, in the order used in the fit.
        subject_labels: Participant labels, one per entry in `subject_ids`.
        condition_labels: Condition of each entry in `subject_ids`.
        output_dir: Folder in which to save the CSV file.
    """
    rows = []
    for variable in idata.posterior.data_vars:
        if not variable.startswith(f"{model_name}_") or not variable.endswith("_subj"):
            continue
        parameter_name = variable.replace(f"{model_name}_", "").replace("_subj", "")
        values = _posterior_mean_by_subject_session(idata, variable)
        for subject_index, subject_id in enumerate(subject_ids):
            rows.append(
                {
                    "model": model_name,
                    "parameter": parameter_name,
                    "subject_id": subject_id,
                    "subject_label": subject_labels[subject_index],
                    "condition": condition_labels[subject_index],
                    "posterior_mean": float(values[subject_index]),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "subject_parameter_summary.csv", index=False)


def _write_posterior_summary_plot(data: dict, output_dir: Path) -> None:
    """Save a bar chart of the observed choice rate in each condition.

    Each participant's choice rate is the mean of their choices. The bars show
    the average of these rates across the participants in each condition. The
    chart is saved as observed_choice_rates.png.

    Args:
        data: Integrated dataset with the array `choices` (participants by
            trials) and the list `condition_labels` (one condition per
            participant).
        output_dir: Folder in which to save the figure.
    """
    subject_choice_rate = np.asarray(data["choices"]).mean(axis=1)
    plot_df = pd.DataFrame(
        {
            "condition": data["condition_labels"],
            "choice_rate": subject_choice_rate,
        }
    )
    observed = plot_df.groupby("condition", sort=False)["choice_rate"].mean()
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(observed)), observed.values, color="#4C78A8")
    plt.xticks(np.arange(len(observed)), observed.index.tolist(), rotation=45, ha="right")
    plt.ylabel("Observed choice rate")
    plt.title("Observed Condition-Level Choice Rates")
    plt.tight_layout()
    plt.savefig(output_dir / "observed_choice_rates.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Fit one candidate model to all chosen conditions and save the results.

    Loads the prepared data for the conditions and, if asked, finds a starting
    point for the chains with Pathfinder or SVI. It then fits the model with
    MCMC, checks the chains, and computes LOO scores after dropping the first
    `--loo-drop-initial-trials` trials (1 by default; 0 keeps all). It saves the
    posterior samples, parameter summaries, a choice-rate figure, the metadata
    and the diagnostics to a folder named after the model. The results are saved
    even if the chains did not converge; the diagnostics file records whether
    they did. The script still ends with exit code 0, so SLURM reports the task
    as completed.

    Raises:
        SystemExit: If both `--pathfinder-init` and `--svi-init` are given.
        ValueError: If `--pathfinder-posterior-samples` is smaller than
            `--num-chains`.
    """
    args = parse_args()
    settings_record = settings_file_record(args.model_config)
    conditions = args.conditions or get_default_conditions(args.config)
    data = load_integrated_dataset(conditions=conditions, data_dir=args.data_dir)
    output_dir = _output_dir(Path(args.output_dir), args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Model:[/cyan] {args.model}")
    console.print(f"[cyan]Conditions:[/cyan] {', '.join(conditions)}")
    console.print(f"[cyan]Subjects:[/cyan] {len(data['subject_ids'])}")
    console.print(f"[cyan]Trials:[/cyan] {data['choices'].shape[-1]}")

    # Initialization method selection
    if args.pathfinder_init and args.svi_init:
        console.print("[red]Error: --pathfinder-init and --svi-init are mutually exclusive[/red]")
        raise SystemExit(1)

    init_params = None
    init_info: dict[str, Any] = {"method": "default"}

    if args.pathfinder_init:
        pf_seed = args.pathfinder_seed if args.pathfinder_seed is not None else args.seed
        console.print("[bold cyan]Running Pathfinder to locate a high-density mode...[/bold cyan]")

        if args.pathfinder_posterior_samples < args.num_chains:
            raise ValueError(
                f"--pathfinder-posterior-samples ({args.pathfinder_posterior_samples}) "
                f"must be >= --num-chains ({args.num_chains})"
            )

        t_pf = time.perf_counter()
        unconstrained_samples, _constrained_samples, pf_elbo = fit_integrated_pathfinder_restarts(
            data=data,
            model_name=args.model,
            num_restarts=args.pathfinder_restarts,
            seed=pf_seed,
            config_path=args.model_config,
            num_pathfinder_samples=args.pathfinder_num_samples,
            num_posterior_samples=args.pathfinder_posterior_samples,
            ftol=args.pathfinder_ftol,
        )
        pf_duration = time.perf_counter() - t_pf

        init_params = select_pathfinder_init_params(
            unconstrained_samples,
            num_chains=args.num_chains,
            seed=pf_seed + 999,
        )

        init_info = {
            "method": "pathfinder",
            "pathfinder_seed": pf_seed,
            "pathfinder_restarts": args.pathfinder_restarts,
            "pathfinder_num_samples": args.pathfinder_num_samples,
            "pathfinder_posterior_samples": args.pathfinder_posterior_samples,
            "pathfinder_ftol": args.pathfinder_ftol,
            "pathfinder_duration": round(pf_duration, 1),
        }
        if pf_elbo is not None:
            init_info["pathfinder_elbo"] = pf_elbo

        elbo_str = f" = {pf_elbo:.2f}" if pf_elbo is not None else ""
        console.print(f"  [green]✓ Pathfinder ELBO{elbo_str}[/green]")
        console.print(f"  [dim]Pathfinder completed in {pf_duration:.1f}s[/dim]")
        console.print(f"  [dim]Selected Pathfinder init params for {args.num_chains} chains[/dim]")

    if args.svi_init:
        svi_seed = args.svi_seed if args.svi_seed is not None else args.seed
        console.print("[bold cyan]Running SVI to locate a high-density mode...[/bold cyan]")

        t_svi = time.perf_counter()
        svi_guide, svi_params, svi_losses = fit_integrated_svi(
            data=data,
            model_name=args.model,
            seed=svi_seed,
            config_path=args.model_config,
            num_steps=args.svi_steps,
            learning_rate=args.svi_learning_rate,
            guide_type=args.svi_guide_type,
            eval_freq=args.svi_eval_freq,
            patience=args.svi_patience,
            min_delta=args.svi_min_delta,
        )
        svi_duration = time.perf_counter() - t_svi

        init_params = svi_to_init_params(
            data=data,
            model_name=args.model,
            guide=svi_guide,
            params=svi_params,
            num_chains=args.num_chains,
            seed=svi_seed,
            config_path=args.model_config,
            jitter_scale=args.svi_jitter_scale,
        )

        init_info = {
            "method": "svi",
            "svi_seed": svi_seed,
            "svi_steps": args.svi_steps,
            "svi_learning_rate": args.svi_learning_rate,
            "svi_guide_type": args.svi_guide_type,
            "svi_final_loss": round(float(svi_losses[-1]), 2),
            "svi_best_loss": round(float(min(svi_losses)), 2),
            "svi_duration": round(svi_duration, 1),
            "svi_converged": len(svi_losses) < args.svi_steps,
        }

        console.print(f"  [green]✓ SVI best loss = {init_info['svi_best_loss']}[/green]")
        console.print(f"  [dim]SVI completed in {svi_duration:.1f}s over {len(svi_losses)} steps[/dim]")
        console.print(f"  [dim]Selected SVI init params for {args.num_chains} chains[/dim]")

    mcmc_fit_kwargs: dict[str, Any] = {
        "data": data,
        "model_name": args.model,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "seed": args.seed,
        "target_accept_prob": args.target_accept_prob,
        "max_tree_depth": args.max_tree_depth,
        "thinning": args.thinning,
        "config_path": args.model_config,
        "progress_bar": True,
    }
    if init_params is not None:
        mcmc_fit_kwargs["init_params"] = init_params
    mcmc = fit_integrated_mcmc(**mcmc_fit_kwargs)
    posterior_samples = mcmc.get_samples(group_by_chain=True)
    diagnostics = diagnose_mcmc(
        mcmc,
        posterior_samples,
        param_names=[
            key
            for key in posterior_samples
            if key.startswith(f"{args.model}_") and not key.endswith("choice_prob")
        ],
        verbose=True,
    )
    if not diagnostics['converged']:
        print_chain_failure_summary(posterior_samples, diagnostics)
    console.print("[bold cyan]Computing trial-level log likelihood post-hoc...[/bold cyan]")
    idata = integrated_samples_to_idata(mcmc, model_name=args.model, data=data, console=console)
    ll_shape = idata.log_likelihood["obs"].shape
    console.print(f"[green]✓ Log-likelihood shape:[/green] {ll_shape}")
    assert len(ll_shape) == 4, (
        f"Expected 4D log-likelihood (chain, draw, subject, trial), "
        f"got shape {ll_shape} with {len(ll_shape)} dimensions"
    )

    n_drop = args.loo_drop_initial_trials
    if n_drop > 0:
        n_total = idata.log_likelihood["obs"].sizes["trial"]
        idata_for_loo = idata.copy()
        idata_for_loo.log_likelihood = xr.Dataset({
            name: da.isel(trial=slice(n_drop, None))
            for name, da in idata.log_likelihood.data_vars.items()
        })
        console.print(
            f"  [dim]Dropped {n_drop} initial trial(s) from LOO "
            f"({n_total - n_drop} of {n_total} trials used)[/dim]"
        )
    else:
        idata_for_loo = idata

    t_loo = time.perf_counter()
    loo = az.loo(idata_for_loo, pointwise=True)
    console.print(f"[dim]PSIS-LOO done in {time.perf_counter() - t_loo:.1f}s[/dim]")

    az.to_netcdf(idata, output_dir / "posterior.nc")
    _write_group_summaries(idata, args.model, output_dir)
    _write_subject_summaries(
        idata,
        args.model,
        data["subject_ids"],
        data["subject_labels"],
        data["condition_labels"],
        output_dir,
    )
    _write_posterior_summary_plot(data, output_dir)

    metadata = {
        "model": args.model,
        "conditions": conditions,
        "n_subjects": len(data["subject_ids"]),
        "n_conditions": len(data["conditions"]),
        "n_trials": int(data["choices"].shape[-1]),
        "n_initial_trials_dropped": n_drop,
        "subject_ids": data["subject_ids"],
        "subject_labels": data["subject_labels"],
        "condition_labels": data["condition_labels"],
        "context": np.asarray(data["context"]).tolist(),
        "initialization": init_info,
        OUTCOME_CODING_KEY: OUTCOME_CODING_REWARD_SAFE,
        **settings_record,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                **diagnostics,
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
