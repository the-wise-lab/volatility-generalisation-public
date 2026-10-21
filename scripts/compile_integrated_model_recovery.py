#!/usr/bin/env python3
"""Compile integrated model-recovery outputs into summary tables.

This is the last step of the integrated model-recovery workflow. Each
replicate simulates a dataset from one model and fits every candidate model
to it. This script reads all the fit summaries, recomputes or reuses the
leave-one-out (LOO) cross-validation scores, ranks the fitted models within
each simulated dataset, and picks the winner. It then writes the ranking and
winner tables, parameter-recovery tables, a table of how well the group-level
effects are recovered, and the confusion and inversion matrices (how often each
simulated model is recovered). The recovery figures are drawn from these tables
by ``make_recovery_figures.py``. It fails if any replicate, simulated-model and
fitted-model fit is missing, or if a fit has no usable ``elpd_loo``. Fits that
did not converge, or that have a high Pareto k, are ranked like any other fit
and flagged in the tables.
See ``docs/3-model-selection-and-recovery.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import arviz as az
import pandas as pd
import numpy as np
import xarray as xr
from rich.console import Console

from volatility_generalisation.modelling_code.integrated_model_recovery import (
    DEFAULT_PARETO_K_THRESHOLD,
    DEFAULT_RESULTS_DIR,
    build_integrated_group_effect_recovery,
    build_integrated_parameter_recovery,
    build_recovery_matrices,
    load_recovery_fit_summary,
    rank_recovery_fits,
    summarize_recovery_dataset_counts,
    validate_recovery_fit_grid,
    select_recovery_winners,
)
from volatility_generalisation.modelling_code.integrated_model_selection import get_default_models


console = Console()


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options: config and results paths, output folder, models,
        required number of replicates, Pareto-k threshold, number of initial
        trials to drop from LOO, overwrite flag, LOO source and number of
        LOO workers.
    """
    parser = argparse.ArgumentParser(description="Compile integrated model-recovery outputs")
    parser.add_argument("--config", default="scripts/config.toml")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Derived output directory (default: results/integrated_model_recovery/summary)",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--replicates",
        type=int,
        required=True,
        help="Number of replicates that should be present (replicates 0 to N-1)",
    )
    parser.add_argument("--pareto-k-threshold", type=float, default=DEFAULT_PARETO_K_THRESHOLD)
    parser.add_argument("--loo-drop-initial-trials", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty derived output directory",
    )
    parser.add_argument(
        "--loo-source",
        choices=["auto", "posterior", "summary"],
        default="auto",
        help=(
            "Use cached fit_summary.json LOO when possible ('auto'), force posterior.nc "
            "recomputation ('posterior'), or require fresh summary metadata ('summary')."
        ),
    )
    parser.add_argument("--loo-workers", type=int, default=1)
    return parser.parse_args()


def _resolve_output_dir(results_dir: Path, output_dir: str | None) -> Path:
    """Choose the folder where the compiled outputs are written.

    Args:
        results_dir: Folder with the recovery fits.
        output_dir: Folder given on the command line, or None to use the
            ``summary`` folder inside the results folder.

    Returns:
        Path of the output folder.
    """
    if output_dir is not None:
        return Path(output_dir)
    return results_dir / "summary"


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create the output folder, refusing to reuse a non-empty one by accident.

    Args:
        output_dir: Folder for the compiled outputs.
        overwrite: If True, allow writing into a folder that already has files.

    Raises:
        RuntimeError: If the path exists but is not a folder, or if the folder
            has files and ``overwrite`` is False.
    """
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise RuntimeError(
                f"Output directory is non-empty: {output_dir}. Re-run with --overwrite to continue."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_rows(results_dir: Path) -> list[dict]:
    """Read every fit summary in the replicate folders.

    Args:
        results_dir: Folder with one ``replicate_*`` subfolder per replicate.

    Returns:
        One dictionary per ``fit_summary.json`` file, with the path of the
        file added as ``summary_path``.
    """
    rows = []
    for replicate_dir in sorted(results_dir.glob("replicate_*")):
        if not replicate_dir.is_dir():
            continue
        for summary_path in sorted(replicate_dir.rglob("fit_summary.json")):
            row = load_recovery_fit_summary(summary_path)
            row["summary_path"] = str(summary_path)
            rows.append(row)
    return rows


def _mask_initial_trials(idata: az.InferenceData, n_initial: int) -> az.InferenceData:
    """Remove the first trials from the log-likelihood before computing LOO.

    Args:
        idata: Fit results holding the pointwise log-likelihood ``obs``, with
            a ``trial`` dimension.
        n_initial: Number of trials to drop from the start of each subject's
            data. Zero keeps every trial.

    Returns:
        The fit results with the initial trials removed from the
        log-likelihood. The input is returned unchanged if ``n_initial`` is
        zero.

    Raises:
        ValueError: If ``n_initial`` is negative or is not less than the
            number of trials.
    """
    if n_initial < 0:
        raise ValueError("n_initial must be non-negative")
    if n_initial == 0:
        return idata
    n_total = idata.log_likelihood["obs"].sizes["trial"]
    if n_initial >= n_total:
        raise ValueError(f"n_initial ({n_initial}) must be less than total trials ({n_total})")
    log_lik_masked = {
        name: da.isel(trial=slice(n_initial, None))
        for name, da in idata.log_likelihood.data_vars.items()
    }
    new = idata.copy()
    new.log_likelihood = xr.Dataset(log_lik_masked)
    return new


def _load_log_likelihood_idata(posterior_path: Path) -> az.InferenceData:
    """Load only the log-likelihood group from a saved posterior file.

    Args:
        posterior_path: Path to the ``posterior.nc`` file of a fit.

    Returns:
        Fit results holding only the log-likelihood group.
    """
    # Load only the log_likelihood group so worker processes avoid materializing the full posterior.
    with xr.open_dataset(posterior_path, group="log_likelihood", engine="h5netcdf") as dataset:
        log_likelihood = dataset.load()
    return az.InferenceData(log_likelihood=log_likelihood)


def _recompute_loo_row(task: tuple[int, dict, int, float]) -> tuple[int, dict, int]:
    """Recompute LOO for one fit from its saved posterior file.

    Drops the initial trials, then computes pointwise LOO and counts the
    trials whose Pareto k value is above the threshold. Takes a single tuple
    so that it can run in a separate process.

    Args:
        task: Tuple of the row index, the fit summary row, the number of
            initial trials to drop, and the Pareto-k threshold.

    Returns:
        Tuple of the row index, the summary row updated with the new LOO
        values (``elpd_loo``, ``se_loo``, ``p_loo``, ``n_bad_pareto_k``,
        ``max_pareto_k``, ``n_trials_used``, ``n_initial_trials_dropped``),
        and the number of trials used.

    Raises:
        FileNotFoundError: If the posterior file is missing.
        ValueError: If the posterior file has no ``log_likelihood['obs']``.
    """
    row_index, row, n_initial_trials_dropped, pareto_k_threshold = task
    posterior_path = Path(str(row["posterior_path"]))
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior.nc at {posterior_path}")

    idata = _load_log_likelihood_idata(posterior_path)
    if not hasattr(idata, "log_likelihood") or "obs" not in idata.log_likelihood:
        raise ValueError(f"posterior.nc at {posterior_path} must contain log_likelihood['obs']")

    idata_for_loo = _mask_initial_trials(idata, n_initial_trials_dropped)
    n_trials_used = int(idata_for_loo.log_likelihood["obs"].sizes["trial"])
    # Provide a fixed relative efficiency so LOO can run from the log_likelihood group alone.
    loo_result = az.loo(idata_for_loo, pointwise=True, reff=1.0)
    pareto_k = loo_result.pareto_k.values

    updated_row = dict(row)
    updated_row.update(
        {
            "elpd_loo": float(loo_result.elpd_loo),
            "se_loo": float(loo_result.se),
            "p_loo": float(loo_result.p_loo),
            "n_bad_pareto_k": int(np.asarray(pareto_k > pareto_k_threshold).sum()),
            "max_pareto_k": float(np.max(pareto_k)),
            "n_trials_used": n_trials_used,
            "n_initial_trials_dropped": n_initial_trials_dropped,
        }
    )
    return row_index, updated_row, n_trials_used


def _summary_loo_staleness_reason(
    row: dict,
    *,
    n_initial_trials_dropped: int,
    pareto_k_threshold: float,
) -> str | None:
    """Check whether the LOO values saved in a fit summary can be reused.

    The saved values are out of date if they were computed with a different
    number of dropped trials or a different Pareto-k threshold, or if a value
    is missing or invalid.

    Args:
        row: Fit summary row read from ``fit_summary.json``.
        n_initial_trials_dropped: Number of initial trials that the current
            run drops.
        pareto_k_threshold: Pareto-k threshold of the current run.

    Returns:
        A short reason why the saved values cannot be used, or None if they
        can be used.
    """
    if row.get("n_initial_trials_dropped") != n_initial_trials_dropped:
        return "missing or mismatched n_initial_trials_dropped"
    if row.get("n_trials_used") is None:
        return "missing n_trials_used"
    try:
        if int(row["n_trials_used"]) < 1:
            return "invalid n_trials_used"
    except (TypeError, ValueError):
        return "invalid n_trials_used"

    elpd_loo = row.get("elpd_loo")
    max_pareto_k = row.get("max_pareto_k")
    try:
        elpd_loo_f = float(elpd_loo)
    except (TypeError, ValueError):
        return "missing or non-finite elpd_loo"
    try:
        max_pareto_k_f = float(max_pareto_k)
    except (TypeError, ValueError):
        return "missing or non-finite max_pareto_k"
    if not np.isfinite(elpd_loo_f):
        return "missing or non-finite elpd_loo"
    if not np.isfinite(max_pareto_k_f):
        return "missing or non-finite max_pareto_k"

    summary_threshold = row.get("pareto_k_threshold")
    if summary_threshold is None or not np.isclose(float(summary_threshold), pareto_k_threshold):
        return "missing or mismatched pareto_k_threshold"

    n_bad_pareto_k = row.get("n_bad_pareto_k")
    try:
        if int(n_bad_pareto_k) < 0:
            return "invalid n_bad_pareto_k"
    except (TypeError, ValueError):
        return "invalid n_bad_pareto_k"

    return None


def _resolve_loo_rows(
    rows: list[dict],
    *,
    loo_source: str,
    loo_workers: int,
    n_initial_trials_dropped: int,
    pareto_k_threshold: float,
) -> tuple[list[dict], set[int]]:
    """Get up-to-date LOO values for every fit summary row.

    Depending on ``loo_source``, each row either keeps the LOO values saved in
    its fit summary or has them recomputed from the saved posterior file.
    Recomputation can run in several processes.

    Args:
        rows: Fit summary rows, one per replicate, simulated model and fitted
            model.
        loo_source: ``"summary"`` requires up-to-date saved values and fails
            otherwise. ``"auto"`` uses saved values where possible and
            recomputes the rest. ``"posterior"`` recomputes every row.
        loo_workers: Number of processes for recomputing LOO.
        n_initial_trials_dropped: Number of initial trials to drop before
            computing LOO.
        pareto_k_threshold: Pareto-k value above which a trial counts as
            poorly estimated.

    Returns:
        Tuple of the rows with current LOO values, in the original order, and
        the set of numbers of trials used across rows.

    Raises:
        ValueError: If ``loo_source`` is ``"summary"`` and a row's saved
            values are out of date or incomplete.
        RuntimeError: If recomputing LOO fails for a row in a worker process.
    """
    resolved_entries: list[tuple[int, dict, int]] = []
    queued_entries: list[tuple[int, dict]] = []

    for row_index, row in enumerate(rows):
        staleness_reason = _summary_loo_staleness_reason(
            row,
            n_initial_trials_dropped=n_initial_trials_dropped,
            pareto_k_threshold=pareto_k_threshold,
        )

        if loo_source == "summary":
            if staleness_reason is not None:
                raise ValueError(
                    "fit_summary.json LOO is stale or incomplete for "
                    f"{row['replicate']} / {row['simulated_model']} → {row['fitted_model']}: {staleness_reason}"
                )
            n_trials_used = int(row["n_trials_used"])
            console.print(
                f"[bold cyan]•[/bold cyan] {row['replicate']} / {row['simulated_model']} → {row['fitted_model']}"
            )
            console.print(
                f"  [dim]Using cached summary LOO[/dim] "
                f"({n_trials_used} trials after dropping {n_initial_trials_dropped})"
            )
            resolved_entries.append((row_index, dict(row), n_trials_used))
            continue

        if loo_source == "auto" and staleness_reason is None:
            n_trials_used = int(row["n_trials_used"])
            console.print(
                f"[bold cyan]•[/bold cyan] {row['replicate']} / {row['simulated_model']} → {row['fitted_model']}"
            )
            console.print(
                f"  [dim]Using cached summary LOO[/dim] "
                f"({n_trials_used} trials after dropping {n_initial_trials_dropped})"
            )
            resolved_entries.append((row_index, dict(row), n_trials_used))
            continue

        console.print(
            f"[bold cyan]•[/bold cyan] {row['replicate']} / {row['simulated_model']} → {row['fitted_model']}"
        )
        if loo_source == "auto":
            console.print(f"  [dim]Cached summary LOO unusable:[/dim] {staleness_reason}; recomputing")
        console.print(f"  [dim]Loading posterior:[/dim] {row['posterior_path']}")
        queued_entries.append((row_index, row))

    if queued_entries:
        if loo_workers > 1 and len(queued_entries) > 1:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            with ProcessPoolExecutor(max_workers=loo_workers) as executor:
                future_to_row = {
                    executor.submit(
                        _recompute_loo_row,
                        (
                            row_index,
                            row,
                            n_initial_trials_dropped,
                            pareto_k_threshold,
                        ),
                    ): row
                    for row_index, row in queued_entries
                }
                recomputed_entries = []
                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    try:
                        recomputed_entries.append(future.result())
                    except Exception as exc:
                        raise RuntimeError(
                            "Failed to recompute LOO for "
                            f"replicate={row['replicate']}, simulated_model={row['simulated_model']}, "
                            f"fitted_model={row['fitted_model']}"
                        ) from exc
        else:
            recomputed_entries = [
                _recompute_loo_row((row_index, row, n_initial_trials_dropped, pareto_k_threshold))
                for row_index, row in queued_entries
            ]

        for row_index, updated_row, n_trials_used in recomputed_entries:
            n_total_trials = int(n_trials_used + n_initial_trials_dropped)
            if n_initial_trials_dropped > 0:
                console.print(
                    f"  [dim]Dropped {n_initial_trials_dropped} initial trial(s) "
                    f"({n_trials_used} of {n_total_trials} trials used)[/dim]"
                )
            if updated_row["n_bad_pareto_k"] > 0:
                console.print(
                    f"  [yellow]⚠ {updated_row['n_bad_pareto_k']} Pareto k > {pareto_k_threshold} "
                    f"(max = {updated_row['max_pareto_k']:.3f})[/yellow]"
                )
            else:
                console.print(f"  [green]✓ Pareto k OK[/green] (max = {updated_row['max_pareto_k']:.3f})")
            resolved_entries.append((row_index, updated_row, n_trials_used))

    resolved_entries.sort(key=lambda entry: entry[0])
    updated_rows = [entry[1] for entry in resolved_entries]
    n_trials_used_values = {entry[2] for entry in resolved_entries}
    return updated_rows, n_trials_used_values


def main() -> None:
    """Compile the recovery fits into tables and a summary file.

    Loads all fit summaries, gets current LOO values, checks that every
    replicate, simulated-model and fitted-model fit is present, ranks every
    fit, picks the winner for each simulated dataset, and writes the tables
    (including the group-effect recovery table) and
    ``model_recovery_summary.json``.

    Raises:
        ValueError: If ``--loo-workers`` is less than 1, if saved LOO values
            are required but out of date, if any fit is outside the expected
            replicates and models or appears twice, or if any fit has a
            missing or non-finite ``elpd_loo``.
        RuntimeError: If there are no results, if any fit is missing, or if
            the output folder cannot be used.
        FileNotFoundError: If a fit of a model to its own data lacks any of
            its saved files.
    """
    args = parse_args()
    if args.loo_workers < 1:
        raise ValueError(f"--loo-workers must be at least 1, got {args.loo_workers}")

    console.print("\n[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]   Integrated Model Recovery Compilation[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]\n")

    results_dir = Path(args.results_dir)
    output_dir = _resolve_output_dir(results_dir, args.output_dir)
    _prepare_output_dir(output_dir, args.overwrite)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Results directory:[/cyan] {results_dir}")
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Pareto-k threshold:[/cyan] {args.pareto_k_threshold}")
    console.print(f"[cyan]Initial trials dropped:[/cyan] {args.loo_drop_initial_trials}\n")
    console.print(f"[cyan]LOO source:[/cyan] {args.loo_source}\n")
    console.print(f"[cyan]LOO workers:[/cyan] {args.loo_workers}\n")

    rows = _load_rows(results_dir)
    if not rows:
        console.print(f"[red]Error: No integrated model-recovery results found in {results_dir}[/red]")
        raise RuntimeError(f"No integrated model-recovery results found in {results_dir}")

    console.print(f"[bold]Processing {len(rows)} fit summaries...[/bold]")

    updated_rows, n_trials_used_values = _resolve_loo_rows(
        rows,
        loo_source=args.loo_source,
        loo_workers=args.loo_workers,
        n_initial_trials_dropped=args.loo_drop_initial_trials,
        pareto_k_threshold=args.pareto_k_threshold,
    )

    loo_df = pd.DataFrame(updated_rows)
    model_order = args.models or get_default_models(args.config)
    console.print(f"[cyan]Models:[/cyan] {', '.join(model_order)}")
    console.print(f"[cyan]Expected replicates:[/cyan] {args.replicates}")
    missing = validate_recovery_fit_grid(
        loo_df,
        model_order=model_order,
        expected_replicates=args.replicates,
    )
    if missing:
        console.print("[red]Error: Missing integrated recovery fit summaries.[/red]")
        missing_text = ", ".join(
            f"(replicate={replicate}, simulated_model={simulated_model}, fitted_model={fitted_model})"
            for replicate, simulated_model, fitted_model in missing
        )
        raise RuntimeError(f"Missing integrated recovery fit summaries: {missing_text}")

    ranked_df = rank_recovery_fits(loo_df)
    ranked_df = ranked_df.sort_values(
        ["replicate", "simulated_model", "loo_rank_within_dataset", "elpd_loo"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    ranked_df.to_csv(tables_dir / "model_recovery_loo.csv", index=False)
    console.print("[green]✓[/green] Saved tables/model_recovery_loo.csv")

    winners_df = select_recovery_winners(loo_df)
    winners_df.to_csv(tables_dir / "model_recovery_winners.csv", index=False)
    console.print("[green]✓[/green] Saved tables/model_recovery_winners.csv")

    parameter_recovery_df, parameter_recovery_summary_df = build_integrated_parameter_recovery(
        ranked_df,
        results_dir=results_dir,
    )
    if not parameter_recovery_df.empty:
        parameter_recovery_df.to_csv(tables_dir / "parameter_recovery.csv", index=False)
        parameter_recovery_summary_df.to_csv(tables_dir / "parameter_recovery_summary.csv", index=False)
        console.print("[green]✓[/green] Saved tables/parameter recovery outputs")
    else:
        console.print("[dim]No parameter recovery outputs were generated.[/dim]")

    group_effect_df = build_integrated_group_effect_recovery(
        ranked_df,
        results_dir=results_dir,
    )
    group_effect_df.to_csv(tables_dir / "group_effect_recovery.csv", index=False)
    console.print("[green]✓[/green] Saved tables/group_effect_recovery.csv")

    confusion_df, inversion_df = build_recovery_matrices(winners_df, model_order=model_order)
    confusion_df.to_csv(tables_dir / "model_recovery_confusion_matrix.csv")
    inversion_df.to_csv(tables_dir / "model_recovery_inversion_matrix.csv")

    summary = {
        "results_dir": str(results_dir),
        "n_fit_rows": int(len(ranked_df)),
        "n_winners": int(len(winners_df)),
        "n_parameter_recovery_rows": int(len(parameter_recovery_df)),
        "n_parameter_recovery_summary_rows": int(len(parameter_recovery_summary_df)),
        "n_group_effect_rows": int(len(group_effect_df)),
        **summarize_recovery_dataset_counts(ranked_df, winners_df),
        "n_winners_not_converged": int((~winners_df["converged"].astype(bool)).sum()),
        "n_winners_with_high_pareto_k": int((winners_df["n_bad_pareto_k"] > 0).sum()),
        "pareto_k_threshold": float(args.pareto_k_threshold),
        "loo_drop_initial_trials": int(args.loo_drop_initial_trials),
        "loo_workers": int(args.loo_workers),
        "n_trials_used": int(next(iter(n_trials_used_values))) if len(n_trials_used_values) == 1 else sorted(n_trials_used_values),
        "winner_rule": (
            "Highest elpd_loo among all fitted models for each simulated data set. "
            "Fits that did not converge or have Pareto k above the threshold are included; "
            "their convergence and Pareto k are recorded in model_recovery_loo.csv."
        ),
    }
    with open(output_dir / "model_recovery_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    console.print(f"[green]✓[/green] Saved {output_dir / 'model_recovery_summary.json'}")


if __name__ == "__main__":
    main()
