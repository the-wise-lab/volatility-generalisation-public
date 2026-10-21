#!/usr/bin/env python3
"""Compile integrated model-selection outputs into comparison tables and figures.

This is the last step of the integrated model-selection workflow, run after
every candidate model has been fitted to the four second-block conditions
together. The script loads each model's saved fit, computes leave-one-out
(LOO) cross-validation scores (optionally also WAIC) after dropping the first
trial or trials, and compares the models with stacking weights. It writes the
LOO table, the model comparison table, a Pareto-k table, and figures of the
stacking weights and model ranking. It fails if a model's fit is missing.
See ``docs/3-model-selection-and-recovery.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from rich.console import Console
from rich.table import Table
from volatility_generalisation.modelling_code.integrated_model_selection import get_default_models

matplotlib.use("Agg")

console = Console()


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options: config and results paths, output folder, models,
        overwrite flag, whether to include WAIC, and the number of initial
        trials to drop from LOO and WAIC.
    """
    parser = argparse.ArgumentParser(description="Compile integrated model-selection outputs")
    parser.add_argument("--config", default="scripts/config.toml")
    parser.add_argument("--results-dir", default="results/integrated_model_selection")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Derived output directory (default: results/integrated_model_selection/model_comparison)",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty derived output directory",
    )
    parser.add_argument("--include-waic", action="store_true")
    parser.add_argument(
        "--loo-drop-initial-trials", type=int, default=1,
        help="Drop N initial trials from LOO/WAIC computation (default: 1). "
             "Set to 0 to include all trials.",
    )
    return parser.parse_args()


def load_fit(result_dir: Path) -> tuple[dict, az.InferenceData]:
    """Load the saved outputs of one integrated model fit.

    Args:
        result_dir: Folder for one model, holding ``metadata.json``,
            ``diagnostics.json`` and ``posterior.nc``.

    Returns:
        Tuple of the metadata, with the diagnostics added under
        ``"diagnostics"``, and the posterior samples with their
        log-likelihood.

    Raises:
        FileNotFoundError: If any of the three files is missing.
    """
    metadata_path = result_dir / "metadata.json"
    diagnostics_path = result_dir / "diagnostics.json"
    posterior_path = result_dir / "posterior.nc"
    missing = [
        path.name for path in [metadata_path, diagnostics_path, posterior_path] if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing expected integrated fit outputs in {result_dir}: {', '.join(missing)}"
        )

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(diagnostics_path, "r", encoding="utf-8") as handle:
        metadata["diagnostics"] = json.load(handle)
    return metadata, az.from_netcdf(posterior_path)


def _load_expected_fit(results_dir: Path, model_name: str) -> tuple[dict, az.InferenceData]:
    """Load the fit of one model that must be present.

    Args:
        results_dir: Folder with one subfolder per model.
        model_name: Name of the model to load.

    Returns:
        Tuple of the metadata with diagnostics and the posterior samples, as
        returned by ``load_fit``.

    Raises:
        FileNotFoundError: If the model's outputs are missing. The message
            names the model and the folder.
    """
    result_dir = results_dir / model_name
    try:
        return load_fit(result_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing expected integrated model-selection output for model '{model_name}' in {result_dir}: {exc}"
        ) from exc


def _resolve_models(
    default_names: list[str],
    explicit_models: list[str] | None,
) -> list[str]:
    """Choose which models to compile.

    Args:
        default_names: Models listed in the config file.
        explicit_models: Models given on the command line, or None to use the
            config list.

    Returns:
        The models to compile.
    """
    if explicit_models is not None:
        return explicit_models
    return list(default_names)


def _resolve_output_dir(
    results_dir: Path,
    output_dir: str | None,
) -> Path:
    """Choose the folder where the compiled outputs are written.

    Args:
        results_dir: Folder with the model fits.
        output_dir: Folder given on the command line, or None to use the
            ``model_comparison`` folder inside the results folder.

    Returns:
        Path of the output folder.
    """
    if output_dir is not None:
        return Path(output_dir)
    return results_dir / "model_comparison"


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


def _mask_initial_trials(
    idata: az.InferenceData, n_initial: int
) -> az.InferenceData:
    """Remove the first trials from the log-likelihood before computing LOO.

    Args:
        idata: Fit results holding the pointwise log-likelihood ``obs``, with
            a ``trial`` dimension.
        n_initial: Number of trials to drop from the start of the data. Zero
            or a negative number keeps every trial.

    Returns:
        The fit results with the initial trials removed from the
        log-likelihood, or the input unchanged if ``n_initial`` is zero or
        negative.

    Raises:
        ValueError: If ``n_initial`` is not less than the number of trials.
    """
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


def build_loo_row(
    model_name: str,
    metadata: dict,
    loo_result: object,
    n_initial_trials_dropped: int = 0,
) -> dict[str, object]:
    """Build one row of the LOO comparison table for a model.

    Args:
        model_name: Name of the model.
        metadata: Fit metadata, with the fit diagnostics under
            ``"diagnostics"``.
        loo_result: LOO result from ArviZ, computed with pointwise values.
        n_initial_trials_dropped: Number of initial trials left out of the LOO
            calculation.

    Returns:
        Dictionary with the LOO estimate and its standard error, the effective
        number of parameters, whether the fit converged, the number of
        divergences, the number of trials with Pareto k above 0.5 and above
        0.7, the largest Pareto k, and the number of trials used.
    """
    pareto_k = loo_result.pareto_k.values
    diagnostics = metadata.get("diagnostics", {})
    return {
        "model": model_name,
        "elpd_loo": float(loo_result.elpd_loo),
        "se": float(loo_result.se),
        "p_loo": float(loo_result.p_loo),
        "converged": bool(diagnostics.get("converged", False)),
        "divergences": int(diagnostics.get("divergences", 0) or 0),
        "n_pareto_k_gt_0_5": int((pareto_k > 0.5).sum()),
        "n_pareto_k_gt_0_7": int((pareto_k > 0.7).sum()),
        "max_pareto_k": float(pareto_k.max()),
        "n_trials_used": int(pareto_k.size),
        "n_initial_trials_dropped": n_initial_trials_dropped,
    }


def _save_weights_figure(compare_df: pd.DataFrame, output_dir: Path) -> None:
    """Save a bar chart of the stacking weight of each model.

    Args:
        compare_df: Model comparison table with a ``weight`` column and one
            row per model.
        output_dir: Folder in which to save ``stacking_weights.png``.
    """
    plt.figure(figsize=(8, 4))
    compare_df["weight"].sort_values(ascending=False).plot(kind="bar", color="#4C78A8")
    plt.ylabel("Stacking weight")
    plt.title("Integrated Model Selection Weights")
    plt.tight_layout()
    plt.savefig(output_dir / "stacking_weights.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_ranking_figure(compare_df: pd.DataFrame, output_dir: Path) -> None:
    """Save a bar chart of the models ranked by their LOO estimate.

    Args:
        compare_df: Model comparison table with one row per model and a column
            whose name contains ``elpd``.
        output_dir: Folder in which to save ``model_ranking.png``.
    """
    elpd_col = next(column for column in compare_df.columns if "elpd" in column)
    ranked = compare_df.sort_values(elpd_col, ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(ranked.index, ranked[elpd_col], color="#72B7B2")
    plt.ylabel(elpd_col)
    plt.title("Integrated Model Ranking")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_pareto_table(
    loo_rows: list[dict[str, object]], output_dir: Path
) -> None:
    """Save a table of the Pareto k counts and maximum for each model.

    Args:
        loo_rows: LOO rows as returned by ``build_loo_row``, one per model.
        output_dir: Folder in which to save ``pareto_k_table.csv``.
    """
    rows = [
        {
            "model": row["model"],
            "pareto_k_gt_0_5": int(row["n_pareto_k_gt_0_5"]),
            "pareto_k_gt_0_7": int(row["n_pareto_k_gt_0_7"]),
            "pareto_k_max": float(row["max_pareto_k"]),
        }
        for row in loo_rows
    ]
    pd.DataFrame(rows).to_csv(output_dir / "pareto_k_table.csv", index=False)


def main() -> None:
    """Compile the fits of all candidate models into comparison outputs.

    Loads each model's fit, computes LOO (and WAIC if requested) with the
    initial trials dropped, compares the models with stacking, and writes the
    tables and figures.

    Raises:
        FileNotFoundError: If the fit of an expected model is missing.
        RuntimeError: If no models are selected, no results are found, or the
            output folder cannot be used.
        ValueError: If the number of initial trials to drop is not less than
            the number of trials.
    """
    console.print("\n[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]   Integrated Model Selection Compilation[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]\n")

    args = parse_args()
    results_dir = Path(args.results_dir)

    console.print(f"[cyan]Results directory:[/cyan] {results_dir}")

    default_names = get_default_models(args.config)
    expected_models = _resolve_models(default_names, args.models)
    if not expected_models:
        console.print(
            "[red]Error: No models matched the selected criteria.[/red]"
        )
        raise RuntimeError(
            "No models matched the selected criteria."
        )

    console.print(
        f"[bold]Processing {len(expected_models)} models:[/bold] "
        f"{', '.join(expected_models)}\n"
    )

    # Resolve output directory
    output_dir = _resolve_output_dir(
        results_dir,
        args.output_dir,
    )
    _prepare_output_dir(output_dir, args.overwrite)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]  Tables:[/cyan] {tables_dir}")
    console.print(f"[cyan]  Figures:[/cyan] {figures_dir}\n")

    loo_results: dict[str, object] = {}
    loo_rows = []
    waic_rows = []

    for i, model_name in enumerate(expected_models, start=1):
        console.print(
            f"[bold cyan][{i}/{len(expected_models)}][/bold cyan] "
            f"Loading [yellow]{model_name}[/yellow]..."
        )
        try:
            metadata, idata = _load_expected_fit(results_dir, model_name)
        except FileNotFoundError as exc:
            console.print(f"[red]  ✗ Failed to load {model_name}: {exc}[/red]")
            raise

        console.print("  [green]✓[/green] Loaded posterior, computing LOO...")

        n_drop = args.loo_drop_initial_trials
        if n_drop > 0:
            n_total = idata.log_likelihood["obs"].sizes["trial"]
            idata_for_ic = _mask_initial_trials(idata, n_drop)
            console.print(
                f"  [dim]Dropped {n_drop} initial trial(s) from IC computation "
                f"({n_total - n_drop} of {n_total} trials used)[/dim]"
            )
        else:
            idata_for_ic = idata

        loo_result = az.loo(idata_for_ic, pointwise=True)
        loo_results[model_name] = loo_result
        loo_row = build_loo_row(model_name, metadata, loo_result, n_initial_trials_dropped=n_drop)
        pareto_gt_07 = loo_row["n_pareto_k_gt_0_7"]
        max_pk = loo_row["max_pareto_k"]
        if pareto_gt_07 > 0:
            console.print(
                f"  [yellow]⚠ {pareto_gt_07} Pareto k > 0.7 "
                f"(max = {max_pk:.3f})[/yellow]"
            )
        else:
            console.print(
                f"  [green]✓ Pareto k OK[/green] (max = {max_pk:.3f})"
            )

        loo_rows.append(loo_row)

        if args.include_waic:
            waic_result = az.waic(idata_for_ic, pointwise=True)
            waic_rows.append(
                {
                    "model": model_name,
                    "elpd_waic": float(waic_result.elpd_waic),
                    "se": float(waic_result.se),
                    "p_waic": float(waic_result.p_waic),
                }
            )

        del idata
        gc.collect()

    if not loo_results:
        console.print(
            f"[red]Error: No integrated model-selection results "
            f"found in {results_dir}[/red]"
        )
        raise RuntimeError(
            f"No integrated model-selection results found in {results_dir}"
        )

    console.print("\n[bold cyan]Sorting and saving results...[/bold cyan]")

    loo_df = pd.DataFrame(loo_rows).sort_values("elpd_loo", ascending=False)
    loo_df.to_csv(tables_dir / "model_selection_loo.csv", index=False)
    console.print("[green]✓ Saved[/green] tables/model_selection_loo.csv")

    title = "LOO Comparison"
    if args.loo_drop_initial_trials > 0:
        title += f" (first {args.loo_drop_initial_trials} trial(s) excluded)"
    loo_table = Table(
        title=title, show_header=True, header_style="bold magenta"
    )
    loo_table.add_column("Model", style="cyan")
    loo_table.add_column("ELPD", justify="right")
    loo_table.add_column("SE", justify="right")
    loo_table.add_column("p_LOO", justify="right")
    loo_table.add_column("Converged", justify="center")
    loo_table.add_column("k > 0.7", justify="right")
    loo_table.add_column("Max k", justify="right")

    for _, row in loo_df.iterrows():
        converged_str = (
            "[green]✓[/green]"
            if row["converged"]
            else "[red]✗[/red]"
        )
        k_07_str = (
            f"[yellow]{int(row['n_pareto_k_gt_0_7'])}[/yellow]"
            if row["n_pareto_k_gt_0_7"] > 0
            else "0"
        )
        loo_table.add_row(
            row["model"],
            f"{row['elpd_loo']:.2f}",
            f"{row['se']:.2f}",
            f"{row['p_loo']:.2f}",
            converged_str,
            k_07_str,
            f"{row['max_pareto_k']:.3f}",
        )
    console.print(loo_table)

    console.print("\n[bold cyan]Comparing models via stacking...[/bold cyan]")
    compare_df = az.compare(loo_results, ic="loo", method="stacking")
    compare_df.to_csv(tables_dir / "model_selection_compare.csv")
    console.print("[green]✓ Saved[/green] tables/model_selection_compare.csv")

    _save_weights_figure(compare_df, figures_dir)
    console.print("[green]✓ Saved[/green] figures/stacking_weights.png")

    _save_ranking_figure(compare_df, figures_dir)
    console.print("[green]✓ Saved[/green] figures/model_ranking.png")

    _save_pareto_table(loo_rows, tables_dir)
    console.print("[green]✓ Saved[/green] tables/pareto_k_table.csv")

    if args.include_waic:
        waic_df = pd.DataFrame(waic_rows).sort_values(
            "elpd_waic", ascending=False
        )
        waic_df.to_csv(tables_dir / "model_selection_waic.csv", index=False)
        console.print("[green]✓ Saved[/green] tables/model_selection_waic.csv")

    console.print("\n[bold green]Compilation complete![/bold green]")
    console.print(f"[bold]Outputs saved to:[/bold] {output_dir.absolute()}\n")


if __name__ == "__main__":
    main()
