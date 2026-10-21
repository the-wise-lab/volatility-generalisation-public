"""
Publication Figures Script

Generates the task-performance statistics and the three agreed publication
figures from the prepared data and existing analysis outputs.
Run with ``--help`` for usage.

Usage::

    uv run python scripts/make_publication_figures.py \\
        --output-dir results/publication_figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from volatility_generalisation.figures.publication_figures import (
    CONDITION_ORDER,
    OUTPUT_FORMATS,
    compute_condition_population_summary,
    compute_learning_rate_subject_summary,
    compute_logit_coefficient_summary,
    compute_marginal_effects,
    compute_ppc_summary,
    condition_labels,
    condition_order,
    extract_group_params,
    extract_subject_params,
    compute_figure1_bracket_stats,
    compute_task_performance,
    ensure_dm_sans_font,
    load_comparison_for_figure2,
    load_posterior_and_metadata,
    load_single_lr_subject_estimates,
    plot_figure1,
    plot_figure2,
    plot_figure3,
    summarise_task_performance,
)
from volatility_generalisation.modelling_code.condition_metadata import (
    CONDITION_METADATA,
)
from volatility_generalisation.modelling_code.data_preparation import (
    normalize_to_reward_coded,
)
# ---------------------------------------------------------------------------
# Prepared-data loading (Figure 2)
# ---------------------------------------------------------------------------


def _load_prepared_ppc_data(path: Path) -> Dict[str, np.ndarray]:
    """Load the arrays needed for the predicted-choice plots from a prepared file.

    Reads only the numeric arrays ``outcomes_0``, ``outcomes_1`` and ``choices``,
    and the outcome-coding label if the file has one. It skips the participant
    IDs, which are stored as Python objects.

    Args:
        path: Path to the prepared ``.npz`` file for one condition.

    Returns:
        Dictionary with the arrays ``outcomes_0``, ``outcomes_1`` and
        ``choices``, and the text label ``outcome_coding`` if the file has one.

    Raises:
        ValueError: If the outcome-coding label is not a single value.
    """
    required_keys = ("outcomes_0", "outcomes_1", "choices")
    with np.load(path) as f:
        data = {key: f[key] for key in required_keys}
        if "outcome_coding" in f:
            metadata = np.asarray(f["outcome_coding"])
            if metadata.ndim != 0:
                raise ValueError(
                    f"Prepared archive '{path}' has non-scalar outcome_coding metadata."
                )
            value = metadata.item()
            data["outcome_coding"] = (
                value.decode() if isinstance(value, bytes) else str(value)
            )
        return data


def _load_prepared_performance_data(path: Path) -> Dict[str, np.ndarray]:
    """Load the arrays needed for task performance from a prepared ``.npz`` file.

    Participant IDs are stored as Python objects, so the file is opened with
    ``allow_pickle=True``. This is only used for the project's own prepared
    files written by ``scripts/prepare_data.py``. Outcomes are converted to
    reward coding (1 = safe) when needed.

    Args:
        path: Path to the prepared ``.npz`` file for one condition.

    Returns:
        Dictionary with reward-coded ``outcomes_0`` and ``outcomes_1``,
        ``choices`` and ``subject_ids``.

    Raises:
        ValueError: If the outcome-coding metadata is not a single value.
    """
    with np.load(path, allow_pickle=True) as f:
        data = {key: f[key] for key in ("outcomes_0", "outcomes_1", "choices")}
        subject_ids = np.asarray(f["subject_ids"])
        metadata = None
        if "outcome_coding" in f:
            coding = np.asarray(f["outcome_coding"])
            if coding.ndim != 0:
                raise ValueError(
                    f"Prepared archive '{path}' has non-scalar outcome_coding metadata."
                )
            value = coding.item()
            metadata = {
                "outcome_coding": value.decode() if isinstance(value, bytes) else str(value)
            }
    data = normalize_to_reward_coded(data, metadata)
    data["subject_ids"] = subject_ids
    return data


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Read the command-line options of the publication figures script.

    The options give the paths of the input files and folders and of the output
    folder. Each has a default path inside the project.

    Args:
        argv: Command-line arguments to read. If ``None``, the arguments the
            script was started with are used.

    Returns:
        The parsed options, with the options ``single_lr_csv``,
        ``comparison_dir``, ``posterior_dir``, ``prepared_data_dir`` and
        ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Generate publication figures from analysis outputs.",
    )

    parser.add_argument(
        "--single-lr-csv",
        default="results/compiled/subject_parameters.csv",
        help="Path to single-learning-rate subject estimates CSV "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--comparison-dir",
        default="results/integrated_model_selection/model_comparison/tables",
        help="Directory containing model comparison tables "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--posterior-dir",
        default="results/integrated_model_selection/dual_lr_single_value",
        help="Directory containing the main model's posterior.nc and "
             "metadata.json (default: %(default)s).",
    )
    parser.add_argument(
        "--prepared-data-dir",
        default="data/prepared",
        help="Directory containing prepared data .npz files "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/publication_figures",
        help="Root output directory for figures and source tables "
             "(default: %(default)s).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Source-table writers
# ---------------------------------------------------------------------------

def _write_source_table(df: pd.DataFrame, path: Path, label: str) -> None:
    """Save a table as a CSV file and print where it went.

    The folder is created if it does not exist.

    Args:
        df: Table to save.
        path: Path of the CSV file.
        label: Name of the table, shown in the message that is printed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [source table] {label} → {path}")


def _write_figure1_source(
    subject_estimates: pd.DataFrame, tables_dir: Path
) -> pd.DataFrame:
    """Build the table of data behind Figure 1 and save it as a CSV file.

    The table keeps the four second-block conditions and adds a participant
    number, the condition label and the figure name. It is saved as
    ``figure-1-source.csv``.

    Args:
        subject_estimates: Table with one row per participant and condition,
            with the columns ``subject_id``, ``condition`` and
            ``learning_rate``, and optionally ``model``.
        tables_dir: Folder for the CSV file.

    Returns:
        The table that was saved.
    """
    columns = ["subject_id", "condition", "learning_rate"]
    if "model" in subject_estimates.columns:
        columns.append("model")
    src = subject_estimates.loc[
        subject_estimates["condition"].isin(CONDITION_ORDER),
        columns,
    ].copy()
    subject_numbers = {
        subject_id: number
        for number, subject_id in enumerate(sorted(src["subject_id"].unique(), key=str), 1)
    }
    src["subject_number"] = src["subject_id"].map(subject_numbers)
    src["condition_label"] = src["condition"].map(condition_labels())
    src["figure"] = "figure-1"
    _write_source_table(src, tables_dir / "figure-1-source.csv", "Figure 1")
    return src


def _write_figure2_source(
    comparison: pd.DataFrame, tables_dir: Path
) -> pd.DataFrame:
    """Build the table of model-comparison data behind Figure 2 and save it.

    The table is saved as ``figure-2-source.csv``.

    Args:
        comparison: Model-comparison table from ``load_comparison_for_figure2``.
        tables_dir: Folder for the CSV file.

    Returns:
        The table that was saved.
    """
    src = comparison[["model", "elpd_diff", "se_diff", "stacking_weight"]].copy()
    src["model_display"] = comparison["model_display"]
    src["y_position"] = comparison["y_position"]
    src["figure"] = "figure-2"
    _write_source_table(src, tables_dir / "figure-2-source.csv", "Figure 2")
    return src


def _write_figure2_ppc_source(
    ppc_by_condition: Dict[str, Dict[str, np.ndarray]],
    tables_dir: Path,
) -> pd.DataFrame:
    """Build the table of predicted and observed choices behind Figure 2.

    The table has one row per condition and trial, and is saved as
    ``figure-2-ppc-source.csv``.

    Args:
        ppc_by_condition: Dictionary that maps each condition name to its
            summary from ``compute_ppc_summary``.
        tables_dir: Folder for the CSV file.

    Returns:
        The table that was saved.
    """
    rows = []
    for cond, ppc in ppc_by_condition.items():
        n = len(ppc["trial_mean_pred"])
        for t in range(n):
            rows.append({
                "figure": "figure-2",
                "condition": cond,
                "condition_label": condition_labels().get(cond, cond),
                "trial_number": t + 1,
                "observed_p_choice_0": float(ppc["trial_mean_obs"][t]),
                "predicted_mean": float(ppc["trial_mean_pred"][t]),
                "ci_lower": float(ppc["ci_lower"][t]),
                "ci_upper": float(ppc["ci_upper"][t]),
            })
    src = pd.DataFrame(rows)
    _write_source_table(src, tables_dir / "figure-2-ppc-source.csv", "Figure 2 PPC")
    return src


def _write_figure3_source(
    subject_summary: pd.DataFrame,
    population_summary: Dict[str, Dict[str, float]],
    effects: Dict[str, pd.DataFrame],
    tables_dir: Path,
) -> pd.DataFrame:
    """Build the tables of data behind Figure 3 and save them as CSV files.

    Three files are saved: ``figure-3-source.csv`` (each participant's mean
    learning rates), ``figure-3-population.csv`` (population means and
    intervals by condition) and ``figure-3-effects.csv`` (effects of current and
    previous volatility and their interaction).

    Args:
        subject_summary: Table from ``compute_learning_rate_subject_summary``.
        population_summary: Nested dictionary from
            ``compute_condition_population_summary``.
        effects: Dictionary from ``compute_marginal_effects``, with one table
            per learning-rate parameter.
        tables_dir: Folder for the CSV files.

    Returns:
        The participant table that was saved as ``figure-3-source.csv``.
    """
    # Subject-level source
    src = subject_summary.copy()
    src["subject_number"] = src["subject_idx"] + 1
    src["figure"] = "figure-3"
    _write_source_table(src, tables_dir / "figure-3-source.csv", "Figure 3 subjects")

    # Population summary (condition-specific)
    pop_rows = []
    for cond, cond_stats in population_summary.items():
        for key, val in cond_stats.items():
            pop_rows.append({
                "figure": "figure-3",
                "condition": cond,
                "parameter": key,
                "estimate": float(val),
            })
    pop_df = pd.DataFrame(pop_rows)
    _write_source_table(
        pop_df, tables_dir / "figure-3-population.csv", "Figure 3 population"
    )

    # Effects source
    rows = []
    for param, ef in effects.items():
        for col in ef.columns:
            rows.append({
                "figure": "figure-3",
                "parameter": param,
                "effect": col,
                "estimate": float(ef[col].iloc[0]),
            })
    effects_df = pd.DataFrame(rows)
    _write_source_table(
        effects_df, tables_dir / "figure-3-effects.csv", "Figure 3 effects"
    )
    return src


def _make_task_performance_outputs(prepared_dir: Path, tables_dir: Path) -> pd.DataFrame:
    """Write the task-performance tables and print the statistics.

    Args:
        prepared_dir: Directory holding one prepared ``.npz`` file per condition.
        tables_dir: Directory for the tables.

    Returns:
        The task-performance summary table.

    Raises:
        FileNotFoundError: If a condition's prepared file is missing.
    """
    prepared_by_condition = {}
    for condition in CONDITION_METADATA:
        path = prepared_dir / f"{condition}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Prepared data for condition '{condition}' not found at '{path}'. "
                "Cannot compute task performance."
            )
        prepared_by_condition[condition] = _load_prepared_performance_data(path)

    participants, schedule_check = compute_task_performance(prepared_by_condition)
    summary = summarise_task_performance(participants)

    _write_source_table(
        participants, tables_dir / "task-performance-source.csv", "Task performance"
    )
    _write_source_table(
        summary, tables_dir / "task-performance-summary.csv", "Task performance summary"
    )
    _write_source_table(
        schedule_check,
        tables_dir / "task-performance-schedule-check.csv",
        "Task performance schedule check",
    )
    by_measure = summary.set_index("measure")
    for measure, label in (("optimal_choices", "Optimal choices"),
                           ("safe_outcomes", "Safe outcomes")):
        row = by_measure.loc[measure]
        print(
            f"  {label}: mean {100 * row['mean']:.1f}% (SD {100 * row['sd']:.1f}%, "
            f"95% CI {100 * row['ci_lower']:.1f}-{100 * row['ci_upper']:.1f}%), "
            f"t({int(row['df'])}) = {row['t']:.2f}, p = {row['p_value']:.3g}"
        )
    print(
        f"  Best possible safe outcomes: "
        f"{100 * by_measure.loc['safe_outcomes', 'best_possible']:.1f}%"
    )
    optimal = by_measure.loc["optimal_choices"]
    print(
        f"  Participants above chance: {int(optimal['n_above_chance'])} of "
        f"{int(optimal['n_participants'])} ({optimal['percent_above_chance']:.1f}%)\n"
    )
    return summary


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Make the task-performance statistics and Figures 1 to 3.

    The function loads the estimates, model comparison and posterior samples,
    saves the tables behind each figure, and saves each figure as PDF, SVG and
    PNG. It stops at the first missing input file or folder.

    Args:
        args: Parsed command-line options from ``parse_args``.

    Raises:
        FileNotFoundError: If an input file or folder is missing.
        ValueError: If the model-comparison table has no retained models, if a
            condition is missing from the posterior, or if no condition has
            predicted choices to plot.
    """

    ensure_dm_sans_font()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Generating the task-performance statistics and Figures 1–3")
    print(f"Output: {output_dir}\n")

    # ---- Task performance ----
    print("--- Task performance ---")
    prepared_dir = Path(args.prepared_data_dir)
    if not prepared_dir.is_dir():
        raise FileNotFoundError(
            f"Prepared data directory not found at '{prepared_dir}'. "
            "Cannot compute task performance."
        )
    _make_task_performance_outputs(prepared_dir, tables_dir)

    # ---- Figure 1: Single learning rate ----
    print("--- Figure 1: Single learning rate ---")
    single_lr_path = Path(args.single_lr_csv)
    if not single_lr_path.is_file():
        raise FileNotFoundError(
            f"Single-LR CSV not found at '{args.single_lr_csv}'. "
            "Cannot generate Figure 1."
        )
    subject_estimates = load_single_lr_subject_estimates(args.single_lr_csv)
    _write_figure1_source(subject_estimates, tables_dir)
    comparisons = compute_figure1_bracket_stats(subject_estimates)
    _write_source_table(
        comparisons, tables_dir / "figure-1-comparisons.csv", "Figure 1 comparisons"
    )
    plot_figure1(
        subject_estimates,
        str(figures_dir / "figure-1-single-learning-rate"),
    )
    print("  [figure] figure-1-single-learning-rate.{pdf,svg,png}\n")

    # ---- Figure 2: Model comparison and posterior predictive checks ----
    print("--- Figure 2: Model comparison and posterior predictive checks ---")
    comparison_path = Path(args.comparison_dir) / "model_selection_compare.csv"
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"Model comparison table not found at '{comparison_path}'. "
            "Expected ArviZ `az.compare` output CSV. "
            "Cannot generate Figure 2."
        )
    comparison_raw = pd.read_csv(comparison_path)
    comparison = load_comparison_for_figure2(comparison_raw)
    if comparison.empty:
        raise ValueError(
            "No retained models found in the comparison table. "
            "Cannot generate Figure 2."
        )
    _write_figure2_source(comparison, tables_dir)

    posterior_dir = Path(args.posterior_dir)
    posterior_path = posterior_dir / "posterior.nc"
    if not posterior_path.is_file():
        raise FileNotFoundError(
            f"Posterior file not found at '{posterior_path}'. "
            "Expected ArviZ NetCDF file (posterior.nc). "
            "Cannot generate Figure 2."
        )

    # Load integrated posterior
    loaded = load_posterior_and_metadata(posterior_path)
    posterior_ds = loaded["posterior"]
    cond_indices = loaded["cond_indices"]
    conditions = loaded["conditions"]

    # Extract subject-level params per condition
    subj_by_cond = extract_subject_params(posterior_ds, cond_indices)

    # Find prepared data files
    prepared_dir = Path(args.prepared_data_dir)
    if not prepared_dir.is_dir():
        raise FileNotFoundError(
            f"Prepared data directory not found at '{prepared_dir}'. "
            "Cannot generate Figure 2."
        )

    ppc_by_condition: Dict[str, Dict[str, np.ndarray]] = {}
    for cond in condition_order():
        if cond not in subj_by_cond:
            raise ValueError(
                f"Condition '{cond}' not found in integrated posterior. "
                "Expected all four second-block conditions to be present."
            )

        # Find matching prepared data file
        cond_prepared_path = prepared_dir / f"{cond}.npz"
        if not cond_prepared_path.is_file():
            raise FileNotFoundError(
                f"Prepared data for condition '{cond}' not found at "
                f"'{cond_prepared_path}'. Cannot generate Figure 2."
            )

        prepared_data = _load_prepared_ppc_data(cond_prepared_path)
        outcome_metadata = (
            {"outcome_coding": prepared_data.pop("outcome_coding")}
            if "outcome_coding" in prepared_data else None
        )
        prepared_data = normalize_to_reward_coded(prepared_data, outcome_metadata)
        ppc = compute_ppc_summary(subj_by_cond[cond], prepared_data)
        ppc_by_condition[cond] = ppc

    if not ppc_by_condition:
        raise ValueError(
            "No conditions had both posterior and prepared data. "
            "Cannot generate Figure 2."
        )

    _write_figure2_ppc_source(ppc_by_condition, tables_dir)
    plot_figure2(
        comparison,
        ppc_by_condition,
        str(figures_dir / "figure-2-model-comparison-and-ppc"),
    )
    print("  [figure] figure-2-model-comparison-and-ppc.{pdf,svg,png}\n")

    # ---- Figure 3: Main-model learning rates ----
    print("--- Figure 3: Main-model learning rates ---")

    posterior_path = Path(args.posterior_dir) / "posterior.nc"
    if not posterior_path.is_file():
        raise FileNotFoundError(
            f"Posterior file not found at '{posterior_path}'. "
            "Expected ArviZ NetCDF file (posterior.nc). "
            "Cannot generate Figure 3."
        )

    # Load integrated posterior
    loaded = load_posterior_and_metadata(posterior_path)
    posterior_ds = loaded["posterior"]
    cond_indices = loaded["cond_indices"]
    conditions = loaded["conditions"]

    # Extract subject-level params per condition (already natural scale)
    subj_by_cond = extract_subject_params(posterior_ds, cond_indices)

    # Extract group-level params (latent scale)
    group_params = extract_group_params(posterior_ds)

    # Determine number of samples (flattened chains)
    first_group_key = next(iter(group_params.keys()))
    n_samples = len(group_params[first_group_key])

    # Compute summaries
    subject_summary = compute_learning_rate_subject_summary(subj_by_cond)
    population_summary = compute_condition_population_summary(
        group_params, n_samples
    )
    effects = compute_marginal_effects(group_params, n_samples)

    _write_figure3_source(subject_summary, population_summary, effects, tables_dir)

    # The same group parameters on the logit scale, where the model and its
    # priors are defined. This is the table to cite for the volatility effects.
    _write_source_table(
        compute_logit_coefficient_summary(group_params),
        tables_dir / "figure-3-coefficients.csv",
        "Figure 3 logit coefficients",
    )

    plot_figure3(
        subject_summary,
        population_summary,
        effects,
        str(figures_dir / "figure-3-learning-rates"),
    )
    print("  [figure] figure-3-learning-rates.{pdf,svg,png}\n")

    print("[bold green]✓ Publication figures complete![/bold green]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main(parse_args())
