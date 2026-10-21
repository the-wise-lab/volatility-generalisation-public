"""
Results Compilation and Analysis Script

Aggregates results from parallel MCMC fitting jobs, performs cross-condition
comparisons, and generates summary figures.

This script runs after the per-condition model fits (``fit_single.py``, run
as SLURM array jobs) have finished. It reads each fit's saved results and
writes a summary table, parameter plots, a subject-level parameter table, and
combined posterior predictive checks. For the second-block conditions, it
also runs a two-way ANOVA (current block by previous block) on each model
parameter. The subject-level table is the input to ``regression_analysis.py``.
See ``docs/2-model-fitting.md``.

By default, one run compiles two analyses, each into its own folder: the
single-learning-rate model (``rescorla_wagner_single_value``) into
``results/compiled`` and the dual-learning-rate model
(``dual_lr_single_value``) into ``results/compiled_dual_lr``.

Usage:
    python compile_results.py --overwrite

    # One model only
    python compile_results.py --analysis-model dual_lr_single_value \
                              --output-dir results/compiled_dual_lr
"""

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from scipy import stats
import json
from typing import Any, Dict, List, NamedTuple
from rich.console import Console
from rich.table import Table
from rich.progress import track
import pingouin as pg
from volatility_generalisation.modelling_code.condition_metadata import (
    CONDITION_METADATA,
    SECOND_BLOCK_CONDITIONS,
)
from volatility_generalisation.modelling_code.data_generation import (
    DEFAULT_MODEL_BOUNDS_CONFIG,
    load_parameter_bounds,
    settings_file_record,
)
from volatility_generalisation.modelling_code.model_simulation import MODEL_INFO
from volatility_generalisation.modelling_code.posterior_utils import posterior_interval

console = Console()

# The default analyses, paired by position: each model is compiled into the
# folder at the same place in the second list.
DEFAULT_ANALYSIS_MODELS = ["rescorla_wagner_single_value", "dual_lr_single_value"]
DEFAULT_OUTPUT_DIRS = ["results/compiled", "results/compiled_dual_lr"]
CROSS_PARAMETER_CORRECTION_CHOICES = ("none", "fdr_bh", "bonferroni")
SUBJECT_PARAMETER_ID_COLUMNS = [
    'subject_id',
    'subject_idx',
    'condition',
    'model',
    'block_type',
    'version',
    'block_number',
    'previous_block_type',
    'converged',
]
OMNIBUS_EFFECT_SOURCES = {
    'current_block': 'current_block',
    'previous_block': 'previous_block',
    'current_block * previous_block': 'interaction',
}
CONDITION_ORDER = {condition: index for index, condition in enumerate(CONDITION_METADATA.keys())}
REQUIRED_FIT_FILES = (
    "metadata.json",
    "diagnostics.json",
    "parameter_summaries.json",
    "posterior_samples.npz",
    "ppc_results.npz",
)
# The folder that holds ``config/``, where the parameter bounds file lives.
PROJECT_ROOT = DEFAULT_MODEL_BOUNDS_CONFIG.parent.parent


class FitFolder(NamedTuple):
    """One folder of saved results for one condition and one model."""

    condition: str
    model: str
    path: Path


class CheckedAnalysis(NamedTuple):
    """One analysis model's fits and tables, read and checked, ready to write.

    Attributes:
        analysis_model: Model compiled, or ``"none"`` for every model found.
        output_dir: Folder the analysis is written to.
        fits: The fit folders used, one per condition (all models for
            ``"none"``).
        bounds: The model's parameter bounds, from ``find_bounds_file``, or
            None for ``"none"``.
        analysis_df: Summary table, one row per model and condition, from
            ``aggregate_results``.
        subject_df: Subject-level parameter table, one row per participant,
            condition and model, from ``build_subject_parameter_table``.
    """

    analysis_model: str
    output_dir: Path
    fits: List[FitFolder]
    bounds: Dict[str, Dict[str, float]] | None
    analysis_df: pd.DataFrame
    subject_df: pd.DataFrame


def load_single_result(result_dir: Path) -> Dict:
    """
    Load results from a single model-condition combination.
    
    Args:
        result_dir: Path to result directory (e.g., VolatileV1_rescorla_wagner_single_value/)
        
    Returns:
        Dictionary with all results
    """
    # Load metadata
    with open(result_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load diagnostics
    with open(result_dir / "diagnostics.json", 'r') as f:
        diagnostics = json.load(f)
    
    # Load parameter summaries
    with open(result_dir / "parameter_summaries.json", 'r') as f:
        parameter_summaries = json.load(f)
    
    return {
        'condition': metadata['condition'],
        'model': metadata['model'],
        'converged': diagnostics['converged'],
        'accuracy': metadata['accuracy'],
        'n_subjects': metadata['n_subjects'],
        'n_trials': metadata['n_trials'],
        'parameters': parameter_summaries,
        'diagnostics': diagnostics,
    }


def _check_fit_folder(fit: FitFolder) -> None:
    """Check that a fit folder is complete and that every file can be read.

    JSON files must parse, and every array in the ``.npz`` files must load, so
    a truncated file is caught here.

    Args:
        fit: The fit folder to check.

    Raises:
        FileNotFoundError: If one of the five result files is missing.
        ValueError: If a file cannot be read, or if the condition or model in
            ``metadata.json`` does not match the folder name.
    """
    metadata = None
    for file_name in REQUIRED_FIT_FILES:
        file_path = fit.path / file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"Fit folder {fit.path} is missing {file_name}.")
        try:
            if file_name.endswith(".json"):
                with open(file_path) as f:
                    content = json.load(f)
                if file_name == "metadata.json":
                    metadata = content
            else:
                with np.load(file_path, allow_pickle=True) as arrays:
                    for array_name in arrays.files:
                        arrays[array_name]
        except Exception as exc:
            raise ValueError(
                f"Cannot read {file_name} in fit folder {fit.path}: {exc}"
            ) from exc

    if metadata.get("condition") != fit.condition or metadata.get("model") != fit.model:
        raise ValueError(
            f"metadata.json in fit folder {fit.path} is for condition '{metadata.get('condition')}' "
            f"and model '{metadata.get('model')}', which does not match the folder name."
        )


def find_fit_folders(results_dir: Path, conditions: List[str]) -> List[FitFolder]:
    """Find and check the fit folders for the chosen conditions.

    A subfolder of ``results_dir`` is a fit folder if its name is
    ``<condition>_<model>``, with a condition from the condition metadata and
    a model from the model list. Longer condition names are tried first, so
    ``StableV4_2`` is not mistaken for ``StableV4``. Other subfolders, such as
    ``compiled``, are ignored, as are fits for conditions that are not listed.
    Fits that did not converge are included. The folders are found first and
    then checked one by one with a progress bar, because checking loads every
    saved array and can take minutes.

    Args:
        results_dir: Folder with one result subfolder per condition and model.
        conditions: Conditions whose fits are wanted.

    Returns:
        The fit folders, sorted by folder name.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist, or a fit folder
            is missing one of its five files.
        ValueError: If a fit folder has a file that cannot be read, or if no
            fit folders are found.
    """
    fits = []
    for path in sorted(results_dir.iterdir()):
        if not path.is_dir():
            continue
        for condition in sorted(CONDITION_METADATA, key=len, reverse=True):
            model = path.name[len(condition) + 1 :]
            if path.name.startswith(condition + "_") and model in MODEL_INFO:
                if condition in conditions:
                    fits.append(FitFolder(condition, model, path))
                break
    if not fits:
        raise ValueError(
            f"No fit folders found in {results_dir} for conditions: {', '.join(conditions)}."
        )

    # Check every fit folder, including those of models that are not being
    # compiled, so that an incomplete run is caught before anything is
    # written. Each check reads every array of both .npz files to catch a
    # file cut short by a job that stopped mid-write, which is the slow part.
    console.print(
        f"\n[bold cyan]Checking {len(fits)} fit folders in {results_dir}...[/bold cyan]"
    )
    for fit in track(fits, description="Checking fit folders..."):
        _check_fit_folder(fit)
    console.print(
        f"[green]✓ All {len(fits)} fit folders are complete and readable[/green]"
    )
    return fits


def aggregate_results(fits: List[FitFolder]) -> pd.DataFrame:
    """
    Load and aggregate all results into a DataFrame.
    
    Args:
        fits: Fit folders from ``find_fit_folders``.
        
    Returns:
        DataFrame with one row per model-condition combination.

    Raises:
        ValueError: If the results in a fit folder cannot be loaded.
    """
    console.print("\n[bold cyan]Loading results...[/bold cyan]")
    
    all_results = []
    
    for fit in track(fits, description="Loading results..."):
        try:
            result = load_single_result(fit.path)
            
            # Flatten parameter summaries into columns
            row = {
                'condition': result['condition'],
                'model': result['model'],
                'converged': result['converged'],
                'accuracy': result['accuracy'],
                'n_subjects': result['n_subjects'],
                'n_trials': result['n_trials'],
                'divergences': result['diagnostics']['divergences'],
            }
            
            # Add parameter means
            for param_name, summary in result['parameters'].items():
                row[f'{param_name}_mean'] = summary['mean']
                row[f'{param_name}_std'] = summary['std']
                row[f'{param_name}_median'] = summary['median']
            
            all_results.append(row)
            
        except Exception as exc:
            raise ValueError(f"Failed to load results from {fit.path}: {exc}") from exc
    
    df = pd.DataFrame(all_results)
    
    console.print(f"[green]✓ Loaded {len(df)} results[/green]")
    
    return df


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Print and save a summary table of all model fits.

    Args:
        df: Table with one row per model and condition, as returned by
            ``aggregate_results``.
        output_dir: Folder in which to save ``summary_table.csv``.
    """
    console.print("\n[bold cyan]Creating summary table...[/bold cyan]")
    
    # Rich table for display
    table = Table(title="Model Fitting Summary", show_header=True, header_style="bold magenta")
    table.add_column("Condition", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Converged", justify="center")
    table.add_column("Accuracy", justify="right")
    table.add_column("N Subjects", justify="right")
    
    for _, row in df.iterrows():
        converged_str = "[green]✓[/green]" if row['converged'] else "[red]✗[/red]"
        table.add_row(
            row['condition'],
            row['model'],
            converged_str,
            f"{row['accuracy']:.3f}",
            str(row['n_subjects'])
        )
    
    console.print(table)
    
    # Save to CSV
    csv_path = output_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]✓ Saved to {csv_path}[/green]")


def plot_parameter_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Plot parameter distributions across conditions.
    
    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    console.print("\n[bold cyan]Generating parameter distribution plots...[/bold cyan]")
    
    # Find all parameter columns (those ending in _mean)
    param_cols = [col for col in df.columns if col.endswith('_mean')]
    param_names = [col.replace('_mean', '') for col in param_cols]
    
    for param_name, param_col in zip(param_names, param_cols):
        # Get data for this parameter (filter out NaNs)
        param_data = df[df[param_col].notna()][['condition', 'model', param_col]]
        
        if param_data.empty:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Group by condition and plot
        conditions = sorted(param_data['condition'].unique())
        x_pos = np.arange(len(conditions))
        width = 0.25
        
        models = sorted(param_data['model'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_data = param_data[param_data['model'] == model]
            values = [
                model_data[model_data['condition'] == cond][param_col].values[0]
                if cond in model_data['condition'].values else np.nan
                for cond in conditions
            ]
            
            ax.bar(x_pos + i * width, values, width, label=model, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel(param_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{param_name.replace("_", " ").title()} Across Conditions', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / f"param_dist_{param_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"  ✓ Saved {param_name} plot")


def _logit(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Apply the logit transform to values between 0 and 1.

    Values are first clipped away from 0 and 1 so that the transform stays
    finite.

    Args:
        x: Values between 0 and 1, such as learning rates.
        eps: Distance from 0 and 1 at which values are clipped.

    Returns:
        Array of logit-transformed values, with the same shape as ``x``.
    """
    xc = np.clip(x, eps, 1.0 - eps)
    return np.log(xc / (1.0 - xc))


def _format_parameter_label(parameter_name: str) -> str:
    """Turn a parameter name into a readable label.

    Args:
        parameter_name: Parameter name, such as ``learning_rate``.

    Returns:
        The name with underscores replaced by spaces and each word
        capitalised, for example ``Learning Rate``.
    """
    return parameter_name.replace('_', ' ').title()


def _get_subject_parameter_columns(df: pd.DataFrame) -> List[str]:
    """List the model-parameter columns in the subject-level table.

    Args:
        df: Subject-level table with identifier columns and one numeric column
            per model parameter.

    Returns:
        Names of the numeric columns that are not identifier columns.
    """
    return [
        column for column in df.columns
        if column not in SUBJECT_PARAMETER_ID_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def find_bounds_file(
    fits: List[FitFolder], project_root: Path = PROJECT_ROOT
) -> Dict[str, Dict[str, float]]:
    """Find the parameter bounds that the fits of one model used.

    Each fit's ``metadata.json`` records the path of its bounds file and, for
    fits made since the hash was added, the SHA-256 hash of the file. Fits with
    a hash are matched by the hash: the recorded path is tried first, then
    ``config/model_bounds.toml`` in this project, and the first file with the
    recorded hash is used. Recorded paths may point to another machine. Fits
    without a hash use their recorded path as it is. All fits must record the
    same hash (or, with no hashes, the same path), so a mix of fits with and
    without a hash stops the compile.

    Args:
        fits: Fit folders of the analysis model, all for the same model.
        project_root: Folder that holds ``config/``. Relative recorded paths
            are taken from here.

    Returns:
        The model's parameter bounds from the matched file, as returned by
        ``load_parameter_bounds``.

    Raises:
        ValueError: If a fit has no recorded path, if the fits did not all
            record the same hash (or, without hashes, the same path), if no
            file matches the recorded hash, or if the file has no bounds for
            the model.
        FileNotFoundError: If a fit without a hash recorded a path that does
            not exist.
        KeyError: If the file has no ``common`` section.
    """
    model = fits[0].model
    recorded_paths = []
    recorded_hashes = []
    for fit in fits:
        with open(fit.path / "metadata.json") as f:
            metadata = json.load(f)
        if not metadata.get("config_path"):
            raise ValueError(
                f"Fit folder {fit.path} does not record which parameter bounds file "
                "it used (no config_path in metadata.json)."
            )
        recorded_paths.append(metadata["config_path"])
        recorded_hashes.append(metadata.get("config_sha256") or None)

    if len(set(recorded_hashes)) > 1:
        raise ValueError(
            f"The fits of model '{model}' do not all record the same parameter bounds file "
            "hash, so they do not identify one file. Recorded hashes: "
            f"{', '.join(sorted(str(recorded_hash) for recorded_hash in set(recorded_hashes)))} "
            "(None means no hash was recorded)."
        )

    # A recorded path that is already absolute is kept as it is by the "/" join.
    candidates = [project_root / path for path in dict.fromkeys(recorded_paths)]
    if recorded_hashes[0]:
        candidates.append(project_root / "config" / "model_bounds.toml")
        bounds_path = next(
            (
                path
                for path in candidates
                if path.is_file()
                and settings_file_record(path)["config_sha256"] == recorded_hashes[0]
            ),
            None,
        )
        if bounds_path is None:
            raise ValueError(
                f"No parameter bounds file matches the one used by fit folder {fits[0].path} "
                f"(recorded path {recorded_paths[0]}, SHA-256 {recorded_hashes[0]}). The settings "
                "file changed after the fit or is not available."
            )
    else:
        if len(candidates) > 1:
            raise ValueError(
                f"The fits of model '{model}' recorded different parameter bounds files: "
                f"{', '.join(sorted(set(recorded_paths)))}."
            )
        bounds_path = candidates[0]
        if not bounds_path.is_file():
            raise FileNotFoundError(
                f"Fit folder {fits[0].path} recorded the parameter bounds file {recorded_paths[0]}, "
                "which does not exist, and it recorded no hash to find another copy."
            )

    try:
        return load_parameter_bounds(model, bounds_path)
    except ValueError as exc:
        raise ValueError(
            f"The parameter bounds file {bounds_path} has no usable bounds for model '{model}': {exc}"
        ) from exc


def _get_parameter_transform(
    parameter_name: str, bounds: Dict[str, Dict[str, float]]
) -> str:
    """Choose how to transform a parameter before the ANOVA.

    Parameters bounded within 0 to 1 in the bounds file, such as learning
    rates, are logit-transformed. Parameters with no minimum and maximum, such
    as fixed values, and all other parameters, are left unchanged.

    Args:
        parameter_name: Name of the parameter.
        bounds: The model's parameter bounds, from ``find_bounds_file``.

    Returns:
        ``"logit"`` or ``"identity"``.
    """
    parameter_bounds = bounds.get(parameter_name, {})

    min_value = parameter_bounds.get('min')
    max_value = parameter_bounds.get('max')
    if min_value is None or max_value is None:
        return 'identity'
    if float(min_value) >= 0.0 and float(max_value) <= 1.0:
        return 'logit'
    return 'identity'


def _apply_parameter_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    """Transform the values of one parameter.

    Args:
        values: Parameter values, one per participant.
        transform_name: ``"logit"`` to apply the logit transform. Any other
            name returns the values unchanged.

    Returns:
        Array of transformed values, as floats. It is always a new array.
    """
    if transform_name == 'logit':
        return _logit(values)
    return values.astype(float, copy=True)


def _apply_cross_parameter_correction(
    parameter_overview: pd.DataFrame,
    correction_method: str,
) -> pd.DataFrame:
    """Correct the ANOVA p-values for testing several parameters.

    For each effect (current block, previous block, and their interaction),
    the p-values of all parameters are corrected together. Parameters without
    a p-value are left out of the correction.

    Args:
        parameter_overview: Table with one row per parameter and raw p-value
            columns such as ``current_block_p_raw``.
        correction_method: ``"none"`` to keep the raw p-values,
            ``"bonferroni"``, or ``"fdr_bh"`` (Benjamini-Hochberg false
            discovery rate).

    Returns:
        A copy of the table with the correction method and a corrected
        p-value column added for each effect.

    Raises:
        ValueError: If the correction method is not recognised.
    """
    corrected = parameter_overview.copy()
    corrected['omnibus_correction_method'] = correction_method

    for effect_key in OMNIBUS_EFFECT_SOURCES.values():
        raw_column = f'{effect_key}_p_raw'
        corrected_column = f'{effect_key}_p_corrected'
        if raw_column not in corrected.columns:
            continue

        corrected[corrected_column] = corrected[raw_column]
        valid_mask = corrected[raw_column].notna()
        if correction_method == 'none' or valid_mask.sum() == 0:
            continue

        p_values = corrected.loc[valid_mask, raw_column].to_numpy(dtype=float)
        if correction_method == 'bonferroni':
            corrected_values = np.minimum(p_values * len(p_values), 1.0)
        elif correction_method == 'fdr_bh':
            _, corrected_values = pg.multicomp(p_values, method='fdr_bh')
        else:
            raise ValueError(f"Unknown correction method: {correction_method}")

        corrected.loc[valid_mask, corrected_column] = corrected_values

    return corrected


def _existing_output_entries(output_dir: Path) -> List[str]:
    """List the files and folders already in the output folder.

    Args:
        output_dir: Folder for the compiled results.

    Returns:
        Sorted names of the entries in the folder. The list is empty if the
        folder does not exist.
    """
    if not output_dir.exists():
        return []
    return sorted(path.name for path in output_dir.iterdir())


def _condition_sort_key(condition: str) -> tuple[int, str]:
    """Give a condition its place in the standard condition order.

    Args:
        condition: Condition name.

    Returns:
        Tuple to sort by. Known conditions come first, in the order of the
        condition metadata. Unknown conditions come last, sorted by name.
    """
    return CONDITION_ORDER.get(condition, len(CONDITION_ORDER)), condition


def run_parameter_analyses(
    subject_df: pd.DataFrame,
    output_dir: Path,
    analysis_model: str,
    bounds: Dict[str, Dict[str, float]],
    cross_parameter_correction: str = 'none',
):
    """Run a two-way ANOVA on each parameter of one model, second block only.

    Uses the second-block conditions, where each participant has a current
    block (Stable or Volatile) and a previous block (Stable or Volatile). For
    each parameter, this saves cell means, normality checks (Shapiro-Wilk),
    an equal-variance check (Levene), the ANOVA table, two follow-up
    comparisons of the previous-block effect, and an interaction plot. Levene's
    test decides whether the follow-up t-tests are Welch or Student tests.
    Finally, it saves an overview table across parameters.

    The analysis stops with an error, rather than dropping a participant, if a
    parameter has a missing or non-finite value for any participant. It also
    stops, rather than skipping a parameter, if any of the four block
    combinations is missing or has fewer than three participants (three is the
    least the Shapiro-Wilk test can use). Finally, it stops if a Bayes factor
    cannot be computed or is not a finite number, or if the ANOVA table lacks
    an effect.

    Args:
        subject_df: Subject-level parameter table from
            ``build_subject_parameter_table``. It must hold only the fits of
            the analysis model, so that every parameter column belongs to it.
        output_dir: Folder in which to save the ``parameters`` folder and
            ``parameter_overview.csv``.
        analysis_model: Model to analyse.
        bounds: The model's parameter bounds, from ``find_bounds_file``. They
            decide which parameters are logit-transformed.
        cross_parameter_correction: Correction for testing several
            parameters: ``"none"``, ``"fdr_bh"`` or ``"bonferroni"``.

    Returns:
        Table with one row per parameter.

    Raises:
        ValueError: If there is no subject-level data, if the table holds other
            models, or if there are no second-block rows or no parameters for
            the model; if a parameter has a missing or
            non-finite value for any participant; if a parameter has a missing
            or small block combination; or if the ANOVA table lacks an effect, its
            p-value column, or the ``F`` or ``np2`` column.
        RuntimeError: If a Bayes factor fails or is not finite.
    """
    console.print("\n[bold cyan]Running parameter-wise second-block analyses...[/bold cyan]")

    if subject_df.empty:
        raise ValueError(
            f"No subject-level data for the parameter analyses of model '{analysis_model}'."
        )

    other_models = sorted(set(subject_df["model"]) - {analysis_model})
    if other_models:
        raise ValueError(
            f"The subject-level table for the analyses of '{analysis_model}' also "
            f"holds other models: {', '.join(other_models)}."
        )
    model_df = subject_df.copy()
    model_df['block_number'] = pd.to_numeric(model_df['block_number'], errors='coerce')
    model_df = model_df[
        (model_df['block_number'] == 2)
        & model_df['block_type'].isin(['Stable', 'Volatile'])
        & model_df['previous_block_type'].isin(['Stable', 'Volatile'])
    ].copy()
    if model_df.empty:
        raise ValueError(
            f"No second-block subject rows found for model '{analysis_model}'."
        )

    parameter_columns = _get_subject_parameter_columns(model_df)
    if not parameter_columns:
        raise ValueError(f"No parameter columns found for model '{analysis_model}'.")
    for column in parameter_columns:
        values = model_df[column].to_numpy(dtype=float)
        missing = model_df.loc[~np.isfinite(values), ["subject_id", "condition"]]
        if not missing.empty:
            raise ValueError(
                f"Parameter '{column}' of model '{analysis_model}' has no finite value "
                f"for {len(missing)} participant(s), for example "
                f"{missing.iloc[0]['subject_id']} in {missing.iloc[0]['condition']}."
            )

    parameter_root = output_dir / 'parameters'
    parameter_root.mkdir(parents=True, exist_ok=True)
    overview_rows = []
    expected_cells = {
        ('Stable', 'Stable'),
        ('Stable', 'Volatile'),
        ('Volatile', 'Stable'),
        ('Volatile', 'Volatile'),
    }

    for index, parameter_name in enumerate(parameter_columns, start=1):
        console.print(f"\n[bold]Parameter {index}/{len(parameter_columns)}:[/bold] {parameter_name}")
        parameter_dir = parameter_root / parameter_name
        parameter_dir.mkdir(parents=True, exist_ok=True)

        parameter_df = model_df[
            ['subject_id', 'subject_idx', 'condition', 'block_type', 'previous_block_type', parameter_name]
        ].copy()
        parameter_df = parameter_df.rename(
            columns={
                'block_type': 'current_block',
                'previous_block_type': 'previous_block',
                parameter_name: 'parameter_value',
            }
        )

        overview_row = {
            'parameter': parameter_name,
            'parameter_label': _format_parameter_label(parameter_name),
            'model': analysis_model,
            "transform": _get_parameter_transform(parameter_name, bounds),
            'n_subjects': len(parameter_df),
            'n_conditions': parameter_df['condition'].nunique(),
            'status': 'ok',
        }

        cell_summary = (
            parameter_df
            .groupby(['condition', 'previous_block', 'current_block'])['parameter_value']
            .agg(n='size', mean='mean', std='std')
            .reset_index()
            .sort_values(['previous_block', 'current_block', 'condition'])
        )
        cell_summary.to_csv(parameter_dir / 'cell_summary.csv', index=False)

        cell_counts = parameter_df.groupby(['current_block', 'previous_block']).size()
        observed_cells = set(cell_counts.index.tolist())
        if observed_cells != expected_cells or (cell_counts < 3).any():
            missing_cells = sorted(expected_cells - observed_cells)
            too_small = [
                f"{current}/{previous}"
                for (current, previous), count in cell_counts.items()
                if count < 3
            ]
            problems = []
            if missing_cells:
                problems.append(
                    "missing cells: " + ", ".join(f"{current}/{previous}" for current, previous in missing_cells)
                )
            if too_small:
                problems.append(
                    "cells (current/previous block) with fewer than 3 participants: "
                    + ", ".join(too_small)
                )
            raise ValueError(
                f"Cannot analyse parameter '{parameter_name}' of model '{analysis_model}': "
                + "; ".join(problems)
            )

        parameter_df['analysis_value'] = _apply_parameter_transform(
            parameter_df['parameter_value'].to_numpy(dtype=float),
            overview_row['transform'],
        )
        console.print(
            f"  Transform: {overview_row['transform']} | "
            f"N = {len(parameter_df)} subjects across {parameter_df['condition'].nunique()} conditions"
        )

        shapiro_rows = []
        for (current_block, previous_block), group in parameter_df.groupby(['current_block', 'previous_block']):
            values = group['analysis_value'].to_numpy(dtype=float)
            statistic, p_value = stats.shapiro(values)
            shapiro_rows.append({
                'current_block': current_block,
                'previous_block': previous_block,
                'transition': f'{previous_block}->{current_block}',
                'n': len(values),
                'W': statistic,
                'p': p_value,
                'normal_p_gt_0_05': p_value > 0.05,
            })
        shapiro_df = pd.DataFrame(shapiro_rows)
        shapiro_df.to_csv(parameter_dir / 'shapiro_checks.csv', index=False)
        overview_row["shapiro_min_p"] = float(shapiro_df["p"].min())

        analysis_groups = [
            group['analysis_value'].to_numpy(dtype=float)
            for _, group in parameter_df.groupby(['current_block', 'previous_block'])
        ]
        levene_stat, levene_p = stats.levene(*analysis_groups)
        use_welch = levene_p < 0.05
        pd.DataFrame([
            {
                'statistic': levene_stat,
                'p': levene_p,
                'test_type': 'Welch' if use_welch else 'Student',
            }
        ]).to_csv(parameter_dir / 'levene_check.csv', index=False)
        overview_row['levene_stat'] = float(levene_stat)
        overview_row['levene_p'] = float(levene_p)

        aov = pg.anova(
            dv='analysis_value',
            between=['current_block', 'previous_block'],
            data=parameter_df,
            detailed=True,
        )
        aov.to_csv(parameter_dir / 'anova_results.csv', index=False)
        p_col = 'p_unc' if 'p_unc' in aov.columns else 'p-unc' if 'p-unc' in aov.columns else None
        missing_columns = [
            column for column in ("F", "np2") if column not in aov.columns
        ]
        if p_col is None:
            missing_columns.append("p-value")
        if missing_columns:
            raise ValueError(
                f"The ANOVA table for parameter '{parameter_name}' has no column for: "
                f"{', '.join(missing_columns)}."
            )
        display_cols = [c for c in ['Source', 'SS', 'DF', 'F', 'np2'] if c in aov.columns]
        if p_col is not None:
            display_cols.insert(4, p_col)
        display_aov = aov[display_cols].copy()
        for column in ['F', 'np2']:
            if column in display_aov.columns:
                display_aov[column] = display_aov[column].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )
        if p_col is not None:
            display_aov[p_col] = display_aov[p_col].apply(
                lambda p: "< .001" if pd.notna(p) and p < 0.001 else f"{p:.3f}" if pd.notna(p) else ""
            )
        console.print(display_aov.to_string(index=False))

        for source_name, effect_key in OMNIBUS_EFFECT_SOURCES.items():
            effect_match = aov[aov['Source'] == source_name]
            if effect_match.empty:
                raise ValueError(
                    f"The ANOVA table for parameter '{parameter_name}' has no row for the "
                    f"effect '{source_name}'."
                )
            effect_row = effect_match.iloc[0]
            overview_row[f"{effect_key}_F"] = float(effect_row["F"])
            overview_row[f"{effect_key}_p_raw"] = float(effect_row[p_col])
            overview_row[f"{effect_key}_np2"] = float(effect_row["np2"])

        condition_lookup = (
            parameter_df[['condition', 'current_block', 'previous_block']]
            .drop_duplicates()
            .set_index(['current_block', 'previous_block'])['condition']
            .to_dict()
        )
        comparisons = [
            {
                'label': 'Previous Block effect | Current = Stable',
                'a': parameter_df[
                    (parameter_df['current_block'] == 'Stable')
                    & (parameter_df['previous_block'] == 'Volatile')
                ]['analysis_value'].to_numpy(dtype=float),
                'b': parameter_df[
                    (parameter_df['current_block'] == 'Stable')
                    & (parameter_df['previous_block'] == 'Stable')
                ]['analysis_value'].to_numpy(dtype=float),
                'label_a': f"Volatile->Stable ({condition_lookup[('Stable', 'Volatile')]})",
                'label_b': f"Stable->Stable ({condition_lookup[('Stable', 'Stable')]})",
            },
            {
                'label': 'Previous Block effect | Current = Volatile',
                'a': parameter_df[
                    (parameter_df['current_block'] == 'Volatile')
                    & (parameter_df['previous_block'] == 'Stable')
                ]['analysis_value'].to_numpy(dtype=float),
                'b': parameter_df[
                    (parameter_df['current_block'] == 'Volatile')
                    & (parameter_df['previous_block'] == 'Volatile')
                ]['analysis_value'].to_numpy(dtype=float),
                'label_a': f"Stable->Volatile ({condition_lookup[('Volatile', 'Stable')]})",
                'label_b': f"Volatile->Volatile ({condition_lookup[('Volatile', 'Volatile')]})",
            },
        ]

        posthoc_rows = []
        for comparison in comparisons:
            group_a = comparison['a']
            group_b = comparison['b']
            t_stat, p_raw = stats.ttest_ind(group_a, group_b, equal_var=not use_welch)
            p_bonf = min(float(p_raw) * 2, 1.0)
            pooled_sd = np.sqrt(
                ((len(group_a) - 1) * group_a.var(ddof=1) + (len(group_b) - 1) * group_b.var(ddof=1))
                / (len(group_a) + len(group_b) - 2)
            )
            cohens_d = (group_a.mean() - group_b.mean()) / pooled_sd if pooled_sd > 0 else 0.0
            try:
                bf10 = float(pg.bayesfactor_ttest(t_stat, len(group_a), len(group_b)))
            except Exception as exc:
                raise RuntimeError(
                    f"The Bayes factor failed for parameter '{parameter_name}', "
                    f"comparison '{comparison['label']}': {exc}"
                ) from exc
            if not math.isfinite(bf10):
                raise RuntimeError(
                    f"The Bayes factor for parameter '{parameter_name}', comparison "
                    f"'{comparison['label']}' is not a finite number: {bf10}"
                )

            console.print(
                f"  {comparison['label']}: "
                f"{comparison['label_a']} M = {group_a.mean():.3f}, SD = {group_a.std(ddof=1):.3f}; "
                f"{comparison['label_b']} M = {group_b.mean():.3f}, SD = {group_b.std(ddof=1):.3f}; "
                f"p = {p_raw:.4f}, p_bonf = {p_bonf:.4f}, d = {cohens_d:.3f}"
            )
            posthoc_rows.append({
                'comparison': comparison['label'],
                'group_a': comparison['label_a'],
                'group_b': comparison['label_b'],
                'n_a': len(group_a),
                'n_b': len(group_b),
                'mean_a_analysis': group_a.mean(),
                'mean_b_analysis': group_b.mean(),
                'std_a_analysis': group_a.std(ddof=1),
                'std_b_analysis': group_b.std(ddof=1),
                't': t_stat,
                'p_raw': p_raw,
                'p_bonferroni': p_bonf,
                'cohens_d': cohens_d,
                'BF10': bf10,
                'test_type': 'Welch' if use_welch else 'Student',
            })
        pd.DataFrame(posthoc_rows).to_csv(parameter_dir / 'posthoc_comparisons.csv', index=False)

        palette = {'Stable': '#4ECDC4', 'Volatile': '#FF6B6B'}
        current_order = ['Stable', 'Volatile']
        rng = np.random.default_rng(42)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        for previous_block, linestyle in [('Stable', '-'), ('Volatile', '--')]:
            means = []
            sems = []
            for current_block in current_order:
                values = parameter_df[
                    (parameter_df['current_block'] == current_block)
                    & (parameter_df['previous_block'] == previous_block)
                ]['parameter_value'].to_numpy(dtype=float)
                means.append(values.mean())
                sems.append(values.std(ddof=1) / np.sqrt(len(values)))
                ax.scatter(
                    rng.normal(current_order.index(current_block), 0.05, len(values)),
                    values,
                    alpha=0.25,
                    s=18,
                    color=palette[previous_block],
                    zorder=2,
                )
            ax.errorbar(
                [0, 1],
                means,
                yerr=sems,
                marker='o',
                markersize=8,
                linewidth=2,
                linestyle=linestyle,
                color=palette[previous_block],
                capsize=6,
                label=f'Previous: {previous_block}',
                zorder=3,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(current_order, fontsize=11)
        ax.set_xlabel('Current Block Volatility', fontsize=12)
        ax.set_ylabel(_format_parameter_label(parameter_name), fontsize=12)
        ax.set_title(
            (
                f"{_format_parameter_label(parameter_name)} by Block Context\n"
                f"(mean +/- SEM; statistics on {overview_row['transform']} scale)"
            ),
            fontsize=11,
        )
        ax.legend(title='Previous Block', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        ax = axes[1]
        parameter_df['transition'] = parameter_df['previous_block'] + '->' + parameter_df['current_block']
        transition_order = ['Stable->Stable', 'Volatile->Stable', 'Stable->Volatile', 'Volatile->Volatile']
        transition_colors = {transition: palette[transition.split('->')[1]] for transition in transition_order}
        sns.stripplot(
            data=parameter_df,
            x='transition',
            y='parameter_value',
            order=transition_order,
            hue='transition',
            palette=transition_colors,
            alpha=0.45,
            size=5,
            jitter=True,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sns.pointplot(
            data=parameter_df,
            x='transition',
            y='parameter_value',
            order=transition_order,
            color='black',
            markers='D',
            linestyles='none',
            errorbar='se',
            capsize=0.12,
            ax=ax,
        )
        ax.set_xlabel('Transition (Previous -> Current)', fontsize=11)
        ax.set_ylabel(_format_parameter_label(parameter_name), fontsize=12)
        ax.set_title('Per-condition parameter values\n(diamonds = mean +/- SEM)', fontsize=11)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        fig.savefig(parameter_dir / 'interaction_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        console.print(f"  Saved outputs to {parameter_dir}")

        overview_rows.append(overview_row)

    parameter_overview = pd.DataFrame(overview_rows)
    parameter_overview = _apply_cross_parameter_correction(
        parameter_overview,
        correction_method=cross_parameter_correction,
    )
    parameter_overview = parameter_overview.sort_values('parameter').reset_index(drop=True)
    overview_path = output_dir / 'parameter_overview.csv'
    parameter_overview.to_csv(overview_path, index=False)

    display_columns = [
        column for column in [
            'parameter',
            'transform',
            'status',
            'current_block_p_raw',
            'current_block_p_corrected',
            'previous_block_p_raw',
            'previous_block_p_corrected',
            'interaction_p_raw',
            'interaction_p_corrected',
        ]
        if column in parameter_overview.columns
    ]
    console.print("\n[bold]Parameter overview:[/bold]")
    console.print(parameter_overview[display_columns].to_string(index=False))
    console.print(f"[green]✓ Saved parameter overview to {overview_path}[/green]")

    return parameter_overview


def _load_prepared_subject_ids(npz_path: Path, fit: FitFolder) -> np.ndarray:
    """Read the participant IDs of one condition from its prepared data file.

    Args:
        npz_path: The prepared ``.npz`` file for the fit's condition.
        fit: The fit folder that needs the IDs, named in error messages.

    Returns:
        The participant IDs, in the order used in the fit.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read or has no ``subject_ids``.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Prepared data file {npz_path} does not exist. Fit folder {fit.path} "
            "needs it because its metadata.json has no subject_ids."
        )
    try:
        with np.load(npz_path, allow_pickle=True) as prepared:
            subject_ids = (
                prepared["subject_ids"] if "subject_ids" in prepared.files else None
            )
    except Exception as exc:
        raise ValueError(f"Cannot read prepared data file {npz_path}: {exc}") from exc
    if subject_ids is None:
        raise ValueError(f"Prepared data file {npz_path} has no subject_ids.")
    return subject_ids


def build_subject_parameter_table(
    fits: List[FitFolder],
    data_dir: Path,
) -> pd.DataFrame:
    """
    Build a subject-level table with one row per subject × condition × model.

    Columns:
        subject_id         – participant ID, as in the behavioural data
        subject_idx        – 0-based positional index within the condition dataset
        condition          – e.g. "VolatileV1"
        model              – e.g. "rescorla_wagner_single_value"
        block_type         – "Stable" or "Volatile"
        version            – "V1" … "V4"
        block_number       – 1 (first block in version) or 2 (second block)
        previous_block_type – "Stable", "Volatile", or blank if first block
        converged          – whether this fit converged
        <param>            – posterior mean for each model parameter

    It only reads the fit folders, so that ``main`` can check every
    participant's estimates before it writes any output.
    ``export_subject_parameters`` saves the table.

    Args:
        fits: Fit folders from ``find_fit_folders``.
        data_dir: Folder with the prepared ``.npz`` files, used to look up
            participant IDs when the fit metadata does not have them.

    Returns:
        The subject-level table.

    Raises:
        ValueError: If a fit has no participants, if the number of
            participants does not match the number of means for a parameter or
            the number of IDs, if a participant has no ID, if a participant's
            mean is missing or not finite, or if there are no rows at all.
        FileNotFoundError: If a needed prepared ``.npz`` file does not exist.
    """
    console.print("\n[bold cyan]Reading subject-level parameters...[/bold cyan]")

    rows = []

    for fit in track(fits, description="Reading parameter summaries..."):
        condition, model = fit.condition, fit.model

        with open(fit.path / "parameter_summaries.json") as f:
            param_summaries = json.load(f)
        with open(fit.path / "metadata.json") as f:
            metadata = json.load(f)

        converged = metadata["converged"]
        cond_meta = CONDITION_METADATA[condition]

        n_subjects = metadata["n_subjects"]
        if n_subjects == 0:
            raise ValueError(
                f"Fit folder {fit.path} has 0 participants (n_subjects in metadata.json)."
            )
        for param_name, summary in param_summaries.items():
            if len(summary["subject_means"]) != n_subjects:
                raise ValueError(
                    f"Fit folder {fit.path}: parameter '{param_name}' has {len(summary['subject_means'])} "
                    f"participant means, but metadata.json says there are {n_subjects} participants."
                )

        # Prefer subject IDs stored in metadata (saved by fit_single.py), fall back to the prepared .npz
        meta_subject_ids = metadata.get('subject_ids')
        if meta_subject_ids is not None:
            subject_ids_arr = np.array(meta_subject_ids)
        else:
            subject_ids_arr = _load_prepared_subject_ids(
                data_dir / f"{condition}.npz", fit
            )
        if len(subject_ids_arr) != n_subjects:
            raise ValueError(
                f"Fit folder {fit.path}: there are {len(subject_ids_arr)} participant IDs, "
                f"but metadata.json says there are {n_subjects} participants."
            )
        if any(subject_id is None for subject_id in subject_ids_arr):
            raise ValueError(
                f"Fit folder {fit.path}: a participant has no ID (subject_id)."
            )
        for param_name, summary in param_summaries.items():
            missing = [
                f"{subject_ids_arr[index]} (subject_idx {index})"
                for index, value in enumerate(summary["subject_means"])
                if not isinstance(value, (int, float)) or not math.isfinite(value)
            ]
            if missing:
                raise ValueError(
                    f"Fit folder {fit.path}: parameter '{param_name}' has no finite "
                    f"posterior mean for participant(s) {', '.join(missing)}."
                )

        # Log block number etc
        if cond_meta.get('version', 'N/A') == 'V3':
            console.print(f"Processing condition {condition}, model {model}, {n_subjects} subjects")
            console.print(f"  Block type: {cond_meta.get('block_type', 'N/A')}, "
                        f"Version: {cond_meta.get('version', 'N/A')}, "
                        f"Block number: {cond_meta.get('block_number', 'N/A')}, "
                        f"Previous block: {cond_meta.get('previous_block_type', 'N/A')}, "
                        f"Converged: {converged}")
            # print(cond_meta)


        for subj_idx in range(n_subjects):
            row: Dict = {
                "subject_id": str(subject_ids_arr[subj_idx]),
                'subject_idx':         subj_idx,
                'condition':           condition,
                'model':               model,
                "block_type": cond_meta["block_type"],
                "version": cond_meta["version"],
                "block_number": cond_meta["block_number"],
                "previous_block_type": cond_meta["previous_block_type"] or "",
                'converged':           converged,
            }
            for param_name, summary in param_summaries.items():
                row[param_name] = summary["subject_means"][subj_idx]

            rows.append(row)

    if not rows:
        raise ValueError("No subject-level rows were found in the fit folders.")

    df = pd.DataFrame(rows)

    # Put identifier columns first, then parameter columns
    id_cols = ['subject_id', 'subject_idx', 'condition', 'model', 'block_type', 'version',
               'block_number', 'previous_block_type', 'converged']
    param_cols = [c for c in df.columns if c not in id_cols]
    return df[id_cols + sorted(param_cols)]


def export_subject_parameters(subject_df: pd.DataFrame, output_dir: Path) -> None:
    """Save the subject-level table as ``subject_parameters.csv``.

    Args:
        subject_df: Table from ``build_subject_parameter_table``.
        output_dir: Folder in which to save the table.
    """
    console.print("\n[bold cyan]Exporting subject-level parameters...[/bold cyan]")
    df = subject_df
    csv_path = output_dir / 'subject_parameters.csv'
    df.to_csv(csv_path, index=False)

    console.print(f"[green]✓ Saved {len(df):,} rows ({df['subject_idx'].nunique()} subjects × "
                  f"{df['condition'].nunique()} conditions × {df['model'].nunique()} models) "
                  f"→ {csv_path}[/green]")


def _plot_combined_ppc_trialwise(trialwise_df: pd.DataFrame, output_path: Path, model_name: str) -> None:
    """Plot predicted and observed choice probability by trial, per condition.

    Draws one panel per condition, in two columns. Each panel shows the
    observed proportion of option 0 choices, the model's predicted
    probability, and the 95% interval of the posterior predictive choices.

    Args:
        trialwise_df: Trial-level table from ``export_combined_ppc``.
        output_path: File in which to save the figure.
        model_name: Model name, shown in the figure title.
    """
    conditions = sorted(trialwise_df['condition'].unique().tolist(), key=_condition_sort_key)
    n_cols = min(2, max(1, len(conditions)))
    n_rows = int(math.ceil(len(conditions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.8 * n_rows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()

    legend_handles = None
    legend_labels = None
    for axis, condition in zip(axes_array, conditions):
        condition_df = trialwise_df[trialwise_df['condition'] == condition].sort_values('trial')
        trials = condition_df['trial'].to_numpy()
        axis.fill_between(
            trials,
            condition_df['predicted_choice_prob_ci_lower'].to_numpy(),
            condition_df['predicted_choice_prob_ci_upper'].to_numpy(),
            alpha=0.25,
            color='steelblue',
            label='95% interval',
        )
        axis.plot(
            trials,
            condition_df['predicted_choice_prob'].to_numpy(),
            color='steelblue',
            linewidth=2,
            label='Predicted',
        )
        axis.plot(
            trials,
            condition_df['observed_choice_prob'].to_numpy(),
            color='black',
            linewidth=1.5,
            label='Observed',
        )
        axis.axhline(0.5, color='red', linestyle=':', alpha=0.4)
        axis.set_title(condition)
        axis.set_xlabel('Trial')
        axis.set_ylabel('P(Choose Option 0)')
        axis.set_ylim(0.0, 1.0)
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    for axis in axes_array[len(conditions):]:
        axis.set_visible(False)

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc='upper center', ncol=3, frameon=False)
    fig.suptitle(f'{model_name} posterior predictive check', y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_combined_ppc_scatter(trialwise_df: pd.DataFrame, output_path: Path, model_name: str) -> None:
    """Plot predicted against observed choice probability, one point per trial.

    All conditions share one panel, with a different colour for each
    condition and a dashed line where predicted equals observed.

    Args:
        trialwise_df: Trial-level table from ``export_combined_ppc``.
        output_path: File in which to save the figure.
        model_name: Model name, shown in the figure title.
    """
    plt.figure(figsize=(6.5, 6.5))
    cmap = plt.get_cmap('tab10')
    conditions = sorted(trialwise_df['condition'].unique().tolist(), key=_condition_sort_key)
    for index, condition in enumerate(conditions):
        condition_df = trialwise_df[trialwise_df['condition'] == condition]
        plt.scatter(
            condition_df['observed_choice_prob'],
            condition_df['predicted_choice_prob'],
            s=22,
            alpha=0.75,
            color=cmap(index % 10),
            label=condition,
        )
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1)
    plt.xlabel('Observed P(Choose Option 0)')
    plt.ylabel('Predicted P(Choose Option 0)')
    plt.title(f'{model_name} posterior predictive scatter')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def export_combined_ppc(
    fits: List[FitFolder],
    output_dir: Path,
):
    """Combine the posterior predictive checks of all conditions.

    Reads ``ppc_results.npz`` from each result folder. For each trial, it
    averages observed and predicted choices over participants, and takes the
    2.5th and 97.5th percentiles (a 95% interval) across posterior draws. It
    saves trial-level and per-condition summary tables (mean absolute error,
    root mean squared error, interval width) and two figures. The tables hold
    every model. The figures are drawn for one model each; if there is more
    than one model, the model name is added to the figure file names.

    Args:
        fits: Fit folders from ``find_fit_folders``.
        output_dir: Folder in which to save the tables and figures.

    Returns:
        Tuple of the trial-level table and the per-condition summary table.

    Raises:
        ValueError: If the PPC file of a fit folder is missing or cannot be
            read.
    """
    console.print("\n[bold cyan]Compiling posterior predictive checks...[/bold cyan]")

    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for fit in track(fits, description="Reading PPC outputs..."):
        condition, model = fit.condition, fit.model
        ppc_path = fit.path / "ppc_results.npz"

        try:
            with np.load(ppc_path, allow_pickle=True) as ppc:
                actual_choices = np.asarray(ppc['actual_choices'], dtype=float)
                cond_meta = CONDITION_METADATA[condition]
                n_subjects, n_trials = actual_choices.shape
                observed_by_trial = actual_choices.mean(axis=0)

                if 'all_simulated_probs' in ppc.files:
                    all_simulated_probs = np.asarray(ppc['all_simulated_probs'], dtype=float)
                    predicted_by_draw_trial = all_simulated_probs.mean(axis=1)
                    predicted_by_trial = predicted_by_draw_trial.mean(axis=0)
                else:
                    predicted_by_trial = np.asarray(ppc['simulated_probs'], dtype=float).mean(axis=0)
                    predicted_by_draw_trial = predicted_by_trial[None, :]

                if 'all_simulated_choices' in ppc.files:
                    posterior_predictive_by_draw_trial = np.asarray(
                        ppc['all_simulated_choices'], dtype=float
                    ).mean(axis=1)
                else:
                    posterior_predictive_by_draw_trial = predicted_by_draw_trial

                lower, upper = posterior_interval(
                    posterior_predictive_by_draw_trial, axis=0
                )
        except Exception as exc:
            raise ValueError(
                f"Cannot read ppc_results.npz in fit folder {fit.path}: {exc}"
            ) from exc

        for trial in range(n_trials):
            trial_rows.append(
                {
                    'model': model,
                    'condition': condition,
                    'block_type': cond_meta.get('block_type', ''),
                    'version': cond_meta.get('version', ''),
                    'block_number': cond_meta.get('block_number', ''),
                    'previous_block_type': cond_meta.get('previous_block_type') or '',
                    'trial': trial,
                    'n_subjects': int(n_subjects),
                    'n_draws_used': int(posterior_predictive_by_draw_trial.shape[0]),
                    'observed_choice_prob': float(observed_by_trial[trial]),
                    'predicted_choice_prob': float(predicted_by_trial[trial]),
                    'predicted_choice_prob_ci_lower': float(lower[trial]),
                    'predicted_choice_prob_ci_upper': float(upper[trial]),
                }
            )

        residual = predicted_by_trial - observed_by_trial
        summary_rows.append(
            {
                'model': model,
                'condition': condition,
                'block_type': cond_meta.get('block_type', ''),
                'version': cond_meta.get('version', ''),
                'block_number': cond_meta.get('block_number', ''),
                'previous_block_type': cond_meta.get('previous_block_type') or '',
                'n_subjects': int(n_subjects),
                'n_trials': int(n_trials),
                'n_draws_used': int(posterior_predictive_by_draw_trial.shape[0]),
                'observed_choice_prob_mean': float(observed_by_trial.mean()),
                'predicted_choice_prob_mean': float(predicted_by_trial.mean()),
                'mae': float(np.abs(residual).mean()),
                'rmse': float(np.sqrt(np.mean(residual ** 2))),
                'mean_interval_width': float(np.mean(upper - lower)),
            }
        )

    trialwise_df = pd.DataFrame(trial_rows).sort_values(
        ['model', 'block_number', 'condition', 'trial'],
        kind='stable',
    ).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ['model', 'block_number', 'condition'],
        kind='stable',
    ).reset_index(drop=True)

    trialwise_path = output_dir / 'posterior_predictive_trialwise.csv'
    summary_path = output_dir / 'posterior_predictive_summary.csv'
    trialwise_df.to_csv(trialwise_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    console.print(f"[green]✓ Saved combined PPC table[/green] {trialwise_path}")
    console.print(f"[green]✓ Saved combined PPC table[/green] {summary_path}")

    model_names = sorted(trialwise_df["model"].unique().tolist())
    for model_name in model_names:
        model_df = trialwise_df[trialwise_df["model"] == model_name]
        suffix = f"_{model_name}" if len(model_names) > 1 else ""
        trialwise_figure_path = (
            output_dir / f"posterior_predictive_trialwise{suffix}.png"
        )
        scatter_figure_path = output_dir / f"posterior_predictive_scatter{suffix}.png"
        _plot_combined_ppc_trialwise(model_df, trialwise_figure_path, model_name)
        _plot_combined_ppc_scatter(model_df, scatter_figure_path, model_name)
        console.print(
            f"[green]✓ Saved combined PPC figure[/green] {trialwise_figure_path}"
        )
        console.print(
            f"[green]✓ Saved combined PPC figure[/green] {scatter_figure_path}"
        )
    return trialwise_df, summary_df


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options: the results and data folders, the models to
        analyse and their output folders (two lists, paired by position), the
        conditions to compile, the correction across parameters, and the
        overwrite flag.
    """
    parser = argparse.ArgumentParser(
        description="Compile and analyze MCMC fitting results"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing result subdirectories'
    )
    parser.add_argument(
        '--output-dir',
        nargs="+",
        default=DEFAULT_OUTPUT_DIRS,
        help="Folders to save compiled results and figures in, one for each "
        "--analysis-model, in the same order. Default: %(default)s.",
    )
    parser.add_argument(
        '--analysis-model',
        nargs="+",
        default=DEFAULT_ANALYSIS_MODELS,
        help="Models to compile. Each model's outputs (summary, plots, subject-level "
        "export, and parameter analyses) go to the --output-dir at the same place "
        "in its list. Each model must have been fitted in every condition given by "
        '--conditions. Use "none" to compile every model found and skip the second-block '
        "parameter analyses. Default: %(default)s.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITION_METADATA),
        default=list(CONDITION_METADATA),
        help='Conditions to compile. Unless the analysis model is "none", these must include all '
        "the second-block conditions (%s). Default: all conditions."
        % ", ".join(SECOND_BLOCK_CONDITIONS),
    )
    parser.add_argument(
        '--cross-parameter-correction',
        type=str,
        default='none',
        choices=CROSS_PARAMETER_CORRECTION_CHOICES,
        help='Optional omnibus-only p-value correction applied across parameters '
             'for each ANOVA effect. Default: %(default)s.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/prepared',
        help='Directory containing prepared .npz files (for subject ID lookup)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow writing into a non-empty output directory.'
    )
    return parser.parse_args()


def _check_analysis_choices(analysis_models: List[str], output_dirs: List[str]) -> None:
    """Check that each analysis model has its own output folder.

    Args:
        analysis_models: Models to compile, from ``--analysis-model``.
        output_dirs: Output folders, from ``--output-dir``, paired with the
            models by position.

    Raises:
        ValueError: If the two lists have different lengths, or if two
            analyses would write into the same folder.
    """
    if len(analysis_models) != len(output_dirs):
        raise ValueError(
            f"Give one --output-dir for each --analysis-model: got {len(analysis_models)} "
            f"model(s) ({', '.join(analysis_models)}) and {len(output_dirs)} folder(s) "
            f"({', '.join(output_dirs)})."
        )
    resolved = [Path(folder).resolve() for folder in output_dirs]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "Each --analysis-model needs a different --output-dir, but some folders "
            f"repeat: {', '.join(output_dirs)}."
        )


def check_analysis(
    analysis_model: str,
    output_dir: Path,
    all_fits: List[FitFolder],
    conditions: List[str],
    results_dir: Path,
    data_dir: Path,
) -> CheckedAnalysis:
    """Pick, read and check the fits of one analysis model, writing nothing.

    Args:
        analysis_model: Model to compile, or ``"none"`` for every model found.
        output_dir: Folder the analysis will be written to.
        all_fits: Every fit folder found for the chosen conditions.
        conditions: Conditions to compile.
        results_dir: Folder with the fit folders, named in error messages.
        data_dir: Folder with the prepared data files, for subject IDs.

    Returns:
        The analysis's fits, parameter bounds (None for ``"none"``), summary
        table and subject-level table.

    Raises:
        ValueError: If the analysis model has no fits, is missing in a listed
            condition, or if a fit folder or the parameter bounds file cannot
            be used.
    """
    fits = all_fits
    bounds = None
    if analysis_model != "none":
        available_models = sorted({fit.model for fit in all_fits})
        if analysis_model not in available_models:
            raise ValueError(
                f"The analysis model '{analysis_model}' has no fit folders in {results_dir}. "
                f"Models found: {', '.join(available_models)}."
            )
        fits = [fit for fit in all_fits if fit.model == analysis_model]
        fitted_conditions = {fit.condition for fit in fits}
        missing_conditions = [c for c in conditions if c not in fitted_conditions]
        if missing_conditions:
            raise ValueError(
                f"The analysis model '{analysis_model}' has no fit folder in these "
                f"conditions: {', '.join(missing_conditions)}."
            )
        console.print(f"[blue]Applying model filter: {analysis_model}[/blue]")
        bounds = find_bounds_file(fits)

    analysis_df = aggregate_results(fits)
    subject_df = build_subject_parameter_table(fits, data_dir)
    return CheckedAnalysis(
        analysis_model, output_dir, fits, bounds, analysis_df, subject_df
    )


def write_analysis(analysis: CheckedAnalysis, cross_parameter_correction: str) -> None:
    """Write one checked analysis into its output folder.

    Writes the summary table, parameter plots, subject-level parameter table,
    combined posterior predictive checks and, unless the analysis model is
    ``"none"``, the second-block parameter analyses.

    Args:
        analysis: The analysis from ``check_analysis``.
        cross_parameter_correction: Correction for testing several
            parameters: ``"none"``, ``"fdr_bh"`` or ``"bonferroni"``.

    Raises:
        ValueError: If a posterior predictive file cannot be read, or if a
            second-block analysis cannot be run, for example because a block
            combination has fewer than three participants.
        RuntimeError: If a Bayes factor fails or is not finite.
    """
    output_dir = analysis.output_dir
    console.print(
        f"\n[bold cyan]Writing {analysis.analysis_model} to {output_dir}[/bold cyan]"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    create_summary_table(analysis.analysis_df, output_dir)
    plot_parameter_distributions(analysis.analysis_df, output_dir)
    export_subject_parameters(analysis.subject_df, output_dir)
    export_combined_ppc(analysis.fits, output_dir)

    if analysis.analysis_model != "none":
        run_parameter_analyses(
            analysis.subject_df,
            output_dir,
            analysis.analysis_model,
            analysis.bounds,
            cross_parameter_correction=cross_parameter_correction,
        )
    else:
        console.print(
            "[yellow]Second-block analyses not run: the analysis model is none.[/yellow]"
        )


def main():
    """Compile the model fits into summary tables, figures and analyses.

    Each analysis model is compiled into its own output folder; by default,
    ``rescorla_wagner_single_value`` into ``results/compiled`` and
    ``dual_lr_single_value`` into ``results/compiled_dual_lr``. Before writing
    anything, the function checks every output folder, finds the fit folders
    for the chosen conditions, checks that each analysis model is present in
    every one of them, and reads and checks every fit and each participant's
    estimates, for all the analyses. It then writes each analysis in turn:
    the summary table, parameter plots, subject-level parameter table,
    combined posterior predictive checks and second-block parameter analyses.
    With the analysis model ``none``, every model found is compiled into that
    folder and the second-block analyses are not run. Fits that did not
    converge are compiled, with their converged flag recorded.

    Raises:
        RuntimeError: If an output folder already has files and
            ``--overwrite`` was not given.
        ValueError: If the models and output folders do not pair up, if an
            analysis model is not among the fits, if it is missing in a listed
            condition, if the listed conditions leave out a second-block
            condition, or if a fit folder or the parameter bounds file cannot
            be used.
    """
    args = parse_args()

    console.print(
        "\n[bold magenta]═══════════════════════════════════════[/bold magenta]"
    )
    console.print("[bold magenta]   Results Compilation Pipeline[/bold magenta]")
    console.print(
        "[bold magenta]═══════════════════════════════════════[/bold magenta]\n"
    )

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    _check_analysis_choices(args.analysis_model, args.output_dir)
    output_dirs = [Path(folder) for folder in args.output_dir]

    for output_dir in output_dirs:
        existing_entries = _existing_output_entries(output_dir)
        if existing_entries and not args.overwrite:
            preview = ", ".join(existing_entries[:10])
            if len(existing_entries) > 10:
                preview += ", ..."
            raise RuntimeError(
                f"Existing compiled outputs found in {output_dir}: {preview}. "
                "Use --overwrite to replace them."
            )
        if existing_entries and args.overwrite:
            console.print(
                f"[yellow]Overwriting compiled outputs in {output_dir} ({len(existing_entries)} existing entries).[/yellow]"
            )

    if any(model != "none" for model in args.analysis_model):
        missing_second_block = [
            c for c in SECOND_BLOCK_CONDITIONS if c not in args.conditions
        ]
        if missing_second_block:
            raise ValueError(
                "The second-block analyses need all the second-block conditions, but "
                f"--conditions leaves out: {', '.join(missing_second_block)}. "
                "Add them, or use --analysis-model none."
            )

    all_fits = find_fit_folders(results_dir, args.conditions)

    # Read and check the fits, and every participant's estimates, for every
    # analysis before writing anything
    analyses = [
        check_analysis(
            model, output_dir, all_fits, args.conditions, results_dir, data_dir
        )
        for model, output_dir in zip(args.analysis_model, output_dirs)
    ]

    for analysis in analyses:
        write_analysis(analysis, args.cross_parameter_correction)

    console.print("\n[bold green]✓ Compilation complete![/bold green]\n")
    for analysis in analyses:
        console.print(
            f"[bold]{analysis.analysis_model} saved to:[/bold] {analysis.output_dir.absolute()}"
        )
    console.print()


if __name__ == '__main__':
    main()
