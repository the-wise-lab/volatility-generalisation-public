#!/usr/bin/env python3
"""Run posterior predictive checks for saved integrated fits.

This script reuses existing ``posterior.nc`` and ``metadata.json`` outputs from
integrated model selection runs. It reconstructs trialwise ``P(choose option 0)``
from Bernoulli log likelihoods, aggregates by condition/trial, writes tables,
and saves compact diagnostic figures.

It runs after the integrated model fits are complete (see
``docs/3-model-selection-and-recovery.md``), for every fitted model or for
the models named on the command line. For each model and condition, it
compares the observed proportion of option 0 choices on each trial with the
model's predicted probability and a 95% posterior predictive interval. It
also reports the accuracy of the model's most likely choice, and prints a
warning for any condition below the accuracy threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from rich.console import Console

from volatility_generalisation.modelling_code.integrated_model_selection import load_integrated_dataset
from volatility_generalisation.modelling_code.hierarchical_bayes import run_single_subject_scan
from volatility_generalisation.modelling_code.model_simulation import MODEL_INFO
from volatility_generalisation.modelling_code.posterior_utils import posterior_interval

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

console = Console()

DEFAULT_RESULTS_DIR = "results/integrated_model_selection"
DEFAULT_DATA_DIR = "data/prepared"


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options: results, output and data folders, models,
        overwrite flag, number of posterior draws, accuracy warning threshold
        and random seed.
    """
    parser = argparse.ArgumentParser(description="Run integrated posterior predictive checks")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Derived output directory (default: results/integrated_model_selection/posterior_predictive)",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty derived output directory",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--n-draws",
        type=int,
        default=None,
        help="Maximum number of posterior draws to subsample per model (default: all)",
    )
    parser.add_argument(
        "--accuracy-warning-threshold",
        type=float,
        default=0.60,
        help="Warn when posterior-mean binary choice accuracy is below this value",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_fit(result_dir: Path) -> tuple[dict[str, Any], az.InferenceData]:
    """Load the saved metadata and posterior samples of one model fit.

    Args:
        result_dir: Folder for one model, holding ``metadata.json`` and
            ``posterior.nc``.

    Returns:
        Tuple of the metadata dictionary and the posterior samples with their
        log-likelihood.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    metadata_path = result_dir / "metadata.json"
    posterior_path = result_dir / "posterior.nc"
    missing = [path.name for path in [metadata_path, posterior_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing expected integrated fit outputs in {result_dir}: {', '.join(missing)}"
        )

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata, az.from_netcdf(posterior_path)


def _fit_is_complete(result_dir: Path) -> bool:
    """Check whether a model folder has both saved fit files.

    Args:
        result_dir: Folder for one model.

    Returns:
        True if both ``metadata.json`` and ``posterior.nc`` exist.
    """
    return (result_dir / "metadata.json").exists() and (result_dir / "posterior.nc").exists()


def _available_models(results_dir: Path) -> list[str]:
    """List the models that have a complete saved fit.

    A subfolder named after a known model (a key of ``MODEL_INFO``) must hold
    both fit files, whether or not the fit converged. Other subfolders, such as
    ``model_comparison``, are used only if they hold both fit files.

    Args:
        results_dir: Folder with one subfolder per model.

    Returns:
        Sorted names of the subfolders that hold both fit files.

    Raises:
        FileNotFoundError: If the results folder does not exist, or a
            subfolder named after a known model lacks ``metadata.json`` or
            ``posterior.nc``. The message names the folder.
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    names: list[str] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if _fit_is_complete(child):
            names.append(child.name)
        elif child.name in MODEL_INFO:
            raise FileNotFoundError(
                f"Incomplete integrated fit for model {child.name} in {child}: "
                "it needs both metadata.json and posterior.nc"
            )
    return names


def _resolve_models(
    results_dir: Path,
    explicit_models: list[str] | None,
) -> list[str]:
    """Choose which models to check.

    Args:
        results_dir: Folder with one subfolder per model.
        explicit_models: Models given on the command line, or None to check
            every model with a complete fit.

    Returns:
        The models to check.

    Raises:
        FileNotFoundError: If a model named on the command line has no
            complete fit, or if no models are named and a folder named after
            a known model is incomplete.
        RuntimeError: If no models are named and none has a complete fit.
    """
    if explicit_models is not None:
        missing = [name for name in explicit_models if not _fit_is_complete(results_dir / name)]
        if missing:
            raise FileNotFoundError(
                f"Missing explicit integrated fit(s): {', '.join(missing)}"
            )
        return explicit_models

    models = _available_models(results_dir)
    if not models:
        raise RuntimeError(
            f"No complete integrated fits found in {results_dir}"
        )
    return models


def _resolve_output_dir(
    results_dir: Path,
    output_dir: str | None,
) -> Path:
    """Choose the folder where the tables and figures are written.

    Args:
        results_dir: Folder with the model fits.
        output_dir: Folder given on the command line, or None to use the
            ``posterior_predictive`` folder inside the results folder.

    Returns:
        Path of the output folder.
    """
    if output_dir is not None:
        return Path(output_dir)
    return results_dir / "posterior_predictive"


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create the output folder, refusing to reuse a non-empty one by accident.

    Args:
        output_dir: Folder for the tables and figures.
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


def _choice_prob_from_log_likelihood(log_likelihood: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Recover the model's choice probability from the log-likelihood.

    The log-likelihood of each trial is the log probability of the choice the
    participant made. Taking its exponential gives that probability. If the
    choice was coded 0, the probability is subtracted from 1, so the result is
    always the probability of the choice coded 1. Choices are coded 1 for
    option 0 and 0 for option 1, so the result is the probability of choosing
    option 0.

    Args:
        log_likelihood: Log-likelihood of each observed choice, with shape
            (n_draws, n_participants, n_trials).
        choices: Observed choices, coded 0 or 1, with shape
            (n_participants, n_trials).

    Returns:
        Array with the same shape as ``log_likelihood``, holding the
        probability of choice 1, limited to the range 0 to 1.
    """
    observed_choice_prob = np.exp(np.asarray(log_likelihood, dtype=float))
    choices_array = np.asarray(choices, dtype=float)
    reconstructed = np.where(choices_array > 0.5, observed_choice_prob, 1.0 - observed_choice_prob)
    return np.clip(reconstructed, 0.0, 1.0)


def _select_draw_indices(n_draws_available: int, n_draws: int | None, seed: int) -> np.ndarray:
    """Choose which posterior draws to use.

    Uses every draw if no limit is given or the limit is at least the number
    of draws. Otherwise it picks a random subset without repeats.

    Args:
        n_draws_available: Number of posterior draws available, with chains
            combined.
        n_draws: Largest number of draws to use, or None to use all.
        seed: Seed for the random choice of draws.

    Returns:
        Sorted array of the indices of the chosen draws.
    """
    if n_draws is None or n_draws >= n_draws_available:
        return np.arange(n_draws_available)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_draws_available, size=n_draws, replace=False))


def _posterior_subject_choice_probabilities(
    idata: az.InferenceData,
    data: dict[str, Any],
    model_name: str,
    n_draws: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Compute choice probabilities by running the model on posterior draws.

    For each chosen posterior draw, this runs the model on each participant's
    outcomes and choices, using that participant's parameter values from the
    draw. It is used when the saved fit has no usable trial-level
    log-likelihood.

    Args:
        idata: Posterior samples with participant-level parameters.
        data: Trial data with ``choices``, ``outcomes_0`` and ``outcomes_1``,
            each of shape (n_participants, n_trials).
        model_name: Name of the model.
        n_draws: Largest number of posterior draws to use, or None to use all.
        seed: Seed for the random choice of draws.

    Returns:
        Array of shape (n_draws_used, n_participants, n_trials) with the
        model's probability of choosing option 0.

    Raises:
        ValueError: If the posterior has no participant-level variables for
            the model.
    """
    posterior = idata.posterior
    prefix = f"{model_name}_"
    subject_suffix = "_subj"
    subject_variables = [
        name
        for name in posterior.data_vars
        if name.startswith(prefix) and name.endswith(subject_suffix)
    ]
    if not subject_variables:
        raise ValueError(
            f"No subject-level posterior variables found for model '{model_name}'"
        )

    choices = np.asarray(data["choices"], dtype=float)
    outcomes_0 = np.asarray(data["outcomes_0"], dtype=float)
    outcomes_1 = np.asarray(data["outcomes_1"], dtype=float)
    n_subjects = choices.shape[0]

    subject_params_flat = {
        variable_name[len(prefix) : -len(subject_suffix)]: np.asarray(posterior[variable_name].values).reshape(-1, n_subjects)
        for variable_name in subject_variables
    }
    n_draws_available = next(iter(subject_params_flat.values())).shape[0]
    draw_indices = _select_draw_indices(n_draws_available, n_draws, seed)
    selected_params = {name: values[draw_indices] for name, values in subject_params_flat.items()}

    def per_subject_predict(params, o0, o1, c):
        """Run the model for one participant.

        Args:
            params: Parameter values for this participant.
            o0: Outcomes of option 0 on each trial.
            o1: Outcomes of option 1 on each trial.
            c: The participant's choices on each trial.

        Returns:
            Array with the model's probability of choosing option 0 on each
            trial.
        """
        return run_single_subject_scan(model_name, params, o0, o1, c)

    subject_vmapped = jax.vmap(per_subject_predict, in_axes=(0, 0, 0, 0))
    draw_vmapped = jax.vmap(
        lambda params, subject_o0, subject_o1, subject_choices: subject_vmapped(
            params, subject_o0, subject_o1, subject_choices
        ),
        in_axes=(0, None, None, None),
    )

    return np.asarray(
        draw_vmapped(
            selected_params,
            jnp.asarray(outcomes_0),
            jnp.asarray(outcomes_1),
            jnp.asarray(choices),
        )
    )


def _trialwise_condition_ppc(
    idata: az.InferenceData,
    data: dict[str, Any],
    model_name: str,
    n_draws: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare predicted and observed choices by condition and trial.

    Gets the model's choice probability for each posterior draw, participant
    and trial, either from the saved log-likelihood or by re-running the
    model. For each condition, it then averages the predicted probability
    over participants and draws, and simulates choices to get a 95% posterior
    predictive interval (2.5th to 97.5th percentile). The summary also gives
    the accuracy of predicting each participant's choice from the posterior
    mean probability (choice 1 if the probability is at least 0.5), and the
    mean absolute error and root mean squared error between predicted and
    observed probabilities.

    Args:
        idata: Posterior samples with the log-likelihood.
        data: Trial data with ``choices`` and ``condition_labels``. Choices
            have shape (n_participants, n_trials).
        model_name: Name of the model.
        n_draws: Largest number of posterior draws to use, or None to use all.
        seed: Seed for choosing draws and simulating choices.

    Returns:
        Tuple of two tables. The first has one row per condition and trial.
        The second has one row per condition.
    """
    choices = np.asarray(data["choices"], dtype=float)
    condition_labels = list(data["condition_labels"])
    conditions = list(dict.fromkeys(condition_labels))
    rng = np.random.default_rng(seed)

    log_likelihood = np.asarray(idata.log_likelihood["obs"].values, dtype=float)
    if log_likelihood.ndim >= 4 and log_likelihood.shape[-1] == choices.shape[1]:
        log_likelihood = log_likelihood.reshape(-1, *log_likelihood.shape[2:])
        draw_indices = _select_draw_indices(log_likelihood.shape[0], n_draws, seed)
        log_likelihood = log_likelihood[draw_indices]
        reconstructed_probs = _choice_prob_from_log_likelihood(log_likelihood, choices)
    else:
        reconstructed_probs = _posterior_subject_choice_probabilities(
            idata=idata,
            data=data,
            model_name=model_name,
            n_draws=n_draws,
            seed=seed,
        )
        draw_indices = np.arange(reconstructed_probs.shape[0])

    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    condition_labels_array = np.asarray(condition_labels)
    n_trials = choices.shape[1]

    for condition in conditions:
        condition_mask = condition_labels_array == condition
        if not np.any(condition_mask):
            continue

        observed_by_trial = choices[condition_mask].mean(axis=0)
        condition_probabilities = reconstructed_probs[:, condition_mask, :]
        posterior_by_draw_trial = condition_probabilities.mean(axis=1)
        posterior_predictive_by_draw_trial = rng.binomial(1, condition_probabilities).mean(axis=1)
        predicted_by_trial = posterior_by_draw_trial.mean(axis=0)
        predicted_by_subject_trial = condition_probabilities.mean(axis=0) >= 0.5
        predictive_accuracy = (predicted_by_subject_trial == choices[condition_mask]).mean()
        lower, upper = posterior_interval(posterior_predictive_by_draw_trial, axis=0)

        for trial in range(n_trials):
            trial_rows.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "trial": trial,
                    "n_subjects": int(condition_mask.sum()),
                    "observed_choice_prob": float(observed_by_trial[trial]),
                    "predicted_choice_prob": float(predicted_by_trial[trial]),
                    "predicted_choice_prob_ci_lower": float(lower[trial]),
                    "predicted_choice_prob_ci_upper": float(upper[trial]),
                    "n_draws_used": int(len(draw_indices)),
                }
            )

        residuals = predicted_by_trial - observed_by_trial
        summary_rows.append(
            {
                "model": model_name,
                "condition": condition,
                "n_subjects": int(condition_mask.sum()),
                "n_trials": int(n_trials),
                "n_draws_used": int(len(draw_indices)),
                "observed_choice_prob_mean": float(observed_by_trial.mean()),
                "predicted_choice_prob_mean": float(predicted_by_trial.mean()),
                "predictive_accuracy": float(predictive_accuracy),
                "mae": float(np.abs(residuals).mean()),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "mean_interval_width": float(np.mean(upper - lower)),
            }
        )

    return pd.DataFrame(trial_rows), pd.DataFrame(summary_rows)


def _warn_low_predictive_accuracy(summary_df: pd.DataFrame, threshold: float) -> int:
    """Print a warning for each condition with low predictive accuracy.

    Args:
        summary_df: Per-condition summary table with a ``predictive_accuracy``
            column.
        threshold: Accuracy below which a condition is reported.

    Returns:
        Number of conditions below the threshold. It is 0 if the table is
        empty or has no accuracy column.
    """
    if summary_df.empty or "predictive_accuracy" not in summary_df:
        return 0

    low_accuracy = summary_df[summary_df["predictive_accuracy"] < threshold]
    for row in low_accuracy.itertuples(index=False):
        console.print(
            "[bold yellow]Warning:[/bold yellow] "
            f"{row.model} / {row.condition} predictive accuracy is "
            f"{row.predictive_accuracy:.3f}, below threshold {threshold:.3f}."
        )
    return int(len(low_accuracy))


def _plot_model_scatter(trialwise_df: pd.DataFrame, output_path: Path, model_name: str) -> None:
    """Plot predicted against observed choice probability, one point per trial.

    All conditions share one panel, with a different colour for each
    condition and a dashed line where predicted equals observed.

    Args:
        trialwise_df: Trial-level table for one model.
        output_path: File in which to save the figure.
        model_name: Model name, shown in the figure title.
    """
    plt.figure(figsize=(6, 6))
    conditions = list(dict.fromkeys(trialwise_df["condition"]))
    cmap = plt.get_cmap("tab10")
    for index, condition in enumerate(conditions):
        condition_df = trialwise_df[trialwise_df["condition"] == condition]
        plt.scatter(
            condition_df["observed_choice_prob"],
            condition_df["predicted_choice_prob"],
            s=18,
            alpha=0.75,
            color=cmap(index % 10),
            label=condition,
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1)
    plt.xlabel("Observed P(choose option 0)")
    plt.ylabel("Predicted P(choose option 0)")
    plt.title(f"{model_name} posterior predictive scatter")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_model_trialwise(trialwise_df: pd.DataFrame, output_path: Path, model_name: str) -> None:
    """Plot predicted and observed choice probability by trial, per condition.

    Draws one panel per condition, stacked vertically. Each panel shows the
    observed proportion of option 0 choices, the model's predicted
    probability, and the 95% posterior predictive interval.

    Args:
        trialwise_df: Trial-level table for one model.
        output_path: File in which to save the figure.
        model_name: Model name, shown in the figure title.
    """
    conditions = list(dict.fromkeys(trialwise_df["condition"]))
    n_conditions = len(conditions)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(10, max(3.0, 2.8 * n_conditions)), sharex=True)
    axes_array = np.atleast_1d(axes)

    for axis, condition in zip(axes_array, conditions):
        condition_df = trialwise_df[trialwise_df["condition"] == condition].sort_values("trial")
        trials = condition_df["trial"].to_numpy()
        axis.fill_between(
            trials,
            condition_df["predicted_choice_prob_ci_lower"].to_numpy(),
            condition_df["predicted_choice_prob_ci_upper"].to_numpy(),
            color="steelblue",
            alpha=0.2,
            label="95% posterior predictive interval",
        )
        axis.plot(trials, condition_df["predicted_choice_prob"], color="steelblue", linewidth=2, label="Predicted")
        axis.plot(trials, condition_df["observed_choice_prob"], color="black", linewidth=1.5, label="Observed")
        axis.axhline(0.5, color="red", linestyle=":", alpha=0.4)
        axis.set_title(condition)
        axis.set_ylabel("P(choose option 0)")
        axis.set_ylim(0, 1)

    axes_array[-1].set_xlabel("Trial")
    axes_array[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{model_name} posterior predictive by condition", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the posterior predictive checks for every chosen model.

    Loads each model's fit and its trial data, compares predicted and observed
    choices, prints a warning for conditions with low accuracy, then saves the
    tables for all models and two figures for each model.

    Raises:
        KeyError: If a model's ``metadata.json`` has no list of conditions.
        FileNotFoundError: If a chosen model's fit is missing.
        RuntimeError: If no complete fits are found or the output folder
            cannot be used.
    """
    args = parse_args()
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)

    models = _resolve_models(results_dir, args.models)
    output_dir = _resolve_output_dir(
        results_dir,
        args.output_dir,
    )
    _prepare_output_dir(output_dir, args.overwrite)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Results directory:[/cyan] {results_dir}")
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Models:[/cyan] {', '.join(models)}")

    trialwise_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    model_payloads: list[tuple[str, pd.DataFrame]] = []
    draw_seed = args.seed

    for model_name in models:
        metadata, idata = _load_fit(results_dir / model_name)
        conditions = metadata.get("conditions")
        if not conditions:
            raise KeyError(f"metadata.json for {model_name} must contain conditions")

        data = load_integrated_dataset(conditions=conditions, data_dir=data_dir)
        trialwise_df, summary_df = _trialwise_condition_ppc(
            idata=idata,
            data=data,
            model_name=model_name,
            n_draws=args.n_draws,
            seed=draw_seed,
        )
        trialwise_frames.append(trialwise_df)
        summary_frames.append(summary_df)
        model_payloads.append((model_name, trialwise_df))
        _warn_low_predictive_accuracy(summary_df, args.accuracy_warning_threshold)

    trialwise_output = pd.concat(trialwise_frames, ignore_index=True) if trialwise_frames else pd.DataFrame()
    summary_output = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    trialwise_output.to_csv(tables_dir / "posterior_predictive_trialwise.csv", index=False)
    summary_output.to_csv(tables_dir / "posterior_predictive_summary.csv", index=False)

    for model_name, trialwise_df in model_payloads:
        _plot_model_scatter(
            trialwise_df,
            figures_dir / f"posterior_predictive_scatter_{model_name}.png",
            model_name,
        )
        _plot_model_trialwise(
            trialwise_df,
            figures_dir / f"posterior_predictive_trialwise_{model_name}.png",
            model_name,
        )


if __name__ == "__main__":
    main()
