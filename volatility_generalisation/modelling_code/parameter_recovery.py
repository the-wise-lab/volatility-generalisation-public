"""Parameter recovery workflow built on hierarchical MCMC fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from .data_generation import (
    generate_synthetic_data_single_model,
    load_parameter_bounds,
    settings_file_record,
)
from .data_preparation import OUTCOME_CODING_KEY, normalize_to_reward_coded
from .diagnostics import diagnose_mcmc, print_chain_failure_summary, save_diagnostics
from .hierarchical_bayes import fit_mcmc

console = Console()

RECOVERY_BETA_SHAPE = 3.0


def load_prepared_schedule(data_path: Union[str, Path]) -> Dict[str, jnp.ndarray]:
    """Load the outcome schedule of the first participant in a prepared file.

    The schedule is later reused for every simulated participant. The outcomes
    are converted to reward coding (1 = safe) using the file's
    ``outcome_coding`` entry. A file without that entry is treated as
    explosion-coded and flipped, as the fitting code does.

    Args:
        data_path: Path to the prepared ``.npz`` file, which must contain
            ``outcomes_0`` and ``outcomes_1``.

    Returns:
        Dictionary with ``outcomes_0`` and ``outcomes_1``, each a one-dimensional
        reward-coded array with one value per trial, taken from the first
        participant.

    Raises:
        ValueError: If the outcome arrays are not two-dimensional (participants
            by trials), do not have the same shape, or have no participants.
    """
    with np.load(Path(data_path), allow_pickle=True) as data_np:
        outcomes_0 = np.asarray(data_np["outcomes_0"], dtype=np.float64)
        outcomes_1 = np.asarray(data_np["outcomes_1"], dtype=np.float64)
        metadata = {}
        if OUTCOME_CODING_KEY in data_np:
            metadata[OUTCOME_CODING_KEY] = str(data_np[OUTCOME_CODING_KEY])

    if outcomes_0.ndim != 2 or outcomes_1.ndim != 2:
        raise ValueError("Prepared outcomes must be 2D arrays of shape (subjects, trials).")
    if outcomes_0.shape != outcomes_1.shape:
        raise ValueError("Prepared outcomes_0 and outcomes_1 arrays must have matching shapes.")
    if outcomes_0.shape[0] == 0:
        raise ValueError("Prepared data must contain at least one subject row.")

    reward_coded = normalize_to_reward_coded(
        {"outcomes_0": outcomes_0[0], "outcomes_1": outcomes_1[0]},
        metadata=metadata,
    )
    return {
        "outcomes_0": jnp.array(reward_coded["outcomes_0"], dtype=jnp.float64),
        "outcomes_1": jnp.array(reward_coded["outcomes_1"], dtype=jnp.float64),
    }


def get_parameter_names(
    model_name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """List the parameter names of a model, in the order of the bounds file.

    Args:
        model_name: Name of the model.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.

    Returns:
        List of parameter names.
    """
    return list(load_parameter_bounds(model_name, config_path).keys())


def pack_true_parameter_array(
    true_params: Dict[str, Any],
    param_names: Sequence[str],
) -> np.ndarray:
    """Combine the true parameter values into one participants-by-parameters array.

    Args:
        true_params: Dictionary from parameter name to a one-dimensional array
            with one true value per participant.
        param_names: Parameter names, in the order of the columns.

    Returns:
        Array of shape (n_subjects, n_parameters).

    Raises:
        KeyError: If a parameter name is missing from ``true_params``.
        ValueError: If a parameter's values are not one-dimensional.
    """
    columns = []
    for param_name in param_names:
        if param_name not in true_params:
            raise KeyError(f"Missing true parameter '{param_name}'.")
        values = np.asarray(true_params[param_name], dtype=float)
        if values.ndim != 1:
            raise ValueError(
                f"True parameter '{param_name}' must be 1D, got shape {values.shape}."
            )
        columns.append(values)

    return np.column_stack(columns)


def pack_posterior_parameter_samples(
    posterior_samples: Dict[str, Any],
    model_name: str,
    param_names: Sequence[str],
) -> np.ndarray:
    """Combine the participant-level posterior samples into one array.

    For each parameter, reads the site ``<model_name>_<parameter>_subj``.
    Samples from all chains are merged into one sample axis. A one-dimensional
    site is treated as a single sample.

    Args:
        posterior_samples: Dictionary from site name to posterior samples, with
            participants as the last axis.
        model_name: Name of the model, used to build the site names.
        param_names: Parameter names, in the order of the last axis of the
            result.

    Returns:
        Array of shape (n_samples, n_subjects, n_parameters).

    Raises:
        KeyError: If a parameter's site is missing from ``posterior_samples``.
        ValueError: If a site has a different shape from the first site, or
            has no dimensions.
    """
    packed = []
    expected_shape = None

    for param_name in param_names:
        site_name = f"{model_name}_{param_name}_subj"
        if site_name not in posterior_samples:
            raise KeyError(f"Missing posterior site '{site_name}'.")

        values = np.asarray(posterior_samples[site_name], dtype=float)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        elif values.ndim >= 2:
            values = values.reshape(-1, values.shape[-1])
        else:
            raise ValueError(
                f"Posterior site '{site_name}' must have at least one dimension."
            )

        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError(
                f"Posterior site '{site_name}' has shape {values.shape}, "
                f"expected {expected_shape}."
            )

        packed.append(values)

    return np.stack(packed, axis=-1)


def build_parameter_recovery_dataframe(
    condition: str,
    model_name: str,
    param_names: Sequence[str],
    true_values: np.ndarray,
    recovered_samples: np.ndarray,
) -> pd.DataFrame:
    """Tabulate each participant's true and recovered parameter values.

    The recovered value is the mean of the posterior samples.

    Args:
        condition: Condition name, stored in every row.
        model_name: Name of the model, stored in every row.
        param_names: Parameter names, in the order of the last axis of the
            arrays.
        true_values: True values, of shape (n_subjects, n_parameters).
        recovered_samples: Posterior samples, of shape
            (n_samples, n_subjects, n_parameters).

    Returns:
        Table with one row per participant. It has the columns ``condition``,
        ``model`` and ``subject_idx``, and for each parameter ``true_<name>``,
        ``recovered_<name>`` and ``error_<name>`` (recovered minus true).
    """
    recovered_means = recovered_samples.mean(axis=0)
    rows = []

    for subject_idx in range(true_values.shape[0]):
        row = {
            "condition": condition,
            "model": model_name,
            "subject_idx": subject_idx,
        }
        for param_idx, param_name in enumerate(param_names):
            true_value = float(true_values[subject_idx, param_idx])
            recovered_value = float(recovered_means[subject_idx, param_idx])
            row[f"true_{param_name}"] = true_value
            row[f"recovered_{param_name}"] = recovered_value
            row[f"error_{param_name}"] = recovered_value - true_value
        rows.append(row)

    return pd.DataFrame(rows)


def _safe_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Pearson correlation, or NaN when it is undefined.

    Args:
        x: First set of values.
        y: Second set of values.

    Returns:
        The correlation between ``x`` and ``y``. It is NaN if either array has
        fewer than two values or if either array is constant.
    """
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_parameter_recovery(
    condition: str,
    model_name: str,
    param_names: Sequence[str],
    true_values: np.ndarray,
    recovered_samples: np.ndarray,
) -> pd.DataFrame:
    """Summarise how well each parameter was recovered.

    Uses the mean of the posterior samples as the recovered value.

    Args:
        condition: Condition name, stored in every row.
        model_name: Name of the model, stored in every row.
        param_names: Parameter names, in the order of the last axis of the
            arrays.
        true_values: True values, of shape (n_subjects, n_parameters).
        recovered_samples: Posterior samples, of shape
            (n_samples, n_subjects, n_parameters).

    Returns:
        Table with one row per parameter and the columns ``condition``,
        ``model``, ``parameter``, ``pearson_r`` (correlation of true and
        recovered values), ``rmse`` (root mean squared error), ``bias`` (mean of
        recovered minus true) and ``n_subjects``.
    """
    recovered_means = recovered_samples.mean(axis=0)
    rows = []

    for param_idx, param_name in enumerate(param_names):
        true_param = true_values[:, param_idx]
        recovered_param = recovered_means[:, param_idx]
        errors = recovered_param - true_param
        rows.append(
            {
                "condition": condition,
                "model": model_name,
                "parameter": param_name,
                "pearson_r": _safe_pearson_r(true_param, recovered_param),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "bias": float(np.mean(errors)),
                "n_subjects": int(true_values.shape[0]),
            }
        )

    return pd.DataFrame(rows)


def run_parameter_recovery(
    models: Sequence[str],
    conditions: Sequence[str],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_subjects: int,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    target_accept_prob: float = 0.95,
    max_tree_depth: int = 10,
    thinning: int = 1,
    config_path: Optional[Union[str, Path]] = None,
    progress_bar: bool = True,
) -> List[Dict[str, Any]]:
    """Run parameter recovery for each model in each condition, using MCMC.

    For each condition and model, the function reuses the first participant's
    outcome schedule from the condition's prepared data, simulates
    ``n_subjects`` participants with parameters drawn from a beta distribution
    (shape 3.0) within the parameter bounds, fits the model with MCMC and
    compares the recovered parameters with the true ones. It writes two CSV
    files and a ``run_metadata.json`` (the run's condition,
    model, participant count and seed, and the bounds file's path and SHA-256
    hash, and the fit's ``converged`` flag, ``divergences``, ``max_rhat`` and
    ``min_ess``) to ``<output_dir>/<condition>/<model>/``. It also writes the
    fit's ``diagnostics.json``, and the summary CSV has ``rhat``, ``ess`` and
    ``converged`` columns. Every fit's chains are checked for convergence on
    the participant-level sites that are recovered. A fit that did not
    converge is recorded, not treated as a failure: its chain summary is
    printed and the run carries on with all its usual outputs. The run's
    random seed is ``seed`` plus 1000 times the condition's position plus the
    model's position.

    Args:
        models: Names of the models to run.
        conditions: Names of the conditions to run. Each needs a prepared
            ``<condition>.npz`` file in ``data_dir``.
        data_dir: Folder with the prepared ``.npz`` files.
        output_dir: Folder to write results to. It is created if needed.
        n_subjects: Number of simulated participants.
        num_warmup: Number of MCMC warmup steps.
        num_samples: Number of MCMC samples per chain.
        num_chains: Number of MCMC chains.
        seed: Base random seed.
        target_accept_prob: Target acceptance probability for NUTS.
        max_tree_depth: Maximum tree depth for NUTS.
        thinning: Keep every this many samples.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.
        progress_bar: Whether to show the MCMC progress bar.

    Returns:
        List with one dictionary per condition and model. Each has the keys
        ``condition``, ``model``, ``output_dir``, ``parameter_recovery_csv``,
        ``parameter_recovery_summary_csv`` and ``converged`` (whether the fit
        passed the convergence check).

    Raises:
        FileNotFoundError: If a condition's prepared ``.npz`` file does not
            exist.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_record = settings_file_record(config_path)

    results = []
    for condition_idx, condition in enumerate(conditions):
        data_path = data_dir / f"{condition}.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Prepared condition file not found: {data_path}")

        schedule = load_prepared_schedule(data_path)
        n_trials = int(schedule["outcomes_0"].shape[0])

        for model_idx, model_name in enumerate(models):
            run_seed = seed + (condition_idx * 1_000) + model_idx
            run_dir = output_dir / condition / model_name
            run_dir.mkdir(parents=True, exist_ok=True)

            console.print(
                Panel.fit(
                    (
                        f"[bold]{condition}[/bold] | [bold magenta]{model_name}[/bold magenta]\n"
                        f"Subjects: [cyan]{n_subjects}[/cyan]    "
                        f"Trials: [cyan]{n_trials}[/cyan]    "
                        f"Seed: [cyan]{run_seed}[/cyan]\n"
                        f"Output: [dim]{run_dir}[/dim]"
                    ),
                    border_style="green",
                )
            )

            synthetic = generate_synthetic_data_single_model(
                model_name=model_name,
                n_subjects=n_subjects,
                n_trials=n_trials,
                rng_seed=run_seed,
                external_outcomes=schedule,
                config_path=config_path,
                distribution="beta",
                beta_shape=RECOVERY_BETA_SHAPE,
            )

            param_names = get_parameter_names(model_name, config_path)
            true_values = pack_true_parameter_array(
                synthetic["true_params"],
                param_names,
            )

            mcmc = fit_mcmc(
                data=synthetic["data_for_fit"],
                model_names=model_name,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                seed=run_seed,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                thinning=thinning,
                progress_bar=progress_bar,
                config_path=config_path,
            )
            chain_samples = mcmc.get_samples(group_by_chain=True)
            site_names = [f"{model_name}_{name}_subj" for name in param_names]
            diagnostics = diagnose_mcmc(
                mcmc=mcmc,
                posterior_samples=chain_samples,
                param_names=site_names,
                verbose=True,
            )
            if not diagnostics["converged"]:
                print_chain_failure_summary(chain_samples, diagnostics)

            posterior_samples = mcmc.get_samples()
            recovered_samples = pack_posterior_parameter_samples(
                posterior_samples,
                model_name,
                param_names,
            )

            recovery_df = build_parameter_recovery_dataframe(
                condition,
                model_name,
                param_names,
                true_values,
                recovered_samples,
            )
            summary_df = summarize_parameter_recovery(
                condition,
                model_name,
                param_names,
                true_values,
                recovered_samples,
            )
            summary_df["rhat"] = [diagnostics["rhat"][site] for site in site_names]
            summary_df["ess"] = [diagnostics["ess"][site] for site in site_names]
            summary_df["converged"] = diagnostics["converged"]

            recovery_path = run_dir / "parameter_recovery.csv"
            summary_path = run_dir / "parameter_recovery_summary.csv"
            recovery_df.to_csv(recovery_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            save_diagnostics(diagnostics, str(run_dir / "diagnostics.json"))
            run_metadata = {
                "condition": condition,
                "model": model_name,
                "n_subjects": n_subjects,
                "seed": run_seed,
                "converged": diagnostics["converged"],
                "divergences": diagnostics["divergences"],
                "max_rhat": float(np.max(list(diagnostics["rhat"].values()))),
                "min_ess": float(np.min(list(diagnostics["ess"].values()))),
                **settings_record,
            }
            with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
                json.dump(run_metadata, handle, indent=2)

            results.append(
                {
                    "condition": condition,
                    "model": model_name,
                    "output_dir": str(run_dir),
                    "parameter_recovery_csv": str(recovery_path),
                    "parameter_recovery_summary_csv": str(summary_path),
                    "converged": diagnostics["converged"],
                }
            )

            console.print(
                f"[green]Saved[/green] CSV summaries and run details to [bold]{run_dir}[/bold]"
            )

    return results
