"""Model recovery for the integrated models.

Simulates choices from a known model with known parameters, fits the candidate
models to the simulated data, and checks whether the fitting picks out the
model that produced the data and recovers its parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .data_generation import load_parameter_bounds
from .hierarchical_bayes import transform_param
from .integrated_model_selection import load_integrated_dataset
from .model_simulation import create_initial_state, get_model_info

DEFAULT_RESULTS_DIR = "results/integrated_model_recovery"
DEFAULT_PARETO_K_THRESHOLD = 0.7
REPLICATE_SEED_STRIDE = 100_000
MODEL_SEED_STRIDE = 1_000


def build_recovery_job_grid(
    replicates: int,
    simulated_models: Sequence[str],
    fitted_models: Sequence[str],
) -> list[dict[str, Any]]:
    """List every fit to run for the model-recovery check.

    Makes one job for each combination of replicate, simulated model and fitted
    model. Jobs are ordered by replicate, then simulated model, then fitted
    model.

    Args:
        replicates: Number of simulated data sets per simulated model.
        simulated_models: Names of the models that generate the data.
        fitted_models: Names of the models that are fitted to each data set.

    Returns:
        List of dictionaries, each with the keys ``replicate`` (counted from
        0), ``simulated_model`` and ``fitted_model``.

    Raises:
        ValueError: If ``replicates`` is less than 1.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least 1")

    jobs = []
    for replicate in range(replicates):
        for simulated_model in simulated_models:
            for fitted_model in fitted_models:
                jobs.append(
                    {
                        "replicate": replicate,
                        "simulated_model": simulated_model,
                        "fitted_model": fitted_model,
                    }
                )
    return jobs


def compute_recovery_seed(
    replicate: int,
    simulated_model: str,
    model_order: Sequence[str],
    base_seed: int = 42,
) -> int:
    """Work out the random seed for one simulated data set.

    The seed is the base seed, plus 100,000 for each replicate, plus 1,000 for
    each position of the simulated model in ``model_order``. Each replicate and
    each simulated model therefore gets its own seed.

    Args:
        replicate: Replicate number, counted from 0.
        simulated_model: Name of the model that generates the data.
        model_order: Model names in the fixed order used for seeding.
        base_seed: Seed for replicate 0 and the first model.

    Returns:
        The seed for this replicate and simulated model.

    Raises:
        ValueError: If ``simulated_model`` is not in ``model_order``.
    """
    if simulated_model not in model_order:
        raise ValueError(f"Unknown simulated model '{simulated_model}'")

    model_index = list(model_order).index(simulated_model)
    return base_seed + (replicate * REPLICATE_SEED_STRIDE) + (model_index * MODEL_SEED_STRIDE)


def recovery_output_dir(
    base_output_dir: Path,
    replicate: int,
    simulated_model: str,
    fitted_model: str,
) -> Path:
    """Give the folder that holds the results of one fit.

    Args:
        base_output_dir: Folder that holds all model-recovery results.
        replicate: Replicate number, counted from 0.
        simulated_model: Name of the model that generated the data.
        fitted_model: Name of the model that was fitted to the data.

    Returns:
        The path ``<base_output_dir>/replicate_NNN/<simulated_model>/<fitted_model>``,
        where NNN is the replicate number padded to three digits.
    """
    return (
        base_output_dir
        / f"replicate_{replicate:03d}"
        / simulated_model
        / fitted_model
    )


def _simulate_subject_session_choices(
    model_name: str,
    params: dict[str, float],
    outcomes_0: np.ndarray,
    outcomes_1: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the choices of one participant, trial by trial.

    On each trial the model gives the probability of choosing option 0. That
    probability is clipped to between 1e-5 and 1 - 1e-5, and a choice is drawn
    from it (1 = option 0, 0 = option 1). The model is then updated with the
    drawn choice and the trial's outcomes.

    Args:
        model_name: Name of the model.
        params: Parameter values for this participant. Names ending in
            ``_init`` set the starting state and are not used in the updates.
        outcomes_0: Outcomes of option 0, one value per trial.
        outcomes_1: Outcomes of option 1, one value per trial.
        rng: Random number generator used to draw the choices.

    Returns:
        Tuple of two arrays of shape (n_trials,): the probability of choosing
        option 0 on each trial, and the simulated choices.
    """
    model_info = get_model_info(model_name)
    state = create_initial_state(model_name, params)
    scan_params = {key: value for key, value in params.items() if not key.endswith("_init")}

    n_trials = len(outcomes_0)
    choice_prob_0 = np.zeros(n_trials, dtype=float)
    choices = np.zeros(n_trials, dtype=float)

    for trial_index in range(n_trials):
        current_state = {name: state[name] for name in model_info.state_vars}
        results = model_info.func(
            outcome_option0=outcomes_0[trial_index],
            trial_number=trial_index,
            outcome_option1=outcomes_1[trial_index],
            choice=0.0,
            **current_state,
            **scan_params,
        )
        choice_prob = float(np.asarray(results[0]))
        choice_prob = float(np.clip(choice_prob, 1e-5, 1.0 - 1e-5))
        choice = float(rng.binomial(1, choice_prob))

        choice_prob_0[trial_index] = choice_prob
        choices[trial_index] = choice

        update_results = model_info.func(
            outcome_option0=outcomes_0[trial_index],
            trial_number=trial_index,
            outcome_option1=outcomes_1[trial_index],
            choice=choice,
            **current_state,
            **scan_params,
        )
        state = {
            name: update_results[index]
            for name, index in zip(model_info.state_vars, model_info.state_indices)
        }

    return choice_prob_0, choices


def _select_balanced_subject_indices(
    condition_labels: Sequence[str],
    conditions: Sequence[str],
    n_subjects: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Pick participants so that every condition is represented evenly.

    Goes through the conditions in order, taking one randomly chosen unused
    participant from each condition per round, until ``n_subjects`` have been
    chosen. A condition that runs out of participants is skipped.

    Args:
        condition_labels: Condition of every participant in the data set.
        conditions: Names of the conditions to draw from.
        n_subjects: Number of participants to pick.
        rng: Random number generator used to shuffle each condition's
            participants.

    Returns:
        Sorted integer array of the positions of the chosen participants.

    Raises:
        ValueError: If ``n_subjects`` is smaller than the number of conditions,
            or if fewer than ``n_subjects`` participants are available across
            the conditions.
    """
    if n_subjects < len(conditions):
        raise ValueError(
            f"Requested {n_subjects} subjects but need at least {len(conditions)} to represent "
            "each integrated condition"
        )

    condition_labels = np.asarray(condition_labels)
    condition_indices = {
        condition: rng.permutation(np.flatnonzero(condition_labels == condition)).tolist()
        for condition in conditions
    }

    selected: list[int] = []
    while len(selected) < n_subjects:
        made_progress = False
        for condition in conditions:
            if condition_indices[condition]:
                selected.append(condition_indices[condition].pop())
                made_progress = True
                if len(selected) == n_subjects:
                    break
        if not made_progress:
            break

    if len(selected) != n_subjects:
        raise ValueError(
            f"Requested {n_subjects} subjects but only {len(selected)} could be sampled across "
            "the selected conditions"
        )

    return np.asarray(sorted(selected), dtype=int)


def simulate_integrated_dataset(
    simulated_model: str,
    conditions: Sequence[str],
    n_subjects: int,
    seed: int,
    data_dir: str | Path = "data/prepared",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Simulate a data set from one model, using real task outcomes.

    Picks ``n_subjects`` real participants, spread evenly over the conditions,
    and keeps their outcome sequences and context codes. It then draws a true
    parameter value for each simulated participant from a hierarchical
    distribution with the same structure as the fitted model. For each
    parameter that is not fixed, the raw value is a group mean, plus a weight
    for each of current volatility, previous volatility and their interaction
    times the participant's context code, plus a group spread times a
    participant offset. The group mean is drawn from Normal(prior_mu_mean,
    prior_mu_sigma). The three context weights are drawn from Normal(0,
    prior_mu_sigma), not from prior_beta_sigma as in the fitted model. The
    spread is the absolute value of a Normal(0, prior_sigma_scale) draw, and
    ``fixed_subject_sigma`` is ignored. The participant offsets are drawn from
    Normal(0, prior_offset_sigma). The raw value is mapped into the parameter's
    bounds. Finally it simulates each participant's choices with
    ``_simulate_subject_session_choices``, using the seed plus the
    participant's position as that participant's seed.

    Args:
        simulated_model: Name of the model that generates the choices.
        conditions: Names of the conditions to draw participants from.
        n_subjects: Number of participants to simulate.
        seed: Random seed for picking participants, drawing parameters and
            simulating choices.
        data_dir: Folder with the prepared empirical data.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.

    Returns:
        Dictionary with these entries:
        - outcomes_0, outcomes_1: outcomes of each option, shape
          (n_subjects, n_trials)
        - choices: simulated choices (1 = option 0), shape (n_subjects, n_trials)
        - context: context codes, shape (n_subjects, 3)
        - subject_ids, subject_labels, condition_labels: details of the
          participants whose outcomes were used
        - choice_probabilities: probability of choosing option 0, shape
          (n_subjects, n_trials)
        - true_parameters: true value of each parameter for each participant
        - hierarchy_parameters: group-level values drawn for each parameter
        - simulation_seed: the seed
        - simulated_model: the model name
        - conditions: the condition names

    Raises:
        ValueError: If ``n_subjects`` is larger than the number of participants
            available across the conditions.
    """
    empirical = load_integrated_dataset(conditions=conditions, data_dir=data_dir)
    available_subjects = len(empirical["subject_ids"])
    if n_subjects > available_subjects:
        raise ValueError(
            f"Requested {n_subjects} subjects but only {available_subjects} are available "
            f"across the selected conditions"
        )

    rng = np.random.RandomState(seed)
    selected_indices = _select_balanced_subject_indices(
        empirical["condition_labels"],
        empirical["conditions"],
        n_subjects,
        rng,
    )

    outcomes_0 = np.asarray(empirical["outcomes_0"])[selected_indices].astype(float)
    outcomes_1 = np.asarray(empirical["outcomes_1"])[selected_indices].astype(float)
    context = np.asarray(empirical["context"])[selected_indices].astype(float)
    selected_subject_ids = [empirical["subject_ids"][index] for index in selected_indices]
    selected_subject_labels = [empirical["subject_labels"][index] for index in selected_indices]
    selected_condition_labels = [empirical["condition_labels"][index] for index in selected_indices]

    bounds = load_parameter_bounds(simulated_model, config_path)

    current_vol = context[:, 0]
    previous_vol = context[:, 1]
    current_previous_interaction = context[:, 2]

    subject_parameters: dict[str, np.ndarray] = {}
    hierarchy_parameters: dict[str, dict[str, Any]] = {}

    for parameter_name, parameter_bounds in bounds.items():
        if "fixed" in parameter_bounds:
            value = np.full(n_subjects, float(parameter_bounds["fixed"]), dtype=float)
            subject_parameters[parameter_name] = value
            hierarchy_parameters[parameter_name] = {"fixed": float(parameter_bounds["fixed"])}
            continue

        prior_mu_mean = float(parameter_bounds.get("prior_mu_mean", 0.0))
        prior_mu_sigma = float(parameter_bounds.get("prior_mu_sigma", 0.2))
        prior_sigma_scale = float(parameter_bounds.get("prior_sigma_scale", 1.0))
        prior_offset_sigma = float(parameter_bounds.get("prior_offset_sigma", 0.5))

        mu = rng.normal(prior_mu_mean, prior_mu_sigma)
        beta_current = rng.normal(0.0, prior_mu_sigma)
        beta_previous = rng.normal(0.0, prior_mu_sigma)
        beta_interaction = rng.normal(0.0, prior_mu_sigma)
        subject_sigma = abs(rng.normal(0.0, prior_sigma_scale))
        subject_offset_raw = rng.normal(0.0, prior_offset_sigma, size=n_subjects)

        raw_value = (
            mu
            + (beta_current * current_vol)
            + (beta_previous * previous_vol)
            + (beta_interaction * current_previous_interaction)
            + (subject_sigma * subject_offset_raw)
        )
        transformed = np.asarray(
            transform_param(
                jnp.asarray(raw_value),
                float(parameter_bounds["min"]),
                float(parameter_bounds["max"]),
            )
        )

        subject_parameters[parameter_name] = transformed
        hierarchy_parameters[parameter_name] = {
            "mu": float(mu),
            "beta_current_vol": float(beta_current),
            "beta_previous_vol": float(beta_previous),
            "beta_current_previous_interaction": float(beta_interaction),
            "subject_sigma": float(subject_sigma),
            "subject_offset_raw": subject_offset_raw.tolist(),
        }

    choices = np.zeros_like(outcomes_0, dtype=float)
    choice_probabilities = np.zeros_like(outcomes_0, dtype=float)
    for subject_index in range(n_subjects):
        subject_params = {
            parameter_name: float(values[subject_index])
            for parameter_name, values in subject_parameters.items()
        }
        subject_rng = np.random.RandomState(seed + subject_index)
        subject_choice_prob, subject_choices = _simulate_subject_session_choices(
            simulated_model,
            subject_params,
            outcomes_0[subject_index],
            outcomes_1[subject_index],
            subject_rng,
        )
        choice_probabilities[subject_index] = subject_choice_prob
        choices[subject_index] = subject_choices

    return {
        "outcomes_0": jnp.asarray(outcomes_0),
        "outcomes_1": jnp.asarray(outcomes_1),
        "choices": jnp.asarray(choices),
        "context": jnp.asarray(context),
        "subject_ids": selected_subject_ids,
        "subject_labels": selected_subject_labels,
        "condition_labels": selected_condition_labels,
        "choice_probabilities": choice_probabilities,
        "true_parameters": subject_parameters,
        "hierarchy_parameters": hierarchy_parameters,
        "simulation_seed": seed,
        "simulated_model": simulated_model,
        "conditions": list(empirical["conditions"]),
    }


def rank_recovery_fits(loo_df: pd.DataFrame) -> pd.DataFrame:
    """Rank the fitted models within each simulated data set.

    Within each data set (one replicate of one simulated model), every fit is
    ranked by ``elpd_loo``, with rank 1 for the highest. Fits that did not
    converge, and fits with a high Pareto k, are ranked like any other fit.
    Their convergence and Pareto k values stay in the returned table.

    Args:
        loo_df: One row per fit, with the columns ``replicate``,
            ``simulated_model``, ``fitted_model`` and ``elpd_loo``.

    Returns:
        Copy of ``loo_df`` with an added column, ``loo_rank_within_dataset``.

    Raises:
        ValueError: If any fit has an ``elpd_loo`` that is missing or not
            finite. The message names the replicate, simulated model and
            fitted model of each such fit.
    """
    ranked = loo_df.copy()
    elpd_loo = pd.to_numeric(ranked["elpd_loo"], errors="coerce")
    not_finite = ranked[~np.isfinite(elpd_loo)]
    if not not_finite.empty:
        fits = "; ".join(
            f"replicate={row.replicate}, simulated_model={row.simulated_model}, "
            f"fitted_model={row.fitted_model}"
            for row in not_finite.itertuples()
        )
        raise ValueError(
            f"Cannot rank fits with a missing or non-finite elpd_loo: {fits}"
        )

    ranked["loo_rank_within_dataset"] = (
        elpd_loo.groupby([ranked["replicate"], ranked["simulated_model"]], sort=False)
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return ranked


def select_recovery_winners(loo_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best-ranked fit for each simulated data set.

    Args:
        loo_df: One row per fit. See ``rank_recovery_fits`` for the columns.

    Returns:
        Data frame with one row per data set, sorted by replicate, simulated
        model and fitted model.

    Raises:
        ValueError: If any fit has an ``elpd_loo`` that is missing or not
            finite (see ``rank_recovery_fits``).
    """
    ranked = rank_recovery_fits(loo_df)
    winners = ranked[ranked["loo_rank_within_dataset"] == 1].copy()
    return winners.sort_values(["replicate", "simulated_model", "fitted_model"]).reset_index(drop=True)


def validate_recovery_fit_grid(
    loo_df: pd.DataFrame,
    model_order: Sequence[str],
    expected_replicates: int | Sequence[int],
) -> list[tuple[int, str, str]]:
    """Find the fits that are missing from the results.

    Every replicate should have a fit for each combination of simulated model
    and fitted model in ``model_order``. The expected replicates are always
    given, never worked out from ``loo_df``, so a replicate with no fits at
    all is reported as missing. The results must also hold nothing else: a
    fit from another replicate or model, or a second fit for the same
    combination, would change the winner counts, so it raises an error.

    Args:
        loo_df: One row per fit, with the columns ``replicate``,
            ``simulated_model`` and ``fitted_model``.
        model_order: Names of the models that were both simulated and fitted.
        expected_replicates: Replicates that should be present. A number
            means replicates 0 up to that number minus 1. A list gives the
            replicate numbers.

    Returns:
        List of (replicate, simulated_model, fitted_model) for each missing fit.

    Raises:
        ValueError: If ``expected_replicates`` is a number less than 1, or if
            any fit is outside the expected replicates and models, or appears
            more than once. The message names each such fit.
    """
    observed_keys = [
        (int(row.replicate), str(row.simulated_model), str(row.fitted_model))
        for row in loo_df.itertuples(index=False)
    ]
    observed = set(observed_keys)
    if isinstance(expected_replicates, int):
        if expected_replicates < 1:
            raise ValueError("expected_replicates must be at least 1")
        replicate_values = list(range(expected_replicates))
    else:
        replicate_values = sorted({int(value) for value in expected_replicates})

    expected = {
        (replicate, simulated_model, fitted_model)
        for replicate in replicate_values
        for simulated_model in model_order
        for fitted_model in model_order
    }
    unexpected = sorted(observed - expected)
    duplicated = sorted({key for key in observed_keys if observed_keys.count(key) > 1})
    if unexpected or duplicated:
        problems = [f"not expected: {key}" for key in unexpected]
        problems += [f"more than one fit: {key}" for key in duplicated]
        raise ValueError(
            "Recovery results do not match the expected replicates and models "
            "(replicate, simulated_model, fitted_model): " + "; ".join(problems)
        )

    missing: list[tuple[int, str, str]] = []

    for replicate in replicate_values:
        for simulated_model in model_order:
            for fitted_model in model_order:
                candidate = (replicate, simulated_model, fitted_model)
                if candidate not in observed:
                    missing.append(candidate)

    return missing


def summarize_recovery_dataset_counts(
    ranked_df: pd.DataFrame,
    winners_df: pd.DataFrame,
) -> dict[str, int]:
    """Count the simulated data sets, and how many have a winning model.

    A data set is one replicate of one simulated model. Every ranked data set
    has a winner, so the two counts should be equal.

    Args:
        ranked_df: Ranked fits for all data sets, from ``rank_recovery_fits``.
        winners_df: Winning fits, from ``select_recovery_winners``.

    Returns:
        Dictionary with ``n_total_datasets`` and ``n_winner_datasets``.
    """
    total_datasets = int(ranked_df.groupby(["replicate", "simulated_model"], sort=False).ngroups)
    winner_datasets = int(winners_df.groupby(["replicate", "simulated_model"], sort=False).ngroups)
    return {
        "n_total_datasets": total_datasets,
        "n_winner_datasets": winner_datasets,
    }


def build_recovery_matrices(
    winners_df: pd.DataFrame,
    model_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the confusion matrix and the inversion matrix from the winners.

    First counts how often each fitted model won for each simulated model. The
    confusion matrix divides each row by its total, so it shows, for each
    simulated model, the share of data sets won by each fitted model. The
    inversion matrix divides each column by its total, so it shows, for each
    winning model, the share of its wins that came from each simulated model.
    Rows or columns with no wins are filled with zeros. Winners whose model
    names are not in ``model_order`` are ignored.

    Args:
        winners_df: Winning fits, with the columns ``simulated_model`` and
            ``fitted_model``.
        model_order: Model names, in the order of the rows and columns.

    Returns:
        Tuple of the confusion matrix (rows: simulated model, columns:
        recovered model) and the inversion matrix (rows: recovered model,
        columns: simulated model).
    """
    counts = pd.DataFrame(0.0, index=list(model_order), columns=list(model_order))
    for row in winners_df.itertuples(index=False):
        if row.simulated_model in counts.index and row.fitted_model in counts.columns:
            counts.loc[row.simulated_model, row.fitted_model] += 1.0

    row_sums = counts.sum(axis=1).replace(0.0, np.nan)
    confusion = counts.div(row_sums, axis=0).fillna(0.0)

    column_sums = counts.sum(axis=0).replace(0.0, np.nan)
    inversion = counts.div(column_sums, axis=1).T.fillna(0.0)
    inversion.index.name = "recovered_model"
    inversion.columns.name = "simulated_model"

    confusion.index.name = "simulated_model"
    confusion.columns.name = "recovered_model"
    return confusion, inversion


def _safe_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Pearson correlation, or NaN if it cannot be computed.

    Args:
        x: First set of values.
        y: Second set of values.

    Returns:
        The Pearson correlation, or NaN if either set has fewer than two values
        or has all values equal.
    """
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_integrated_parameter_recovery_artifacts(
    result_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], az.InferenceData]:
    """Load the saved simulation details and posterior for one fit.

    Reads ``simulation_metadata.json``, ``dataset_metadata.json`` and
    ``posterior.nc`` from the result folder.

    Args:
        result_dir: Folder that holds the results of one fit.

    Returns:
        Tuple of the simulation metadata, the dataset metadata and the
        posterior samples.

    Raises:
        FileNotFoundError: If any of the three files is missing. The message
            names the folder and the missing files.
    """
    simulation_metadata_path = result_dir / "simulation_metadata.json"
    dataset_metadata_path = result_dir / "dataset_metadata.json"
    posterior_path = result_dir / "posterior.nc"

    artifact_paths = [simulation_metadata_path, dataset_metadata_path, posterior_path]
    missing_paths = [path.name for path in artifact_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Incomplete integrated parameter-recovery artifacts in {result_dir}: "
            f"missing {', '.join(missing_paths)}"
        )

    with open(simulation_metadata_path, "r", encoding="utf-8") as handle:
        simulation_metadata = json.load(handle)
    with open(dataset_metadata_path, "r", encoding="utf-8") as handle:
        dataset_metadata = json.load(handle)
    posterior = az.from_netcdf(posterior_path)
    return simulation_metadata, dataset_metadata, posterior


def _posterior_site_to_subject_samples(site: Any, site_name: str, n_subjects: int) -> np.ndarray:
    """Reshape a participant-level posterior into one row per posterior sample.

    Chains and draws are merged into a single first axis.

    Args:
        site: Posterior samples of one participant-level parameter, with the
            participant on the last axis (for example chain, draw, participant).
        site_name: Name of the posterior site, used in error messages.
        n_subjects: Number of participants expected on the last axis.

    Returns:
        Array of shape (n_chains * n_draws, n_subjects).

    Raises:
        ValueError: If the samples have fewer than two dimensions, or the last
            axis does not have ``n_subjects`` values.
    """
    values = np.asarray(site, dtype=float)
    if values.ndim < 2:
        raise ValueError(
            f"Posterior site '{site_name}' must have at least two dimensions, got shape {values.shape}."
        )
    if values.shape[-1] != n_subjects:
        raise ValueError(
            f"Posterior site '{site_name}' has {values.shape[-1]} subject values, expected {n_subjects}."
        )
    return values.reshape((-1, values.shape[-1]))


def _resolve_parameter_recovery_result_dir(
    results_dir: Path,
    row: dict[str, Any],
) -> Path:
    """Find the folder that holds the saved results of one fit.

    Args:
        results_dir: Folder that holds all model-recovery results.
        row: Results of one fit, with ``replicate``, ``simulated_model`` and
            ``fitted_model``, and optionally ``summary_path``.

    Returns:
        The folder that contains ``summary_path`` if the row has one. Otherwise
        the folder given by ``recovery_output_dir`` for the row.
    """
    summary_path = row.get("summary_path")
    if summary_path:
        return Path(summary_path).parent
    replicate = int(row["replicate"])
    simulated_model = str(row["simulated_model"])
    fitted_model = str(row["fitted_model"])
    return recovery_output_dir(results_dir, replicate, simulated_model, fitted_model)


def build_integrated_parameter_recovery(
    ranked_df: pd.DataFrame,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check how well each simulated parameter is recovered by the same model.

    Only fits where the fitted model is the simulated model are used. For each
    such fit, the function loads the saved true parameters and posterior. It
    compares each participant's true value with the posterior mean. Every such
    fit must have its saved results. Fits that did not converge, or have a high
    Pareto k, are kept, and their ``converged``, ``n_bad_pareto_k`` and
    ``max_pareto_k`` values are recorded in both tables.

    Args:
        ranked_df: One row per fit, from ``rank_recovery_fits``. Rows may also
            have ``summary_path``, ``converged``, ``n_bad_pareto_k`` and
            ``max_pareto_k``.
        results_dir: Folder that holds all model-recovery results.

    Returns:
        Tuple of two data frames. The first has one row per participant and
        parameter, with the true value, the posterior mean and the error
        (posterior mean minus true value). The second has one row per fit and
        parameter, with the Pearson correlation, the root-mean-square error and
        the bias (mean error) across participants. Both are empty, with the
        same columns, if there are no fits of a model to its own data.

    Raises:
        TypeError: If the simulation metadata has no ``true_parameters``
            object.
        KeyError: If the dataset metadata lacks participant details, or the
            posterior lacks a parameter's site.
        ValueError: If the participant details differ in length, or a true
            parameter does not have one value per participant.
        FileNotFoundError: If a fit's results folder lacks any of its saved
            files. The message names the folder.
    """
    diagonal_df = ranked_df[ranked_df["simulated_model"] == ranked_df["fitted_model"]].copy()
    if diagonal_df.empty:
        empty_rows_columns = [
            "replicate",
            "simulated_model",
            "fitted_model",
            "parameter",
            "subject_index",
            "subject_id",
            "subject_label",
            "condition_label",
            "true_value",
            "recovered_mean",
            "error",
            "converged",
            "n_bad_pareto_k",
            "max_pareto_k",
        ]
        empty_summary_columns = [
            "replicate",
            "simulated_model",
            "fitted_model",
            "parameter",
            "pearson_r",
            "rmse",
            "bias",
            "n_subjects",
            "converged",
            "n_bad_pareto_k",
            "max_pareto_k",
        ]
        return pd.DataFrame(columns=empty_rows_columns), pd.DataFrame(columns=empty_summary_columns)

    parameter_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for row in diagonal_df.to_dict(orient="records"):
        result_dir = _resolve_parameter_recovery_result_dir(results_dir, row)
        simulation_metadata, dataset_metadata, posterior = (
            _load_integrated_parameter_recovery_artifacts(result_dir)
        )
        true_parameters = simulation_metadata.get("true_parameters")
        if not isinstance(true_parameters, dict):
            raise TypeError(
                f"simulation_metadata.json in {result_dir} must contain a 'true_parameters' object"
            )

        subject_ids = dataset_metadata.get("subject_ids")
        subject_labels = dataset_metadata.get("subject_labels")
        condition_labels = dataset_metadata.get("condition_labels")
        if subject_ids is None or subject_labels is None or condition_labels is None:
            raise KeyError(
                f"dataset_metadata.json in {result_dir} must contain subject_ids, subject_labels, and condition_labels"
            )

        subject_ids_list = list(subject_ids)
        subject_labels_list = list(subject_labels)
        condition_labels_list = list(condition_labels)
        n_subjects = len(subject_ids_list)
        if len(subject_labels_list) != n_subjects or len(condition_labels_list) != n_subjects:
            raise ValueError(
                f"dataset_metadata.json in {result_dir} has inconsistent subject metadata lengths"
            )

        for parameter_name, true_values_raw in true_parameters.items():
            site_name = f"{row['fitted_model']}_{parameter_name}_subj"
            if site_name not in posterior.posterior.data_vars:
                raise KeyError(f"Missing posterior site '{site_name}' in {result_dir}")

            true_values = np.asarray(true_values_raw, dtype=float)
            if true_values.ndim != 1 or true_values.shape[0] != n_subjects:
                raise ValueError(
                    f"True parameter '{parameter_name}' in {result_dir} must contain {n_subjects} subject values"
                )

            recovered_samples = _posterior_site_to_subject_samples(
                posterior.posterior[site_name], site_name, n_subjects
            )
            recovered_means = recovered_samples.mean(axis=0)
            errors = recovered_means - true_values

            for subject_index in range(n_subjects):
                parameter_rows.append(
                    {
                        "replicate": int(row["replicate"]),
                        "simulated_model": str(row["simulated_model"]),
                        "fitted_model": str(row["fitted_model"]),
                        "parameter": parameter_name,
                        "subject_index": subject_index,
                        "subject_id": subject_ids_list[subject_index],
                        "subject_label": subject_labels_list[subject_index],
                        "condition_label": condition_labels_list[subject_index],
                        "true_value": float(true_values[subject_index]),
                        "recovered_mean": float(recovered_means[subject_index]),
                        "error": float(errors[subject_index]),
                        "converged": bool(row.get("converged", False)),
                        "n_bad_pareto_k": int(row.get("n_bad_pareto_k", 0) or 0),
                        "max_pareto_k": float(row.get("max_pareto_k", float("nan"))),
                    }
                )

            summary_rows.append(
                {
                    "replicate": int(row["replicate"]),
                    "simulated_model": str(row["simulated_model"]),
                    "fitted_model": str(row["fitted_model"]),
                    "parameter": parameter_name,
                    "pearson_r": _safe_pearson_r(true_values, recovered_means),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "bias": float(np.mean(errors)),
                    "n_subjects": int(n_subjects),
                    "converged": bool(row.get("converged", False)),
                    "n_bad_pareto_k": int(row.get("n_bad_pareto_k", 0) or 0),
                    "max_pareto_k": float(row.get("max_pareto_k", float("nan"))),
                }
            )

    rows_df = pd.DataFrame(parameter_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not rows_df.empty:
        rows_df = rows_df.sort_values(
            ["replicate", "simulated_model", "parameter", "subject_index"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["replicate", "simulated_model", "parameter"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    return rows_df, summary_df


GROUP_EFFECT_NAMES = (
    "mu",
    "beta_current_vol",
    "beta_previous_vol",
    "beta_current_previous_interaction",
    "subject_sigma",
)

GROUP_EFFECT_COLUMNS = [
    "replicate",
    "simulated_model",
    "parameter",
    "effect",
    "true_value",
    "posterior_mean",
    "eti95_lower",
    "eti95_upper",
    "converged",
    "n_bad_pareto_k",
    "max_pareto_k",
]


def build_integrated_group_effect_recovery(
    ranked_df: pd.DataFrame,
    results_dir: Path,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Check how well the group-level effects are recovered by the same model.

    Only fits where the fitted model is the simulated model are used. Every
    parameter that is not fixed has a group mean (``mu``), a weight for each of
    current volatility, previous volatility and their interaction, and, unless
    its bounds give ``fixed_subject_sigma``, a spread between participants
    (``subject_sigma``). For each of these effects, the function takes the true
    value that the simulation drew (saved under ``hierarchy_parameters``) and
    pairs it with the posterior of the same effect, read from the site
    ``<model>_<parameter>_<effect>``. The posterior is summarised by its mean
    and its 95% equal-tailed interval (the 2.5th and 97.5th percentiles of all
    chains and draws). The values are on the raw scale, before the parameter is
    mapped into its bounds. Fits that did not converge, or have a high Pareto
    k, are kept, and their ``converged``, ``n_bad_pareto_k`` and
    ``max_pareto_k`` values are recorded.

    Args:
        ranked_df: One row per fit, from ``rank_recovery_fits``, with
            ``converged``, ``n_bad_pareto_k`` and ``max_pareto_k``. Rows may
            also have ``summary_path``.
        results_dir: Folder that holds all model-recovery results.
        config_path: Optional path to the parameter bounds file, used to see
            which parameters have a fixed spread between participants. If None,
            uses the default file.

    Returns:
        Data frame with one row per fit, parameter and effect, and the columns
        ``replicate``, ``simulated_model``, ``parameter``, ``effect``,
        ``true_value``, ``posterior_mean``, ``eti95_lower``, ``eti95_upper``,
        ``converged``, ``n_bad_pareto_k`` and ``max_pareto_k``. It is empty,
        with the same columns, if there are no fits of a model to its own data.

    Raises:
        TypeError: If the simulation metadata has no ``hierarchy_parameters``
            object.
        ValueError: If ``hierarchy_parameters`` lacks a parameter of the model,
            or holds one that the model does not have. The message names the
            folder and the parameters. It is also raised if a true value or a
            posterior draw is not finite, or a posterior site is not a single
            value for each chain and draw.
        KeyError: If a true effect is missing from the metadata, or the
            posterior lacks an effect's site. The message names the folder and
            the parameter or site.
        FileNotFoundError: If a fit's results folder lacks any of its saved
            files. The message names the folder.
    """
    diagonal_df = ranked_df[ranked_df["simulated_model"] == ranked_df["fitted_model"]]

    rows: list[dict[str, Any]] = []
    for row in diagonal_df.to_dict(orient="records"):
        result_dir = _resolve_parameter_recovery_result_dir(results_dir, row)
        simulation_metadata, _, posterior = (
            _load_integrated_parameter_recovery_artifacts(result_dir)
        )
        hierarchy = simulation_metadata.get("hierarchy_parameters")
        if not isinstance(hierarchy, dict):
            raise TypeError(
                f"simulation_metadata.json in {result_dir} must contain a "
                "'hierarchy_parameters' object"
            )
        model_name = str(row["fitted_model"])
        bounds = load_parameter_bounds(model_name, config_path)
        # Every parameter of the model must have saved true values. A parameter
        # that is missing here would otherwise drop out of the comparison
        # without any sign that it was never checked.
        absent = sorted(set(bounds) - set(hierarchy))
        unexpected = sorted(set(hierarchy) - set(bounds))
        if absent or unexpected:
            raise ValueError(
                f"hierarchy_parameters in {result_dir} does not match the parameters "
                f"of {model_name}. Missing: {absent}. Not in the bounds: {unexpected}."
            )

        for parameter_name, true_effects in hierarchy.items():
            # A fixed parameter has no group-level effects to recover.
            if "fixed" in true_effects:
                continue
            for effect in GROUP_EFFECT_NAMES:
                # With a fixed spread between participants the fitted model
                # has no spread site, so there is nothing to compare.
                if (
                    effect == "subject_sigma"
                    and "fixed_subject_sigma" in bounds[parameter_name]
                ):
                    continue
                if effect not in true_effects:
                    raise KeyError(
                        f"simulation_metadata.json in {result_dir} has no true "
                        f"'{effect}' for parameter '{parameter_name}'"
                    )
                site_name = f"{model_name}_{parameter_name}_{effect}"
                if site_name not in posterior.posterior.data_vars:
                    raise KeyError(
                        f"Missing posterior site '{site_name}' in {result_dir}"
                    )

                # Pair the value that the simulation drew for this effect with
                # the posterior of the same effect, so each point in the
                # recovery figure compares like with like.
                true_value = float(true_effects[effect])
                draws = np.asarray(posterior.posterior[site_name], dtype=float)
                if draws.ndim != 2:
                    raise ValueError(
                        f"Posterior site '{site_name}' in {result_dir} must have one value "
                        f"per chain and draw, got shape {draws.shape}."
                    )
                if not np.isfinite(true_value) or not np.all(np.isfinite(draws)):
                    raise ValueError(
                        f"True value or posterior draws of site '{site_name}' in "
                        f"{result_dir} are not finite."
                    )
                draws = draws.reshape(-1)
                lower, upper = np.percentile(draws, [2.5, 97.5])
                rows.append(
                    {
                        "replicate": int(row["replicate"]),
                        "simulated_model": str(row["simulated_model"]),
                        "parameter": parameter_name,
                        "effect": effect,
                        "true_value": true_value,
                        "posterior_mean": float(draws.mean()),
                        "eti95_lower": float(lower),
                        "eti95_upper": float(upper),
                        "converged": bool(row["converged"]),
                        "n_bad_pareto_k": int(row["n_bad_pareto_k"]),
                        "max_pareto_k": float(row["max_pareto_k"]),
                    }
                )

    effects_df = pd.DataFrame(rows, columns=GROUP_EFFECT_COLUMNS)
    # A stable sort keeps the effects in the order of GROUP_EFFECT_NAMES.
    return effects_df.sort_values(
        ["replicate", "simulated_model", "parameter"], kind="stable"
    ).reset_index(drop=True)


def load_recovery_fit_summary(summary_path: Path) -> dict[str, Any]:
    """Read the summary of one fit from its JSON file.

    Args:
        summary_path: Path to the JSON summary file.

    Returns:
        The contents of the file as a dictionary.
    """
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
