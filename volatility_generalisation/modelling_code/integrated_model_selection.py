"""Fit and compare the integrated hierarchical models.

The integrated models fit participants from several second-block conditions
together. Each parameter can depend on the volatility of the current block, the
volatility of the previous block and their interaction. This module loads the
prepared data, defines the model, fits it (with MCMC, Pathfinder or SVI) and
builds the posterior summaries used to compare models.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import arviz as az
import blackjax
import jax
import optax
import xarray as xr
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
from numpyro.infer.util import init_to_value, initialize_model

from .condition_metadata import build_context_regressors, second_block_conditions
from .data_generation import load_parameter_bounds
from .data_preparation import normalize_to_reward_coded
from .hierarchical_bayes import run_single_subject_scan, transform_param
from .pipeline_config import load_pipeline_config


def get_default_models(config_path: Union[str, Path] = "scripts/config.toml") -> list[str]:
    """Get the default list of models to fit from the settings file.

    Args:
        config_path: Path to the TOML settings file.

    Returns:
        The model names listed under ``fitting`` in the file, or an empty list
        if there are none.
    """
    config = load_pipeline_config(config_path)
    return config.get("fitting", {}).get("models", [])


def get_default_conditions(config_path: Union[str, Path] = "scripts/config.toml") -> list[str]:
    """Get the default list of conditions for integrated fitting.

    Args:
        config_path: Path to the TOML settings file.

    Returns:
        The condition names listed under ``integrated_fitting`` in the file, or
        the four second-block conditions if the file does not list any.
    """
    config = load_pipeline_config(config_path)
    return config.get("integrated_fitting", {}).get("conditions", second_block_conditions())


def _normalise_subject_ids(subject_ids: Optional[np.ndarray], n_subjects: int, condition: str) -> list[str]:
    """Turn participant identifiers into a list of text labels.

    Args:
        subject_ids: Participant identifiers, or None if the data file has none.
        n_subjects: Number of participants (rows) in the data.
        condition: Condition name, used in the error message.

    Returns:
        List of identifiers as text. If ``subject_ids`` is None, the labels are
        ``subject_000``, ``subject_001`` and so on.

    Raises:
        ValueError: If the number of identifiers differs from ``n_subjects``.
    """
    if subject_ids is None:
        return [f"subject_{index:03d}" for index in range(n_subjects)]

    ids = np.asarray(subject_ids).tolist()
    if len(ids) != n_subjects:
        raise ValueError(
            f"Condition {condition} has {len(ids)} subject IDs but {n_subjects} rows"
        )
    return [str(subject_id) for subject_id in ids]


def _load_condition_file(condition_path: Path) -> dict[str, Any]:
    """Load the prepared data of one condition from its ``.npz`` file.

    The outcomes are converted to reward coding (1 = safe, 0 = explosion) unless
    the file says they already are.

    Args:
        condition_path: Path to the condition's ``.npz`` file.

    Returns:
        Dictionary with ``outcomes_0``, ``outcomes_1`` and ``choices`` (float
        arrays with one row per participant and one column per trial) and
        ``subject_ids`` (None if the file has no participant identifiers).

    Raises:
        ValueError: If ``outcomes_0``, ``outcomes_1`` and ``choices`` do not
            all have the same shape.
    """
    loaded = np.load(condition_path, allow_pickle=True)
    outcomes_0 = np.asarray(loaded["outcomes_0"], dtype=float)
    outcomes_1 = np.asarray(loaded["outcomes_1"], dtype=float)
    choices = np.asarray(loaded["choices"], dtype=float)

    if outcomes_0.shape != outcomes_1.shape or outcomes_0.shape != choices.shape:
        raise ValueError(
            f"Mismatched array shapes in {condition_path.name}: "
            f"outcomes_0={outcomes_0.shape}, outcomes_1={outcomes_1.shape}, choices={choices.shape}"
        )

    # Build metadata dict from npz file attributes
    metadata = {}
    if "outcome_coding" in loaded:
        metadata["outcome_coding"] = str(loaded["outcome_coding"])

    # Normalise to reward-coded (handles files with and without outcome-coding
    # metadata)
    data_dict = {
        "outcomes_0": outcomes_0,
        "outcomes_1": outcomes_1,
        "choices": choices,
    }
    data_dict = normalize_to_reward_coded(data_dict, metadata=metadata)

    return {
        "outcomes_0": data_dict["outcomes_0"],
        "outcomes_1": data_dict["outcomes_1"],
        "choices": data_dict["choices"],
        "subject_ids": loaded["subject_ids"] if "subject_ids" in loaded else None,
    }


def load_integrated_dataset(
    conditions: Sequence[str],
    data_dir: Union[str, Path] = "data/prepared",
) -> dict[str, Any]:
    """Load and stack the prepared data of several conditions.

    Reads ``<data_dir>/<condition>.npz`` for each condition and places the
    participants of the conditions one after another, in the order given. Adds
    the context codes (current volatility, previous volatility and their
    interaction) for each participant's condition.

    Args:
        conditions: Names of the conditions to load. Each must be a
            second-block condition.
        data_dir: Folder with the prepared ``.npz`` files.

    Returns:
        Dictionary with these entries:
        - outcomes_0, outcomes_1, choices: arrays of shape
          (n_subjects, n_trials), reward-coded
        - context: array of shape (n_subjects, 3) with the context codes
        - subject_ids: participant identifiers, as text
        - subject_labels: identifiers of the form ``<condition>__<id>``
        - condition_labels: each participant's condition
        - conditions: the condition names that were requested

    Raises:
        ValueError: If no conditions are given, or if the conditions do not
            all have the same number of trials. (The context codes also raise
            a ValueError for a condition that is not a second-block condition.)
        FileNotFoundError: If a condition's ``.npz`` file does not exist.
    """
    if not conditions:
        raise ValueError("At least one condition is required")

    data_dir = Path(data_dir)
    condition_payloads: list[tuple[str, dict[str, Any], list[str]]] = []

    for condition in conditions:
        condition_path = data_dir / f"{condition}.npz"
        if not condition_path.exists():
            raise FileNotFoundError(f"Prepared condition file not found: {condition_path}")

        payload = _load_condition_file(condition_path)
        n_subjects = payload["choices"].shape[0]
        subject_ids = _normalise_subject_ids(payload["subject_ids"], n_subjects, condition)
        condition_payloads.append((condition, payload, subject_ids))

    subject_arrays: dict[str, list[np.ndarray]] = {"outcomes_0": [], "outcomes_1": [], "choices": []}
    n_trials: Optional[int] = None
    subject_ids_all: list[str] = []
    subject_labels: list[str] = []
    condition_labels: list[str] = []
    context_rows: list[np.ndarray] = []

    for condition, payload, subject_ids in condition_payloads:
        if n_trials is None:
            n_trials = payload["choices"].shape[-1]
        elif payload["choices"].shape[-1] != n_trials:
            raise ValueError(
                f"Condition {condition} has {payload['choices'].shape[-1]} trials; expected {n_trials}"
            )

        for key in subject_arrays:
            subject_arrays[key].append(payload[key])

        subject_ids_all.extend(subject_ids)
        subject_labels.extend(f"{condition}__{subject_id}" for subject_id in subject_ids)
        condition_labels.extend([condition] * len(subject_ids))

        condition_context = build_context_regressors([condition])[0]
        context_rows.append(np.repeat(condition_context[None, :], len(subject_ids), axis=0))

    stacked = {
        key: np.concatenate(value, axis=0).astype(float)
        for key, value in subject_arrays.items()
    }

    return {
        "outcomes_0": jnp.asarray(stacked["outcomes_0"]),
        "outcomes_1": jnp.asarray(stacked["outcomes_1"]),
        "choices": jnp.asarray(stacked["choices"]),
        "context": jnp.asarray(np.concatenate(context_rows, axis=0)),
        "subject_ids": subject_ids_all,
        "subject_labels": subject_labels,
        "condition_labels": condition_labels,
        "conditions": list(conditions),
    }


def _resolve_parameter_priors(parameter_bounds: dict[str, float]) -> dict[str, float]:
    """Get the prior settings of one parameter, filling in defaults.

    The defaults are 0.0 for ``prior_mu_mean``, 0.2 for ``prior_mu_sigma``, the
    value of ``prior_mu_sigma`` for ``prior_beta_sigma``, 1.0 for
    ``prior_sigma_scale`` and 0.5 for ``prior_offset_sigma``.

    Args:
        parameter_bounds: Bounds and optional prior settings of the parameter.

    Returns:
        Dictionary with the five prior settings.
    """
    prior_mu_sigma = parameter_bounds.get("prior_mu_sigma", 0.2)
    return {
        "prior_mu_mean": parameter_bounds.get("prior_mu_mean", 0.0),
        "prior_mu_sigma": prior_mu_sigma,
        "prior_beta_sigma": parameter_bounds.get("prior_beta_sigma", prior_mu_sigma),
        "prior_sigma_scale": parameter_bounds.get("prior_sigma_scale", 1.0),
        "prior_offset_sigma": parameter_bounds.get("prior_offset_sigma", 0.5),
    }


def integrated_hierarchical_model(
    data: Dict[str, jnp.ndarray],
    model_name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> None:
    """NumPyro model of one learning model fitted across conditions.

    A parameter with a ``fixed`` value is set to that value for every
    participant. Every other parameter has a group mean, a weight for each of
    current volatility, previous volatility and their interaction, and a spread
    between participants. If the bounds give ``fixed_subject_sigma``, there is
    no spread between participants. A participant's raw value is the group
    mean, plus the weighted context codes of their condition, plus the spread
    times a participant offset. The raw value is mapped into the parameter's
    bounds. The model then predicts each participant's choices trial by trial
    and scores the observed choices. It returns nothing.

    Args:
        data: Dictionary with ``outcomes_0``, ``outcomes_1`` and ``choices``,
            each of shape (n_subjects, n_trials), and ``context`` of shape
            (n_subjects, 3). Choices are 1 for option 0 and 0 for option 1.
        model_name: Name of the model.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.
    """
    outcomes_0 = data["outcomes_0"]
    outcomes_1 = data["outcomes_1"]
    choices = data["choices"]
    context = data["context"]
    n_subjects, _ = choices.shape

    current_vol = context[:, 0]
    previous_vol = context[:, 1]
    current_previous_interaction = context[:, 2]
    bounds = load_parameter_bounds(model_name, config_path)

    subject_params: dict[str, jnp.ndarray] = {}
    source_mus: dict[str, jnp.ndarray] = {}
    for parameter_name, parameter_bounds in bounds.items():
        if "fixed" in parameter_bounds:
            value = jnp.full((n_subjects,), parameter_bounds["fixed"])
            numpyro.deterministic(f"{model_name}_{parameter_name}_subj", value)
            subject_params[parameter_name] = value
            continue

        prior_spec = _resolve_parameter_priors(parameter_bounds)
        prior_mu_mean = prior_spec["prior_mu_mean"]
        prior_mu_sigma = prior_spec["prior_mu_sigma"]
        prior_beta_sigma = prior_spec["prior_beta_sigma"]
        prior_sigma_scale = prior_spec["prior_sigma_scale"]
        prior_offset_sigma = prior_spec["prior_offset_sigma"]

        mu = numpyro.sample(
            f"{model_name}_{parameter_name}_mu",
            dist.Normal(prior_mu_mean, prior_mu_sigma),
        )
        source_mus[parameter_name] = mu
        beta_current = numpyro.sample(
            f"{model_name}_{parameter_name}_beta_current_vol",
            dist.Normal(0.0, prior_beta_sigma),
        )
        beta_previous = numpyro.sample(
            f"{model_name}_{parameter_name}_beta_previous_vol",
            dist.Normal(0.0, prior_beta_sigma),
        )
        beta_interaction = numpyro.sample(
            f"{model_name}_{parameter_name}_beta_current_previous_interaction",
            dist.Normal(0.0, prior_beta_sigma),
        )
        if "fixed_subject_sigma" in parameter_bounds:
            fixed_sigma = parameter_bounds["fixed_subject_sigma"]
            subject_term = fixed_sigma * jnp.zeros(n_subjects)
        else:
            subject_sigma = numpyro.sample(
                f"{model_name}_{parameter_name}_subject_sigma",
                dist.HalfNormal(prior_sigma_scale),
            )

            with numpyro.plate(f"subjects_{model_name}_{parameter_name}", n_subjects):
                subject_offset_raw = numpyro.sample(
                    f"{model_name}_{parameter_name}_subject_offset_raw",
                    dist.Normal(0.0, prior_offset_sigma),
                )
            subject_term = subject_sigma * subject_offset_raw

        raw_value = (
            mu
            + beta_current * current_vol
            + beta_previous * previous_vol
            + beta_interaction * current_previous_interaction
            + subject_term
        )
        value = transform_param(raw_value, parameter_bounds["min"], parameter_bounds["max"])
        numpyro.deterministic(f"{model_name}_{parameter_name}_subj", value)
        subject_params[parameter_name] = value

    choice_prob = jax.vmap(
        lambda params, subject_outcomes_0, subject_outcomes_1, subject_choices: run_single_subject_scan(
            model_name,
            params,
            subject_outcomes_0,
            subject_outcomes_1,
            subject_choices,
        )
    )(
        subject_params,
        outcomes_0,
        outcomes_1,
        choices,
    )
    choice_prob = jnp.clip(choice_prob, 1e-5, 1.0 - 1e-5)

    with numpyro.plate("subjects", n_subjects):
        numpyro.sample("obs", dist.Bernoulli(probs=choice_prob).to_event(1), obs=choices)


def fit_integrated_mcmc(
    data: Dict[str, jnp.ndarray],
    model_name: str,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 1,
    target_accept_prob: float = 0.95,
    max_tree_depth: int = 10,
    seed: int = 0,
    chain_method: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    init_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> MCMC:
    """Fit the integrated model with MCMC (the NUTS sampler).

    If ``chain_method`` is None, chains run in parallel when there is a device
    for each chain, and vectorised on one device otherwise.

    Args:
        data: Integrated data set, from ``load_integrated_dataset``.
        model_name: Name of the model.
        num_warmup: Number of warmup steps.
        num_samples: Number of posterior samples per chain.
        num_chains: Number of chains.
        target_accept_prob: Target acceptance probability for NUTS.
        max_tree_depth: Maximum tree depth for NUTS.
        seed: Random seed.
        chain_method: 'parallel' or 'vectorized'.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.
        init_params: Optional starting values for the chains, with a leading
            chain axis (see ``select_pathfinder_init_params``).
        **kwargs: Extra arguments passed to the NumPyro ``MCMC`` object.

    Returns:
        The NumPyro MCMC object after sampling.
    """
    rng_key = jax.random.PRNGKey(seed)
    n_devices = jax.local_device_count()
    if chain_method is None:
        chain_method = "vectorized" if num_chains > 1 and n_devices < num_chains else "parallel"

    kernel = NUTS(
        integrated_hierarchical_model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        **kwargs,
    )
    run_kwargs: dict[str, Any] = {
        "data": data,
        "model_name": model_name,
        "config_path": config_path,
    }
    if init_params is not None:
        run_kwargs["init_params"] = init_params
    mcmc.run(rng_key, **run_kwargs)
    return mcmc


def fit_integrated_pathfinder(
    data: Dict[str, jnp.ndarray],
    model_name: str,
    seed: int = 0,
    config_path: Optional[Union[str, Path]] = None,
    num_pathfinder_samples: int = 200,
    num_posterior_samples: int = 32,
    ftol: float = 1e-4,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], Optional[float]]:
    """Approximate the posterior of the integrated model with Pathfinder.

    Pathfinder is a fast variational method that finds a high-probability region
    of the posterior. Its samples are sorted from highest to lowest probability
    (lowest to highest potential energy), and samples with a non-finite
    probability are dropped.

    Args:
        data: Integrated data set, from ``load_integrated_dataset``.
        model_name: Name of the model.
        seed: Random seed.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.
        num_pathfinder_samples: Number of samples used inside Pathfinder to
            estimate the ELBO.
        num_posterior_samples: Number of samples drawn from the Pathfinder
            approximation.
        ftol: Convergence tolerance of the optimiser.
        **kwargs: Extra arguments passed to the BlackJAX Pathfinder
            ``approximate`` function.

    Returns:
        Three values:
        - unconstrained samples, on the scale the sampler works on (used as
          MCMC starting values)
        - constrained samples, transformed to the parameters' bounds
        - the ELBO, or None if Pathfinder does not report one

    Raises:
        ValueError: If the ELBO is not finite, or if no sample has a finite
            probability.
    """
    rng_key = jax.random.PRNGKey(seed)

    param_info, potential_fn, post_process_fn, _ = initialize_model(
        rng_key,
        integrated_hierarchical_model,
        model_args=(),
        model_kwargs={
            "data": data,
            "model_name": model_name,
            "config_path": config_path,
        },
        dynamic_args=False,
    )
    init_params = param_info.z

    logdensity_fn = lambda x: -potential_fn(x)

    rng_key, subkey = jax.random.split(rng_key)
    pathfinder_state, _ = blackjax.vi.pathfinder.approximate(
        rng_key=subkey,
        logdensity_fn=logdensity_fn,
        initial_position=init_params,
        num_samples=num_pathfinder_samples,
        ftol=ftol,
        **kwargs,
    )

    rng_key, subkey = jax.random.split(rng_key)
    posterior_samples_unconstrained, _ = blackjax.vi.pathfinder.sample(
        rng_key=subkey,
        state=pathfinder_state,
        num_samples=num_posterior_samples,
    )

    post_process_vmap = jax.vmap(post_process_fn)
    constrained_samples = post_process_vmap(posterior_samples_unconstrained)

    elbo = getattr(pathfinder_state, "elbo", None)
    if elbo is not None:
        elbo = float(elbo)
        if not np.isfinite(elbo):
            raise ValueError(
                "Pathfinder returned a non-finite ELBO; refusing to use its "
                "samples as NUTS initial values."
            )

    potential_energies = np.asarray(jax.vmap(potential_fn)(posterior_samples_unconstrained))
    finite = np.isfinite(potential_energies)
    if not finite.any():
        raise ValueError(
            "Pathfinder produced no finite-potential samples; refusing to use "
            "its samples as NUTS initial values."
        )

    # Put the best finite-potential samples first so downstream init selection
    # starts NUTS at the most plausible Pathfinder draws.
    finite_indices = np.flatnonzero(finite)
    sorted_indices = finite_indices[np.argsort(potential_energies[finite_indices])]
    posterior_samples_unconstrained = {
        k: np.asarray(v)[sorted_indices]
        for k, v in posterior_samples_unconstrained.items()
    }
    constrained_samples = {
        k: np.asarray(v)[sorted_indices]
        for k, v in constrained_samples.items()
    }

    return posterior_samples_unconstrained, constrained_samples, elbo


def select_pathfinder_init_params(
    unconstrained_samples: dict[str, Any],
    num_chains: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Choose starting values for the MCMC chains from Pathfinder samples.

    Takes the first ``num_chains`` samples. If there are fewer samples than
    chains, it draws samples at random with replacement. The samples should
    already be sorted from best to worst, as ``fit_integrated_pathfinder``
    returns them.

    Args:
        unconstrained_samples: Dictionary from site name to unconstrained
            Pathfinder samples, each with the sample index as its first axis.
        num_chains: Number of MCMC chains.
        seed: Random seed, used only when samples must be drawn with
            replacement.

    Returns:
        Dictionary from site name to the chosen values, with the chain as the
        first axis. It can be passed to ``fit_integrated_mcmc`` as
        ``init_params``. It is empty if there are no samples.
    """
    keys = list(unconstrained_samples.keys())
    if not keys:
        return {}

    n_available = len(unconstrained_samples[keys[0]])

    if n_available >= num_chains:
        indices = np.arange(num_chains)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_available, size=num_chains, replace=True)

    return {k: np.asarray(v)[indices] for k, v in unconstrained_samples.items()}


def fit_integrated_pathfinder_restarts(
    data: Dict[str, jnp.ndarray],
    model_name: str,
    num_restarts: int = 1,
    seed: int = 0,
    config_path: Optional[Union[str, Path]] = None,
    num_pathfinder_samples: int = 200,
    num_posterior_samples: int = 32,
    ftol: float = 1e-4,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], Optional[float]]:
    """Run Pathfinder several times and keep the run with the highest ELBO.

    Restart ``i`` uses the random seed ``seed + i``. A restart that raises a
    ValueError is recorded and skipped. A restart with no ELBO is never chosen.

    Args:
        data: Integrated data set, from ``load_integrated_dataset``.
        model_name: Name of the model.
        num_restarts: Number of Pathfinder runs.
        seed: Random seed of the first run.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.
        num_pathfinder_samples: Number of samples used inside Pathfinder to
            estimate the ELBO.
        num_posterior_samples: Number of samples drawn from each Pathfinder
            approximation.
        ftol: Convergence tolerance of the optimiser.
        **kwargs: Extra arguments passed to ``fit_integrated_pathfinder``.

    Returns:
        The unconstrained samples, the constrained samples and the ELBO from the
        restart with the highest ELBO.

    Raises:
        ValueError: If no restart gave a usable result. The message lists the
            error from each failed restart.
    """
    best_elbo: Optional[float] = None
    best_result: Optional[tuple[dict[str, Any], dict[str, Any], Optional[float]]] = None
    failures: list[str] = []

    for restart in range(num_restarts):
        restart_seed = seed + restart
        try:
            result = fit_integrated_pathfinder(
                data,
                model_name,
                seed=restart_seed,
                config_path=config_path,
                num_pathfinder_samples=num_pathfinder_samples,
                num_posterior_samples=num_posterior_samples,
                ftol=ftol,
                **kwargs,
            )
        except ValueError as exc:
            failures.append(f"restart {restart} (seed {restart_seed}): {exc}")
            continue
        elbo = result[2]
        if elbo is not None and (best_elbo is None or elbo > best_elbo):
            best_elbo = elbo
            best_result = result

    if best_result is None:
        details = "; ".join(failures) if failures else "no successful restarts"
        raise ValueError(f"All Pathfinder restarts failed: {details}")

    return best_result


def fit_integrated_svi(
    data: Dict[str, jnp.ndarray],
    model_name: str,
    seed: int = 0,
    config_path: Optional[Union[str, Path]] = None,
    num_steps: int = 10000,
    learning_rate: float = 1e-3,
    guide_type: str = "normal",
    eval_freq: int = 100,
    patience: int = 20,
    min_delta: float = 0.001,
) -> tuple[AutoNormal, dict[str, Any], list[float]]:
    """Fit the integrated model with stochastic variational inference (SVI).

    The learning rate rises from 1e-4 to ``learning_rate`` over the first 10% of
    steps and then falls to 1e-5. Every ``eval_freq`` steps the mean loss is
    compared with the best so far, and fitting stops early after ``patience``
    checks in a row without an improvement of more than ``min_delta``.

    Args:
        data: Integrated data set, from ``load_integrated_dataset``.
        model_name: Name of the model.
        seed: Random seed.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.
        num_steps: Maximum number of SVI steps.
        learning_rate: Peak learning rate.
        guide_type: 'normal' for independent normal distributions, or 'lowrank'
            for a low-rank multivariate normal. (The value 'laplace' is not
            supported.)
        eval_freq: Check for convergence every this many steps.
        patience: Stop after this many checks without an improvement.
        min_delta: The mean loss must fall by more than this to count as an
            improvement.

    Returns:
        The fitted guide, the fitted parameters and the list of losses from
        every step.

    Raises:
        ValueError: If ``guide_type`` is not 'normal' or 'lowrank', or if the
            best loss is not finite.
    """
    rng_key = jax.random.PRNGKey(seed)

    if guide_type == "normal":
        guide_class = AutoNormal
    elif guide_type == "lowrank":
        guide_class = AutoLowRankMultivariateNormal
    else:
        raise ValueError(f"Unknown guide_type: {guide_type}")

    init_loc = numpyro.infer.init_to_uniform(radius=0.5)
    guide = guide_class(integrated_hierarchical_model, init_loc_fn=init_loc)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=learning_rate,
        warmup_steps=int(num_steps * 0.1),
        decay_steps=num_steps,
        end_value=1e-5,
    )
    optimizer_chain = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule),
    )
    optimizer = numpyro.optim.optax_to_numpyro(optimizer_chain)

    svi = SVI(integrated_hierarchical_model, guide, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(rng_key, data=data, model_name=model_name, config_path=config_path)

    losses: list[float] = []
    best_loss = float("inf")
    no_improve_count = 0

    @jax.jit
    def step_fn(state):
        """Do one SVI update step.

        Args:
            state: Current SVI state.

        Returns:
            The updated SVI state and the loss at this step.
        """
        return svi.update(state, data=data, model_name=model_name, config_path=config_path)

    @jax.jit
    def run_chunk(state):
        """Do ``eval_freq`` SVI update steps in a row.

        Args:
            state: Current SVI state.

        Returns:
            The final SVI state and the array of losses from each step.
        """

        def body(s, _):
            """Do one update step inside the loop over steps.

            Args:
                s: Current SVI state.
                _: Unused loop input.

            Returns:
                The updated state and the loss at this step.
            """
            s, l = step_fn(s)
            return s, l
        return jax.lax.scan(body, state, None, length=eval_freq)

    from tqdm import tqdm

    with tqdm(range(0, num_steps, eval_freq), unit="chunk") as pbar:
        for _ in pbar:
            svi_state, chunk_losses = run_chunk(svi_state)
            chunk_losses_np = np.array(chunk_losses)
            losses.extend(chunk_losses_np.tolist())

            avg_loss = float(np.mean(chunk_losses_np))

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            current_lr = schedule(len(losses))
            pbar.set_description(
                f"Loss: {avg_loss:.2f} | Best: {best_loss:.2f} | "
                f"Pat: {no_improve_count}/{patience} | LR: {current_lr:.2e}"
            )
            pbar.update()

            if no_improve_count >= patience:
                break

    params = svi.get_params(svi_state)
    if not np.isfinite(best_loss):
        raise ValueError(f"SVI failed: final loss is non-finite ({best_loss})")

    return guide, params, losses


def svi_to_init_params(
    data: Dict[str, jnp.ndarray],
    model_name: str,
    guide: AutoNormal,
    params: dict[str, Any],
    num_chains: int,
    seed: int = 0,
    config_path: Optional[Union[str, Path]] = None,
    jitter_scale: float = 0.01,
) -> dict[str, Any]:
    """Turn SVI results into starting values for the MCMC chains.

    Draws one posterior sample per chain from the fitted guide, adds a small
    amount of random noise, converts each sample to the unconstrained scale the
    sampler works on, and stacks the samples along a leading chain axis.

    Args:
        data: Integrated data set, from ``load_integrated_dataset``.
        model_name: Name of the model.
        guide: The fitted SVI guide.
        params: Fitted SVI parameters, from ``svi.get_params(svi_state)``.
        num_chains: Number of MCMC chains.
        seed: Random seed.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file.
        jitter_scale: Standard deviation of the Gaussian noise added to the
            samples before they are converted.

    Returns:
        Dictionary from site name to an array of unconstrained values with shape
        ``(num_chains, ...)``. It can be passed to ``fit_integrated_mcmc`` as
        ``init_params``.
    """
    import warnings

    rng_key = jax.random.PRNGKey(seed)
    rng_np = np.random.default_rng(seed + 999)

    # Draw constrained samples from guide
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svi_samples = guide.sample_posterior(rng_key, params, sample_shape=(num_chains,))

    svi_samples = {k: np.asarray(v) for k, v in svi_samples.items()}

    unconstrained_list: list[dict[str, Any]] = []
    for i in range(num_chains):
        chain_values = {k: v[i] for k, v in svi_samples.items()}

        # Add jitter (in constrained space, then let init_to_value convert)
        jittered = {
            k: v + jitter_scale * rng_np.normal(size=v.shape).astype(v.dtype)
            for k, v in chain_values.items()
        }

        param_info, _, _, _ = initialize_model(
            jax.random.PRNGKey(seed + i + 1),
            integrated_hierarchical_model,
            model_args=(),
            model_kwargs={
                "data": data,
                "model_name": model_name,
                "config_path": config_path,
            },
            init_strategy=init_to_value(values=jittered),
            dynamic_args=False,
        )
        unconstrained_list.append(param_info.z)

    # Stack along chain dimension
    stacked = {}
    for k in unconstrained_list[0]:
        stacked[k] = np.stack([p[k] for p in unconstrained_list])

    return stacked


def compute_trial_log_likelihood(
    mcmc: MCMC,
    model_name: str,
    data: Dict[str, jnp.ndarray],
    chunk_size: int = 500,
    console: Optional[Any] = None,
) -> np.ndarray:
    """Recalculate the log likelihood of every trial for every posterior draw.

    Uses the participant-level parameter samples (sites named
    ``<model_name>_<parameter>_subj``) and runs the learning model again for each
    draw. Draws are processed in chunks to limit memory use.

    Args:
        mcmc: NumPyro MCMC object after sampling.
        model_name: Name of the model.
        data: Integrated data set with ``outcomes_0``, ``outcomes_1`` and
            ``choices``, each of shape (n_subjects, n_trials).
        chunk_size: Number of posterior draws processed at a time.
        console: Optional Rich console for progress messages.

    Returns:
        Array of shape (n_chains, n_samples, n_subjects, n_trials) with the log
        probability of each observed choice.

    Raises:
        ValueError: If the samples contain no participant-level sites for this
            model.
    """
    posterior_samples = mcmc.get_samples(group_by_chain=True)
    outcomes_0 = data["outcomes_0"]
    outcomes_1 = data["outcomes_1"]
    choices = data["choices"]
    n_subjects, n_trials = choices.shape

    prefix = f"{model_name}_"
    subject_params = {}
    n_chains = None
    n_samples = None
    for key in posterior_samples:
        if not key.endswith("_subj"):
            continue
        if not key.startswith(prefix):
            continue
        param_name = key[len(prefix) : -5]
        subject_params[param_name] = posterior_samples[key]
        if n_chains is None:
            n_chains, n_samples = posterior_samples[key].shape[:2]

    if not subject_params:
        raise ValueError(
            f"No subject-level posterior samples found for model '{model_name}'"
        )

    n_draws = n_chains * n_samples
    n_chunks = (n_draws + chunk_size - 1) // chunk_size
    subject_params_flat = {
        k: v.reshape(n_draws, n_subjects)
        for k, v in subject_params.items()
    }

    def per_subject_log_lik(params, o0, o1, c):
        """Log likelihood of one participant's choices for one posterior draw.

        Args:
            params: Parameter values of this participant, by parameter name.
            o0: Outcomes of option 0 on each trial.
            o1: Outcomes of option 1 on each trial.
            c: The participant's choices on each trial.

        Returns:
            Array with the log probability of the observed choice on each trial.
            Predicted choice probabilities are clipped to between 1e-5 and
            1 - 1e-5.
        """
        cp = run_single_subject_scan(model_name, params, o0, o1, c)
        cp = jnp.clip(cp, 1e-5, 1.0 - 1e-5)
        return dist.Bernoulli(probs=cp).log_prob(c)

    subject_vmapped = jax.vmap(per_subject_log_lik, in_axes=(0, 0, 0, 0))
    draw_vmapped = jax.vmap(
        lambda params, o0, o1, c: subject_vmapped(params, o0, o1, c),
        in_axes=(0, None, None, None),
    )

    t_alloc = time.perf_counter()
    trial_log_lik = np.empty((n_draws, n_subjects, n_trials), dtype=np.float64)
    if console:
        console.print(
            f"  [dim]Allocated output array ({n_draws} × {n_subjects} × {n_trials})"
            f" in {time.perf_counter() - t_alloc:.1f}s[/dim]"
        )

    if console:
        console.print("  [dim]Compiling JIT (first chunk will be slow)...[/dim]")
    for chunk_idx, start in enumerate(range(0, n_draws, chunk_size)):
        end = min(start + chunk_size, n_draws)
        chunk_params = {
            k: v[start:end]
            for k, v in subject_params_flat.items()
        }
        t_chunk = time.perf_counter()
        chunk_ll = draw_vmapped(chunk_params, outcomes_0, outcomes_1, choices)
        trial_log_lik[start:end] = np.asarray(chunk_ll)
        if console:
            dt = time.perf_counter() - t_chunk
            prefix_label = "  [dim]JIT[/dim] " if chunk_idx == 0 else "         "
            console.print(
                f"{prefix_label}[dim]Chunk {chunk_idx:3d}/{n_chunks-1:3d}"
                f"  [draws {start:5d}-{end:5d}]  done in {dt:.1f}s[/dim]"
            )

    result = trial_log_lik.reshape(n_chains, n_samples, n_subjects, n_trials)
    if console:
        console.print(f"  [dim]Reshaped to {result.shape}[/dim]")
    return result


def integrated_samples_to_idata(
    mcmc: MCMC,
    model_name: Optional[str] = None,
    data: Optional[Dict[str, jnp.ndarray]] = None,
    console: Optional[Any] = None,
) -> az.InferenceData:
    """Convert MCMC results into an ArviZ InferenceData object.

    If both ``model_name`` and ``data`` are given, the trial-by-trial log
    likelihood is recalculated from the posterior samples and stored under the
    name ``obs``, with dimensions chain, draw, subject and trial. If they are not
    given but the samples contain a ``log_lik`` site, that site is stored
    instead.

    Args:
        mcmc: NumPyro MCMC object after sampling.
        model_name: Name of the model. Needed to recalculate the log likelihood.
        data: Integrated data set the model was fitted to. Needed to recalculate
            the log likelihood.
        console: Optional Rich console for progress messages.

    Returns:
        The InferenceData object with the posterior and, when available, the
        log likelihood.
    """
    t0 = time.perf_counter()
    idata = az.from_numpyro(mcmc, log_likelihood=False)
    if console:
        console.print(f"  [dim]az.from_numpyro done in {time.perf_counter() - t0:.1f}s[/dim]")

    if model_name is not None and data is not None:
        t1 = time.perf_counter()
        trial_log_lik = compute_trial_log_likelihood(mcmc, model_name, data, console=console)
        if console:
            console.print(f"  [dim]LL reconstruction done in {time.perf_counter() - t1:.1f}s[/dim]")
        idata.add_groups(
            log_likelihood=xr.Dataset(
                {"obs": xr.DataArray(trial_log_lik, dims=["chain", "draw", "subject", "trial"])}
            )
        )
    elif "log_lik" in mcmc.get_samples(group_by_chain=True):
        log_lik_samples = np.asarray(mcmc.get_samples(group_by_chain=True)["log_lik"])
        dims = ["chain", "draw"] + [f"dim_{i}" for i in range(log_lik_samples.ndim - 2)]
        idata.add_groups(
            log_likelihood=xr.Dataset({"obs": xr.DataArray(log_lik_samples, dims=dims)})
        )

    return idata
