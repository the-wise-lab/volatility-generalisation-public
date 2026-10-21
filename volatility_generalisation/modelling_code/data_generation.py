"""
Functions for generating model data using parameter bounds.
"""

import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import toml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from numpy.random import RandomState
from .batch_utilities import run_model_scan_multi
from .data_preparation import OUTCOME_CODING_KEY, OUTCOME_CODING_REWARD_SAFE
import numpyro.distributions as dist


DEFAULT_MODEL_BOUNDS_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "model_bounds.toml"
)


def load_parameter_bounds(
    model_name: str, config_path: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Load parameter bounds for a specific model from the config file.

    The bounds start from the ``common`` section of the file. For the model
    named exactly ``alb``, the ``temperature`` bounds are removed, because that
    model does not use a softmax choice rule. For any model whose name contains
    ``alb``, the ``alb`` ``common`` bounds are added. Finally the bounds listed
    under the model's own section, if it has one, are added and replace any
    bounds with the same parameter name.

    Args:
        model_name: Name of the model. It must match a section name in the
            config file exactly, unless the name contains ``alb``.
        config_path: Optional path to the config file. If None, uses the
            project-root config/model_bounds.toml

    Returns:
        Dictionary mapping each parameter name to its bounds. The bounds are
        a dictionary with ``min`` and ``max`` values, or a ``fixed`` value. They
        may also hold prior settings such as ``prior_mu_mean`` and
        ``prior_mu_sigma``, if the config file gives them.

    Raises:
        ValueError: If the config file has no section for the model and the
            model name does not contain ``alb``.
    """
    if config_path is None:
        config_path = DEFAULT_MODEL_BOUNDS_CONFIG
    else:
        config_path = Path(config_path)

    bounds = toml.load(config_path)

    # Get model-specific bounds
    model_bounds = {}

    # Add common parameters
    model_bounds.update(bounds["common"])

    # Remove temperature for ALB model as it doesn't use softmax
    if model_name == "alb" and "temperature" in model_bounds:
        del model_bounds["temperature"]

    # Add ALB common parameters if it's an ALB model
    if model_name.startswith("alb") or "alb" in model_name:
        model_bounds.update(bounds["alb"]["common"])

    # Add model-specific parameters
    if model_name in bounds:
        model_specific = {
            k: v for k, v in bounds[model_name].items() if k != "common"
        }
        model_bounds.update(model_specific)
    elif not (model_name.startswith("alb") or "alb" in model_name):
        raise ValueError(f"Unknown model bounds for {model_name}")

    return model_bounds


def settings_file_record(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Describe the parameter bounds file a fit used, for its saved details.

    The record holds the file's path and the SHA-256 hash of its contents, so a
    saved result can be matched to the exact settings that produced it, even if
    the file is later edited. Call it before fitting, so that the hash is of the
    file the fit read, and a missing file stops the run before any fitting.

    Args:
        config_path: Path to the parameter bounds file. If None, uses the same
            default file as ``load_parameter_bounds``.

    Returns:
        Dictionary with ``config_path`` (the path as text) and
        ``config_sha256`` (the hash of the file's bytes, as hexadecimal text).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = DEFAULT_MODEL_BOUNDS_CONFIG if config_path is None else Path(config_path)
    return {
        "config_path": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def generate_random_parameters(
    bounds: Dict[str, Dict[str, float]], 
    n_sets: int = 1,
    rng: Optional[RandomState] = None,
    distribution: str = "beta",
    beta_shape: float = 1.0
) -> Dict[str, jnp.ndarray]:
    """
    Generate random parameters within the specified bounds.

    A parameter with a ``fixed`` bound gets that value in every set. Other
    parameters are drawn between their ``min`` and ``max``, either uniformly
    or from a symmetric Beta distribution rescaled to that range.

    Args:
        bounds: Dictionary mapping each parameter name to its bounds. Each
            entry has either a ``fixed`` value or both ``min`` and ``max``.
        n_sets: Number of parameter sets to generate
        rng: Random number generator. If None, creates a new one without a seed.
        distribution: 'uniform' or 'beta'. Defaults to 'beta'.
        beta_shape: Shape of the symmetric Beta distribution (both shape
            parameters equal this value). Defaults to 1.0, which is uniform.
            Larger values give a bell-like shape.

    Returns:
        Dictionary mapping each parameter name to an array of shape (n_sets,).

    Raises:
        ValueError: If a parameter has no ``fixed`` value and lacks a ``min``
            or ``max`` bound, or if ``distribution`` is not 'uniform' or
            'beta'.
    """

    if rng is None:
        rng = RandomState()
        
    params = {}
    for param_name, param_bounds in bounds.items():
        if "fixed" in param_bounds:
            values = jnp.full(n_sets, param_bounds["fixed"])
            params[param_name] = values
            continue

        if not "min" in param_bounds:
            raise ValueError(
                f"Parameter '{param_name}' does not have 'min' bound."
            )
        if not "max" in param_bounds:
            raise ValueError(
                f"Parameter '{param_name}' does not have 'max' bound."
            )
            
        min_val = param_bounds["min"]
        max_val = param_bounds["max"]
        
        if distribution == "uniform":
            values = rng.uniform(min_val, max_val, size=n_sets)
        elif distribution == "beta":
            # Generate beta distributed values in [0, 1]
            # Using symmetric beta (alpha=beta) for centered gaussian-like shape
            raw_values = rng.beta(beta_shape, beta_shape, size=n_sets)
                
            # Scale to [min, max]
            values = min_val + (max_val - min_val) * raw_values
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        params[param_name] = jnp.array(values)


    return params


def generate_synthetic_data_single_model(
    model_name: str,
    n_subjects: int,
    n_trials: int,
    external_outcomes: Dict[str, jnp.ndarray],
    rng_seed: int = 42,
    config_path: Optional[Union[str, Path]] = None,
    distribution: str = "beta",
    beta_shape: float = 1.0,
    subject_params: Optional[Dict[str, jnp.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Generate synthetic data for a single model.

    Every simulated participant sees the same outcome sequence, taken from
    ``external_outcomes``. Each participant gets a set of parameters, either
    random ones within the bounds in the config file or the ones in
    ``subject_params``. The model then gives the probability of choosing
    option 0 on each trial, and the choices are drawn from those probabilities.

    Args:
        model_name: Name of the model.
        n_subjects: Number of simulated participants.
        n_trials: Number of trials per participant.
        external_outcomes: Dictionary with one-dimensional arrays
            ``outcomes_0`` and ``outcomes_1``, reward-coded (1 = safe), for
            example from ``load_prepared_schedule``. The first ``n_trials``
            values are used.
        rng_seed: Seed for the parameters and the choices.
        config_path: Optional path to the parameter bounds file. If None, uses
            the default file. Used only when ``subject_params`` is None.
        distribution: 'uniform' or 'beta', for drawing random parameters. Used
            only when ``subject_params`` is None.
        beta_shape: Shape of the Beta distribution for random parameters. Used
            only when ``subject_params`` is None.
        subject_params: Optional parameters for each participant, as arrays of
            shape (n_subjects,). If None, random parameters are drawn.

    Returns:
        Dictionary with these entries:
        - data_for_fit: dictionary with ``outcomes_0``, ``outcomes_1`` and
          ``choices``, each of shape (n_subjects, n_trials), and
          ``outcome_coding`` set to reward coding, so that the fitting code
          uses the outcomes as they are. Choices are 1 for option 0 and 0 for
          option 1.
        - true_params: dictionary of the parameters used for each participant
        - true_precisions: array of zeros with shape (n_subjects,). These are
          placeholders, because the choice rule has no precision value.

    Raises:
        ValueError: If ``external_outcomes`` has fewer than ``n_trials`` trials.
    """

    # Split the seed into four keys and draw the choices with the last one.
    # The split is kept fixed so that a given seed always reproduces the same
    # choices.
    choice_key = jax.random.split(jax.random.PRNGKey(rng_seed), 4)[3]

    # Every participant gets the same sequence: the first n_trials trials.
    single_outcomes_0 = external_outcomes["outcomes_0"]
    single_outcomes_1 = external_outcomes["outcomes_1"]
    if len(single_outcomes_0) < n_trials:
        raise ValueError(
            f"External outcomes length ({len(single_outcomes_0)}) is less than n_trials ({n_trials})"
        )
    outcomes_0 = jnp.tile(single_outcomes_0[:n_trials], (n_subjects, 1))
    outcomes_1 = jnp.tile(single_outcomes_1[:n_trials], (n_subjects, 1))

    if subject_params is None:
        bounds = load_parameter_bounds(model_name, config_path)
        rng = np.random.RandomState(rng_seed)
        params = generate_random_parameters(
            bounds,
            n_subjects,
            rng,
            distribution=distribution,
            beta_shape=beta_shape,
        )
    else:
        params = {
            name: jnp.array(values, dtype=jnp.float64)
            for name, values in subject_params.items()
        }

    # Prepare trial data
    batch_trial_data = {
        "outcomes_0": outcomes_0,
        "outcomes_1": outcomes_1
    }

    # Run model simulation
    _, outputs = run_model_scan_multi(
        model_name,
        params,
        batch_trial_data,
        output_vars=["prev_pos"],
        n_subjects=n_subjects
    )

    # outputs["prev_pos"] contains the probability of choosing Option 0
    probs_choose_0 = outputs["prev_pos"] # Shape (n_sub, n_trials)

    # Generate binary choices using 1 = option 0, 0 = option 1.
    choices = dist.Bernoulli(probs=probs_choose_0).sample(choice_key).astype(jnp.float64)
    
    # Mock precisions (not used for Bernoulli, but kept for interface consistency if needed)
    subject_precisions = jnp.zeros(n_subjects) 
    
    return {
        "data_for_fit": {
            "outcomes_0": outcomes_0,
            "outcomes_1": outcomes_1,
            "choices": choices,
            OUTCOME_CODING_KEY: OUTCOME_CODING_REWARD_SAFE,
        },
        "true_params": params,
        "true_precisions": subject_precisions
    }
