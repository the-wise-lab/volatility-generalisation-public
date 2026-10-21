"""
Run the learning models across trials with JAX.

This module provides functions to run learning models across multiple trials
efficiently using JAX's scan operator.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Any,
    Union,
    Optional,
    NamedTuple,
)
import numpy as np
from functools import partial

from behavioural_modelling.learning.beta_models import beta_mean_var, leaky_beta_update
from behavioural_modelling.decision_rules import softmax








def dual_lr_single_value(
    outcome_option0: jax.Array,
    trial_number: int,
    outcome_option1: jax.Array,
    value_0: jax.Array,
    value_1: jax.Array,
    prev_pos: jax.Array,
    learning_rate_pos: float,
    learning_rate_neg: float,
    temperature: float,
    **kwargs: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the dual learning-rate model after one trial.

    The model tracks one value, the expected outcome of option 0. The expected
    outcome of option 1 is always 1 minus that value. The choice probability is
    a softmax of the two values, calculated before the update. The value moves
    towards the outcome of option 0 by the positive learning rate after a
    better-than-expected outcome and by the negative learning rate after a
    worse-than-expected outcome. It does not change if the outcome matches the
    expectation.

    Args:
        outcome_option0: Outcome of option 0 on this trial.
        trial_number: Index of the trial. Not used.
        outcome_option1: Outcome of option 1 on this trial. Not used.
        value_0: Current expected outcome of option 0.
        value_1: Current expected outcome of option 1. Not used.
        prev_pos: Value of the previous-probability state. Not used.
        learning_rate_pos: Learning rate after a positive prediction error.
        learning_rate_neg: Learning rate after a negative prediction error.
        temperature: Softmax temperature. The two expected outcomes are divided
            by it, so a higher temperature gives more random choices.
        **kwargs: Extra trial inputs (such as the choice). Not used.

    Returns:
        Four values:
        - probability of choosing option 0
        - updated expected outcome of option 0
        - updated expected outcome of option 1
        - prediction error (outcome of option 0 minus the old expected outcome)
    """
    del trial_number, outcome_option1, value_1, prev_pos, kwargs

    tracked_value = value_0
    values = jnp.array([tracked_value, 1.0 - tracked_value])
    choice_probs = softmax(values[None, :], temperature)[0]

    prediction_error = outcome_option0 - tracked_value
    alpha_t = (
        (learning_rate_pos * (prediction_error > 0))
        + (learning_rate_neg * (prediction_error < 0))
    )
    new_value_0 = tracked_value + alpha_t * prediction_error
    new_value_1 = 1.0 - new_value_0

    return (
        choice_probs[0].astype(values.dtype),
        new_value_0,
        new_value_1,
        prediction_error,
    )








def rescorla_wagner_single_value(
    outcome_option0: jax.Array,
    trial_number: int,
    outcome_option1: jax.Array,
    value_0: jax.Array,
    value_1: jax.Array,
    prev_pos: jax.Array,
    learning_rate: float,
    temperature: float,
    **kwargs: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the Rescorla-Wagner model after one trial.

    The model tracks one value, the expected outcome of option 0. The expected
    outcome of option 1 is always 1 minus that value. The choice probability is
    a softmax of the two values, calculated before the update. The value moves
    towards the outcome of option 0 by the learning rate times the prediction
    error.

    Args:
        outcome_option0: Outcome of option 0 on this trial.
        trial_number: Index of the trial. Not used.
        outcome_option1: Outcome of option 1 on this trial. Not used.
        value_0: Current expected outcome of option 0.
        value_1: Current expected outcome of option 1. Not used.
        prev_pos: Value of the previous-probability state. Not used.
        learning_rate: Learning rate.
        temperature: Softmax temperature. The two expected outcomes are divided
            by it, so a higher temperature gives more random choices.
        **kwargs: Extra trial inputs (such as the choice). Not used.

    Returns:
        Four values:
        - probability of choosing option 0
        - updated expected outcome of option 0
        - updated expected outcome of option 1
        - prediction error (outcome of option 0 minus the old expected outcome)
    """
    del trial_number, outcome_option1, value_1, prev_pos, kwargs

    values = jnp.array([value_0, 1.0 - value_0])
    choice_probs = softmax(values[None, :], temperature)[0]

    prediction_error = outcome_option0 - value_0
    new_value_0 = value_0 + learning_rate * prediction_error
    new_value_1 = 1.0 - new_value_0

    return (
        choice_probs[0].astype(values.dtype),
        new_value_0,
        new_value_1,
        prediction_error,
    )





def pearce_hall_single_value(
    outcome_option0: jax.Array,
    trial_number: int,
    outcome_option1: jax.Array,
    value_0: jax.Array,
    value_1: jax.Array,
    assoc_0: jax.Array,
    prev_pos: jax.Array,
    kappa: float,
    gamma: float,
    temperature: float,
    **kwargs: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the Pearce-Hall model after one trial.

    The model tracks the expected outcome of option 0 and an associability
    (how much attention the outcomes of option 0 currently get). The expected
    outcome of option 1 is always 1 minus the value of option 0. The choice
    probability is a softmax of the two values, calculated before the update.
    The value moves towards the outcome of option 0 by ``kappa`` times the
    associability times the prediction error. The associability moves towards
    the size of the prediction error at rate ``gamma``.

    Args:
        outcome_option0: Outcome of option 0 on this trial.
        trial_number: Index of the trial. Not used.
        outcome_option1: Outcome of option 1 on this trial. Not used.
        value_0: Current expected outcome of option 0.
        value_1: Current expected outcome of option 1. Not used.
        assoc_0: Current associability of option 0.
        prev_pos: Value of the previous-probability state. Not used.
        kappa: Scaling of the value update.
        gamma: Rate at which the associability follows the size of the
            prediction error.
        temperature: Softmax temperature. The two expected outcomes are divided
            by it, so a higher temperature gives more random choices.
        **kwargs: Extra trial inputs (such as the choice). Not used.

    Returns:
        Five values (although the return annotation says six):
        - probability of choosing option 0
        - updated expected outcome of option 0
        - updated expected outcome of option 1
        - updated associability of option 0
        - prediction error (outcome of option 0 minus the old expected outcome)
    """
    del trial_number, outcome_option1, value_1, prev_pos, kwargs

    values = jnp.array([value_0, 1.0 - value_0])
    choice_probs = softmax(values[None, :], temperature)[0]

    prediction_error = outcome_option0 - value_0
    new_value_0 = value_0 + kappa * assoc_0 * prediction_error
    new_assoc_0 = assoc_0 + gamma * (jnp.abs(prediction_error) - assoc_0)
    new_value_1 = 1.0 - new_value_0

    return (
        choice_probs[0].astype(values.dtype),
        new_value_0,
        new_value_1,
        new_assoc_0,
        prediction_error,
    )











def leaky_beta_single_value(
    outcome_option0: jax.Array,
    trial_number: int,
    outcome_option1: jax.Array,
    beta_a_0: jax.Array,
    beta_b_0: jax.Array,
    prev_pos: jax.Array,
    tau: float,
    decay: float,
    temperature: float,
    **kwargs: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the leaky beta model after one trial, with one shared learning rate.

    The model tracks a beta distribution for option 0, described by its two
    counts (``beta_a_0`` and ``beta_b_0``). The expected outcome of option 0 is
    the mean of that distribution, and the expected outcome of option 1 is 1
    minus that mean. The choice probability is a softmax of the two expected
    outcomes, calculated before the update. Both counts are then multiplied
    by ``decay`` (relative to a baseline of 1, so they shrink towards 1). The
    first count then rises by ``tau`` if the outcome of option 0 is above 0,
    and the second count rises by ``tau`` if it is 0 or below. The counts of
    option 1 are the counts of option 0 swapped.

    Args:
        outcome_option0: Outcome of option 0 on this trial.
        trial_number: Index of the trial. Not used.
        outcome_option1: Outcome of option 1 on this trial. Not used.
        beta_a_0: Current first count of the beta distribution for option 0.
        beta_b_0: Current second count of the beta distribution for option 0.
        prev_pos: Value of the previous-probability state. Not used.
        tau: Amount added to a count, used for both positive outcomes and
            outcomes of 0.
        decay: Factor by which the counts (relative to 1) are multiplied on
            each trial.
        temperature: Softmax temperature. The two expected outcomes are divided
            by it, so a higher temperature gives more random choices.
        **kwargs: Extra trial inputs (such as the choice). Not used.

    Returns:
        Nine values:
        - probability of choosing option 0
        - updated first and second counts for option 0
        - updated first and second counts for option 1 (the counts of option 0
          swapped)
        - updated expected outcome of option 0 and of option 1
        - updated variance for option 0 and for option 1 (the same value)
    """
    del trial_number, outcome_option1, prev_pos, kwargs

    beta_params = jnp.array([[beta_a_0, beta_b_0]])
    means, variances = beta_mean_var(beta_params)
    choice_probs = softmax(jnp.array([[means[0], 1.0 - means[0]]]), temperature)[0]

    new_beta_params = leaky_beta_update(
        estimate=beta_params,
        choices=jnp.array([1]),
        outcome=outcome_option0,
        tau_p=tau,
        tau_n=tau,
        decay=decay,
    )

    new_means, new_variances = beta_mean_var(new_beta_params)
    new_beta_a_0 = new_beta_params[0, 0]
    new_beta_b_0 = new_beta_params[0, 1]
    new_beta_a_1 = new_beta_b_0
    new_beta_b_1 = new_beta_a_0

    return (
        choice_probs[0].astype(means.dtype),
        new_beta_a_0,
        new_beta_b_0,
        new_beta_a_1,
        new_beta_b_1,
        new_means[0],
        1.0 - new_means[0],
        new_variances[0],
        new_variances[0],
    )








def leaky_beta_asymmetric_single_value(
    outcome_option0: jax.Array,
    trial_number: int,
    outcome_option1: jax.Array,
    beta_a_0: jax.Array,
    beta_b_0: jax.Array,
    prev_pos: jax.Array,
    tau_pos: float,
    tau_neg: float,
    decay: float,
    temperature: float,
    **kwargs: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the leaky beta model after one trial, with separate learning rates.

    The model tracks a beta distribution for option 0, described by its two
    counts (``beta_a_0`` and ``beta_b_0``). The expected outcome of option 0 is
    the mean of that distribution, and the expected outcome of option 1 is 1
    minus that mean. The choice probability is a softmax of the two expected
    outcomes, calculated before the update. Both counts are then multiplied
    by ``decay`` (relative to a baseline of 1, so they shrink towards 1). The
    first count then rises by ``tau_pos`` if the outcome of option 0 is above
    0, and the second count rises by ``tau_neg`` if it is 0 or below. The
    counts of option 1 are the counts of option 0 swapped.

    Args:
        outcome_option0: Outcome of option 0 on this trial.
        trial_number: Index of the trial. Not used.
        outcome_option1: Outcome of option 1 on this trial. Not used.
        beta_a_0: Current first count of the beta distribution for option 0.
        beta_b_0: Current second count of the beta distribution for option 0.
        prev_pos: Value of the previous-probability state. Not used.
        tau_pos: Amount added to the first count after a positive outcome.
        tau_neg: Amount added to the second count after an outcome of 0 or
            below.
        decay: Factor by which the counts (relative to 1) are multiplied on
            each trial.
        temperature: Softmax temperature. The two expected outcomes are divided
            by it, so a higher temperature gives more random choices.
        **kwargs: Extra trial inputs (such as the choice). Not used.

    Returns:
        Nine values:
        - probability of choosing option 0
        - updated first and second counts for option 0
        - updated first and second counts for option 1 (the counts of option 0
          swapped)
        - updated expected outcome of option 0 and of option 1
        - updated variance for option 0 and for option 1 (the same value)
    """
    del trial_number, outcome_option1, prev_pos, kwargs

    beta_params = jnp.array([[beta_a_0, beta_b_0]])
    means, variances = beta_mean_var(beta_params)
    choice_probs = softmax(jnp.array([[means[0], 1.0 - means[0]]]), temperature)[0]

    new_beta_params = leaky_beta_update(
        estimate=beta_params,
        choices=jnp.array([1]),
        outcome=outcome_option0,
        tau_p=tau_pos,
        tau_n=tau_neg,
        decay=decay,
    )

    new_means, new_variances = beta_mean_var(new_beta_params)
    new_beta_a_0 = new_beta_params[0, 0]
    new_beta_b_0 = new_beta_params[0, 1]
    new_beta_a_1 = new_beta_b_0
    new_beta_b_1 = new_beta_a_0

    return (
        choice_probs[0].astype(means.dtype),
        new_beta_a_0,
        new_beta_b_0,
        new_beta_a_1,
        new_beta_b_1,
        new_means[0],
        1.0 - new_means[0],
        new_variances[0],
        new_variances[0],
    )





# Model metadata to standardize output handling
class ModelInfo(NamedTuple):
    """Description of one learning model.

    Each model function returns a tuple. The choice probability comes first,
    followed by the updated state values and other outputs. The index lists say
    where to find each named value in that tuple.

    Attributes:
        func: The function that updates the model after one trial.
        state_vars: Names of the values carried from one trial to the next.
        state_indices: Position of each state value in the returned tuple.
        output_vars: Names of the values recorded on every trial.
        output_indices: Position of each recorded value in the returned tuple.
    """

    func: Callable  # Model function
    state_vars: List[str]  # Names of state variables
    state_indices: List[int]  # Indices of state variables in output tuple
    output_vars: List[str]  # Names of output variables to track
    output_indices: List[int]  # Indices of output variables in output tuple

# Dictionary mapping model names to their metadata
# Model returns: (value/choice_prob, state_vars..., other_outputs...)
# Every model uses single-value outputs.
MODEL_INFO = {
    "rescorla_wagner_single_value": ModelInfo(
        func=rescorla_wagner_single_value,
        state_vars=["value_0", "value_1", "prev_pos"],
        state_indices=[1, 2, 0],
        output_vars=["prev_pos", "prediction_error"],
        output_indices=[0, 3],
    ),
    "dual_lr_single_value": ModelInfo(
        func=dual_lr_single_value,
        state_vars=["value_0", "value_1", "prev_pos"],
        state_indices=[1, 2, 0],
        output_vars=["prev_pos", "prediction_error"],
        output_indices=[0, 3],
    ),
    "pearce_hall_single_value": ModelInfo(
        func=pearce_hall_single_value,
        state_vars=["value_0", "value_1", "assoc_0", "prev_pos"],
        state_indices=[1, 2, 3, 0],
        output_vars=["prev_pos", "value_0", "value_1", "assoc_0", "pe"],
        output_indices=[0, 1, 2, 3, 4],
    ),
    "leaky_beta_single_value": ModelInfo(
        func=leaky_beta_single_value,
        state_vars=["beta_a_0", "beta_b_0", "prev_pos"],
        state_indices=[1, 2, 0],
        output_vars=["prev_pos", "value_0", "value_1", "variance_0", "variance_1"],
        output_indices=[0, 5, 6, 7, 8],
    ),
    "leaky_beta_asymmetric_single_value": ModelInfo(
        func=leaky_beta_asymmetric_single_value,
        state_vars=["beta_a_0", "beta_b_0", "prev_pos"],
        state_indices=[1, 2, 0],
        output_vars=["prev_pos", "value_0", "value_1", "variance_0", "variance_1"],
        output_indices=[0, 5, 6, 7, 8],
    ),
}


def get_model_info(model_name: str) -> ModelInfo:
    """
    Get the description of a model by name.

    Args:
        model_name: Name of the model. Upper-case letters are converted to
            lower case.

    Returns:
        The ``ModelInfo`` entry for the model.

    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    if model_name not in MODEL_INFO:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_INFO.keys())}"
        )

    return MODEL_INFO[model_name]


def create_initial_state(
    model_name: str, params: Dict[str, Union[float, np.ndarray, jnp.ndarray]]
) -> Dict[str, jnp.ndarray]:
    """
    Create the starting values of a model's state for a simulation.

    Each state value is read from the parameter named ``<state>_init`` (for
    example ``value_0_init``). If that parameter is missing, the value starts at
    0.5 for values, associabilities and the previous probability, and at 1.0 for
    the beta counts.

    Args:
        model_name: Name of the model
        params: Dictionary of model parameters, which may include ``*_init``
            entries

    Returns:
        Dictionary from state name to its starting value

    Raises:
        ValueError: If the model's state names are not ones this function
            knows about.
    """
    model_info = get_model_info(model_name)

    if model_info.state_vars == ["value_0", "value_1", "prev_pos"]:
        # For Rescorla-Wagner family models
        return {
            "value_0": jnp.array(params.get("value_0_init", 0.5)),
            "value_1": jnp.array(params.get("value_1_init", 0.5)),
            "prev_pos": jnp.array(params.get("prev_pos_init", 0.5)),
        }
    elif model_info.state_vars == ["value_0", "value_1", "assoc_0", "prev_pos"]:
        # For Pearce-Hall single value
        return {
            "value_0": jnp.array(params.get("value_0_init", 0.5)),
            "value_1": jnp.array(params.get("value_1_init", 0.5)),
            "assoc_0": jnp.array(params.get("assoc_0_init", 0.5)),
            "prev_pos": jnp.array(params.get("prev_pos_init", 0.5)),
        }
    elif model_info.state_vars == ["value_0", "value_1", "assoc_0", "assoc_1", "prev_pos"]:
        # For Pearce-Hall
        return {
            "value_0": jnp.array(params.get("value_0_init", 0.5)),
            "value_1": jnp.array(params.get("value_1_init", 0.5)),
            "assoc_0": jnp.array(params.get("assoc_0_init", 0.5)),
            "assoc_1": jnp.array(params.get("assoc_1_init", 0.5)),
            "prev_pos": jnp.array(params.get("prev_pos_init", 0.5)),
        }
    elif model_info.state_vars == ["beta_a_0", "beta_b_0", "beta_a_1", "beta_b_1", "prev_pos"]:
        # For leaky beta models
        return {
            "beta_a_0": jnp.array(params.get("beta_a_0_init", 1.0)),
            "beta_b_0": jnp.array(params.get("beta_b_0_init", 1.0)),
            "beta_a_1": jnp.array(params.get("beta_a_1_init", 1.0)),
            "beta_b_1": jnp.array(params.get("beta_b_1_init", 1.0)),
            "prev_pos": jnp.array(params.get("prev_pos_init", 0.5)),
        }
    elif model_info.state_vars == ["beta_a_0", "beta_b_0", "prev_pos"]:
        # For single-value leaky beta models
        return {
            "beta_a_0": jnp.array(params.get("beta_a_0_init", 1.0)),
            "beta_b_0": jnp.array(params.get("beta_b_0_init", 1.0)),
            "prev_pos": jnp.array(params.get("prev_pos_init", 0.5)),
        }
    else:
        raise ValueError(f"Unknown state variables for model: {model_name}")


def create_scan_fn(
    model_name: str,
    model_params: Dict[str, Any],
    model_info: Optional[ModelInfo] = None,
):
    """
    Create the function that runs one trial of a model inside a JAX scan.

    Args:
        model_name: Name of the model
        model_params: Dictionary of model parameters
        model_info: Optional model description. If None, it is looked up from
            model_name.

    Returns:
        A function that takes the current state and one trial's inputs and
        returns the new state and that trial's recorded outputs. It can be
        passed to ``lax.scan``.
    """
    if model_info is None:
        model_info = get_model_info(model_name)

    # Define a function to update the model state and record outputs for each trial.
    def trial_step(state, trial_inputs):
        """Run the model on one trial.

        Args:
            state: Dictionary of the model's current state values.
            trial_inputs: Tuple of the trial index, the outcome of option 0, the
                outcome of option 1 and the choice on this trial.

        Returns:
            The dictionary of updated state values and the dictionary of values
            recorded on this trial.
        """
        # Unpack trial inputs
        trial_idx, outcome0, outcome1, choice = trial_inputs
        extra_args = {}

        # Prepare current state variables for the model.
        current_model_state = {
            name: state[name] for name in model_info.state_vars
        }

        # Call the model function with the trial data, current state, model parameters and extra args.
        results = model_info.func(
            outcome_option0=outcome0,
            trial_number=trial_idx,
            outcome_option1=outcome1,
            choice=choice,
            **current_model_state,
            **model_params,
            **extra_args,
        )

        # Update the state for the next trial.
        new_state = {
            name: jnp.asarray(results[idx], dtype=state[name].dtype)
            for name, idx in zip(
                model_info.state_vars, model_info.state_indices
            )
        }

        # Record the outputs we want to track across trials.
        trial_outputs = {
            name: results[idx]
            for name, idx in zip(
                model_info.output_vars, model_info.output_indices
            )
        }

        return new_state, trial_outputs

    return trial_step

@partial(jax.jit, static_argnames=["model_name"])
def run_model_scan(
    model_name: str,
    model_params: Dict[str, jax.Array],
    trial_data: Dict[str, jax.Array],
    initial_state: Optional[Dict[str, jax.Array]] = None,
) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    """
    Run any model across multiple trials using JAX scan.

    Args:
        model_name: Name of the model to use. Valid model names are:
            - "rescorla_wagner_single_value": Basic Rescorla-Wagner model with a single value stream
            - "dual_lr_single_value": Dual learning-rate model with a single value stream
            - "pearce_hall_single_value": Pearce-Hall model with a single value stream
            - "leaky_beta_single_value": Leaky beta model with one learning rate
            - "leaky_beta_asymmetric_single_value": Leaky beta model with
              separate learning rates for positive and negative outcomes
        model_params: Dictionary of model parameters. Entries whose names
            contain "_init" set the starting state and are not passed to the
            model.
        trial_data: Dictionary containing trial data. It needs ``outcomes_0``
            and ``outcomes_1``. Choice-dependent models expect a `choices`
            array where 1 means option 0 was chosen and 0 means option 1 was
            chosen. If ``choices`` is missing, it is filled with zeros.
        initial_state: Optional initial model state. If None, will use defaults.

    Returns:
        Tuple containing:
        - Final state
        - Dictionary of intermediate values for all trials
    """
    # Get model metadata
    model_info = get_model_info(model_name)

    # Create initial state if not provided
    if initial_state is None:
        initial_state = create_initial_state(model_name, model_params)

    # Remove any params containing "_init"
    model_params = {
        k: v for k, v in model_params.items() if "_init" not in k
    }

    # Extract trial data
    outcomes_0 = trial_data["outcomes_0"]
    outcomes_1 = trial_data["outcomes_1"]
    choices = trial_data.get("choices", jnp.zeros_like(outcomes_0))
    trial_indices = jnp.arange(len(outcomes_0))

    # Create scan function
    scan_fn = create_scan_fn(model_name, model_params, model_info)

    # Prepare input data based on model requirements
    xs = (trial_indices, outcomes_0, outcomes_1, choices)

    # Run the scan
    final_state, all_outputs = lax.scan(scan_fn, initial_state, xs)

    return final_state, all_outputs
