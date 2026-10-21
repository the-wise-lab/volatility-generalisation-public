"""Run a learning model for many participants at once."""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Union, Any, Optional, Set
from functools import partial

from .model_simulation import (
    run_model_scan,
    create_initial_state,
    get_model_info,
)


def run_model_scan_multi(
    model_name: str,
    model_params_multi: Dict[str, jnp.ndarray],
    trial_data: Dict[str, jnp.ndarray],
    output_vars: Optional[List[str]] = None,
    initial_states: Optional[List[Dict[str, jnp.ndarray]]] = None,
    n_subjects: Optional[int] = None,
) -> Tuple[List[Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]]:
    """
    Run a model scan for multiple subjects using JAX's vmap for efficient parallelization.

    Args:
        model_name: Name of the model to use
        model_params_multi: Dictionary of parameters where each value has shape [n_subjects, ...]
        trial_data: Dictionary of trial data (same for all subjects)
        output_vars: Optional list of output variable names to include (if None, includes all)
        initial_states: Optional list of initial state dictionaries for each subject.
            If None, every participant starts from the model's default state, and
            any ``*_init`` entries in ``model_params_multi`` are ignored.
        n_subjects: Number of subjects (inferred from model_params_multi if not provided)

    Returns:
        Tuple of (final_states, outputs) where:
            final_states: List of final state dictionaries for each subject
            outputs: Dictionary of outputs with shape [n_subjects, n_trials, ...]
    """
    # Determine number of subjects from parameters if not specified
    if n_subjects is None:
        # Take first parameter and get its first dimension
        first_param = list(model_params_multi.values())[0]
        n_subjects = first_param.shape[0]

    # Create initial states for all subjects if not provided
    if initial_states is None:
        initial_states = [
            create_initial_state(model_name, {}) for _ in range(n_subjects)
        ]

    # Convert list of dictionaries to dictionary of arrays for JAX compatibility
    model_info = get_model_info(model_name)
    state_vars = model_info.state_vars

    # Create a dictionary where each key is a state variable and each value is an array of initial values
    initial_states_arrays = {}
    for var in state_vars:
        initial_states_arrays[var] = jnp.array(
            [state[var] for state in initial_states]
        )

    # Determine which output variables to include
    if output_vars is None:
        output_vars = model_info.output_vars

    # Define a function to run model_scan for a single subject
    def run_for_single_subject(subject_idx):
        """Run the model for one participant.

        Takes that participant's parameters, starting state and (if the trial
        data has one row per participant) trial data from the arrays defined
        in the enclosing function.

        Args:
            subject_idx: Index of the participant, counted from zero.

        Returns:
            Tuple of (final_state, outputs) for this participant, as returned
            by ``run_model_scan``.
        """
        # Extract parameters for this subject
        subject_params = {
            k: v[subject_idx] for k, v in model_params_multi.items()
        }

        # Construct initial state dictionary for this subject
        subject_init_state = {
            var: initial_states_arrays[var][subject_idx] for var in state_vars
        }

        # Slice trial data if it has an extra dimension for subjects
        # We assume that if trial_data has 2 dimensions, the first one is subjects
        subject_trial_data = {
            k: v[subject_idx] if v.ndim == 2 else v 
            for k, v in trial_data.items()
        }

        # Run the model for this subject
        final_state, outputs = run_model_scan(
            model_name=model_name,
            model_params=subject_params,
            trial_data=subject_trial_data,
            initial_state=subject_init_state,
        )

        return final_state, outputs

    # Use vmap to vectorize over subjects
    run_vmap = jax.vmap(run_for_single_subject, in_axes=0)

    # Run the vectorized function
    subject_indices = jnp.arange(n_subjects)
    final_states, all_outputs = run_vmap(subject_indices)

    # Convert to more usable format
    final_states_list = [
        {k: v[i] for k, v in final_states.items()} for i in range(n_subjects)
    ]

    # Filter outputs to only include requested variables
    filtered_outputs = {var: all_outputs[var] for var in output_vars if var in all_outputs}

    return final_states_list, filtered_outputs
