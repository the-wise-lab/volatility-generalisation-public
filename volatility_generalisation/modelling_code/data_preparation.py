"""
Convert trial data into the arrays used for model fitting.

This module handles conversion between different data formats, particularly
for tasks where participants learn about complementary stimuli.

Outcome coding conventions:
    ``explosion`` / raw CSV: 1 = balloon exploded (bad/aversive), 0 = safe
    (good/rewarding).
    ``reward_safe`` (model-facing): 1 = safe/rewarding (good), 0 = explosion
    (aversive, bad).
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
from typing import Dict, Union, Optional

OUTCOME_CODING_REWARD_SAFE = "reward_safe"
"""Model-facing outcome coding: 1 = safe/rewarding, 0 = explosion/aversive."""

OUTCOME_CODING_EXPLOSION = "explosion"
"""Raw empirical outcome coding: 1 = explosion, 0 = safe."""

OUTCOME_CODING_KEY = "outcome_coding"
"""Key used for metadata in saved .npz files to record outcome coding."""


def convert_empirical_to_reward_coded(
    data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Convert explosion-coded empirical arrays to reward-coded model arrays.

    Builds a new dictionary and leaves the input unchanged. Flips
    ``outcomes_0`` and ``outcomes_1`` (``reward = 1 - explosion``) and turns
    them into float arrays. Carries ``choices`` and any other entries over
    unchanged.

    Args:
        data: Dictionary of arrays. It should hold ``outcomes_0`` and
            ``outcomes_1`` with explosion coding (1 = explosion, 0 = safe).

    Returns:
        New dictionary with reward-coded outcomes (1 = safe, 0 = explosion)
        and unchanged choices.
    """
    result = {}
    for key, value in data.items():
        if key in ("outcomes_0", "outcomes_1"):
            result[key] = 1.0 - np.asarray(value, dtype=float)
        else:
            result[key] = value
    return result


def normalize_to_reward_coded(
    data: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """Normalise loaded data to reward-coded model arrays.

    If *metadata* is provided and contains ``outcome_coding`` set to
    ``OUTCOME_CODING_REWARD_SAFE``, the data is returned unchanged.
    Otherwise the data is assumed to be explosion-coded and is converted
    with :func:`convert_empirical_to_reward_coded`. This includes the cases
    where no metadata is given and where the metadata has no
    ``outcome_coding`` entry.

    Args:
        data: Loaded data dictionary. It should hold ``outcomes_0`` and
            ``outcomes_1``.
        metadata: Optional metadata, for example from a loaded ``.npz`` file.
            Its ``outcome_coding`` entry says how the outcomes are coded.

    Returns:
        Data dictionary with reward-coded outcomes (1 = safe, 0 = explosion).
        This is the input dictionary itself when it was already reward-coded.
    """
    if metadata is not None and metadata.get(OUTCOME_CODING_KEY) == OUTCOME_CODING_REWARD_SAFE:
        return data
    return convert_empirical_to_reward_coded(data)


def prepare_complementary_data(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    choice_col: str = "choice",
    trial_col: str = "trial_num",
    outcome_col: str = "outcomes_0",
    choice_1_value: Union[int, float] = 1.0,
    choice_2_value: Union[int, float] = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Prepare data from a complementary choice task for model fitting.
    
    In complementary tasks, participants learn about two options where the
    value of one option is 1 minus the value of the other. This function
    converts single-outcome data into the two-outcome format expected by
    the models.

    Each trial number is used as a column position in the output arrays, so
    trial numbers should count from 0. The number of columns is the largest
    trial number plus 1. Trials that a participant has no row for stay 0 in
    every output array. The outcomes are copied as they are coded in the
    data frame; this function does not change the outcome coding.

    Args:
        df: Data frame with one row per participant and trial. It needs the
            columns named by ``subject_col``, ``choice_col``, ``trial_col``
            and ``outcome_col``.
        subject_col: Name of the column holding the participant identifier.
        choice_col: Name of the column holding the chosen option (for example
            1.0 or 2.0).
        trial_col: Name of the column holding the trial number, counted from 0.
        outcome_col: Name of the column holding the outcome (0 or 1) of the
            chosen option.
        choice_1_value: Value in the choice column that means the first option
            (option 0 in the output arrays). Every other value is treated as a
            choice of the second option.
        choice_2_value: Value in the choice column for the second option. The
            code does not use this argument.

    Returns:
        Dictionary with these entries:
            - outcomes_0: Array of shape (n_participants, n_trials) with the
              outcomes of option 0.
            - outcomes_1: Array of shape (n_participants, n_trials) with the
              outcomes of option 1. On each trial this is 1 minus the outcome
              of option 0.
            - choices: Array of shape (n_participants, n_trials) with 1.0 when
              option 0 was chosen and 0.0 when option 1 was chosen.
            - subject_ids: Array of shape (n_participants,) with the
              participant identifiers, in order of first appearance.
    """
    # Get unique subjects
    subjects = df[subject_col].unique()
    n_subjects = len(subjects)
    
    # Get max trial number to determine array size (ensure it's an integer)
    n_trials = int(df[trial_col].max()) + 1  # Assuming 0-indexed trials
    
    # Initialize arrays
    outcomes_0 = np.zeros((n_subjects, n_trials), dtype=np.float32)
    outcomes_1 = np.zeros((n_subjects, n_trials), dtype=np.float32)
    choices = np.zeros((n_subjects, n_trials), dtype=np.float32)
    
    # Process each subject
    for subj_idx, subject in enumerate(subjects):
        subj_data = df[df[subject_col] == subject].sort_values(trial_col)
        
        # Get the actual trials present for this subject
        trials = subj_data[trial_col].values.astype(int)
        
        # Get choices (1 = option 0, 0 = option 1)
        choice_vals = subj_data[choice_col].values
        binary_choices = (choice_vals == choice_1_value).astype(np.float32)
        
        # Get outcomes for the chosen option
        observed_outcomes = subj_data[outcome_col].values.astype(np.float32)
        
        # For each trial, assign outcomes based on choice
        for trial_idx, (trial, choice, outcome) in enumerate(
            zip(trials, binary_choices, observed_outcomes)
        ):
            if choice == 1.0:  # Chose option 0
                outcomes_0[subj_idx, trial] = outcome
                outcomes_1[subj_idx, trial] = 1.0 - outcome  # Complementary
            else:  # Chose option 1
                outcomes_1[subj_idx, trial] = outcome
                outcomes_0[subj_idx, trial] = 1.0 - outcome  # Complementary
            
            choices[subj_idx, trial] = choice
    
    return {
        "outcomes_0": outcomes_0,
        "outcomes_1": outcomes_1,
        "choices": choices,
        "subject_ids": subjects,
    }


def prepare_data_for_fitting(
    df: pd.DataFrame,
    use_jax: bool = True,
    **kwargs
) -> Dict[str, Union[np.ndarray, jnp.ndarray]]:
    """
    Prepare experimental data for model fitting.

    Calls :func:`prepare_complementary_data`. If ``use_jax`` is true, it then
    converts every array except ``subject_ids`` to a JAX array.

    Args:
        df: Data frame with one row per participant and trial.
        use_jax: If true, convert the arrays to JAX arrays. The participant
            identifiers in ``subject_ids`` stay as they are.
        **kwargs: Column names and choice values, passed on to
            :func:`prepare_complementary_data`.

    Returns:
        Dictionary with ``outcomes_0``, ``outcomes_1``, ``choices`` (each of
        shape (n_participants, n_trials)) and ``subject_ids`` (shape
        (n_participants,)).
    """
    data = prepare_complementary_data(df, **kwargs)

    if use_jax:
        # Convert numpy to JAX, but keep subject_ids as is
        jax_data = {}
        for k, v in data.items():
            if k == "subject_ids":
                jax_data[k] = v  # Keep as numpy/original
            else:
                jax_data[k] = jnp.array(v)
        return jax_data
    
    return data
