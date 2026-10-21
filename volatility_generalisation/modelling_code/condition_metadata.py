"""Shared empirical condition metadata and context coding."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


CONDITION_METADATA: Dict[str, Dict[str, object]] = {
    "VolatileV1": {
        "block_type": "Volatile",
        "version": "V1",
        "block_number": 1,
        "previous_block_type": None,
    },
    "StableV1": {
        "block_type": "Stable",
        "version": "V1",
        "block_number": 2,
        "previous_block_type": "Volatile",
    },
    "StableV2": {
        "block_type": "Stable",
        "version": "V2",
        "block_number": 1,
        "previous_block_type": None,
    },
    "VolatileV2": {
        "block_type": "Volatile",
        "version": "V2",
        "block_number": 2,
        "previous_block_type": "Stable",
    },
    "VolatileV3": {
        "block_type": "Volatile",
        "version": "V3",
        "block_number": 1,
        "previous_block_type": None,
    },
    "VolatileV3_2": {
        "block_type": "Volatile",
        "version": "V3",
        "block_number": 2,
        "previous_block_type": "Volatile",
    },
    "StableV4": {
        "block_type": "Stable",
        "version": "V4",
        "block_number": 1,
        "previous_block_type": None,
    },
    "StableV4_2": {
        "block_type": "Stable",
        "version": "V4",
        "block_number": 2,
        "previous_block_type": "Stable",
    },
}

SECOND_BLOCK_CONDITIONS: tuple[str, ...] = (
    "StableV1",
    "VolatileV2",
    "VolatileV3_2",
    "StableV4_2",
)


# Task schedule design. Each block has 120 trials. One ball has an 80% chance
# of being safe and the other 20%, and the two swap after every run of
# ``REVERSAL_INTERVAL_TRIALS[block_type]`` trials: once in stable blocks (after
# trial 60) and every 20 trials in volatile blocks.
BLOCK_TRIAL_COUNT = 120
REVERSAL_INTERVAL_TRIALS: Dict[str, int] = {"Stable": 60, "Volatile": 20}
BETTER_OPTION_SAFE_PROBABILITY = 0.8


def better_option_by_trial(block_type: str, first_better_option: int) -> np.ndarray:
    """Return which option has the higher chance of being safe on each trial.

    The better option starts as ``first_better_option`` and swaps after every
    run of ``REVERSAL_INTERVAL_TRIALS[block_type]`` trials, across
    ``BLOCK_TRIAL_COUNT`` trials.

    Args:
        block_type: ``"Stable"`` or ``"Volatile"``.
        first_better_option: Option (0 or 1) that is better on the first trial.

    Returns:
        Integer array of length ``BLOCK_TRIAL_COUNT`` holding the better option
        (0 or 1) on each trial.

    Raises:
        ValueError: If the block type is unknown or the option is not 0 or 1.
    """
    if block_type not in REVERSAL_INTERVAL_TRIALS:
        raise ValueError(f"Unknown block type '{block_type}'")
    if first_better_option not in (0, 1):
        raise ValueError(f"first_better_option must be 0 or 1, got {first_better_option}")
    run_number = np.arange(BLOCK_TRIAL_COUNT) // REVERSAL_INTERVAL_TRIALS[block_type]
    return ((run_number + first_better_option) % 2).astype(int)


def second_block_conditions() -> List[str]:
    """List the names of the four conditions that are a participant's second block.

    Returns:
        Names of the second-block conditions, in the order they are defined in
        ``SECOND_BLOCK_CONDITIONS``.
    """
    return list(SECOND_BLOCK_CONDITIONS)


def build_context_regressors(conditions: Sequence[str]) -> np.ndarray:
    """Build centered second-block regressors for integrated fitting.

    Returns columns for current volatility, previous volatility, and their
    interaction. Only second-block conditions are valid because the integrated
    models now match the second-block ANOVA design. Volatile is coded as 0.5
    and stable as -0.5, so the interaction is +0.25 when both blocks match and
    -0.25 when they differ.

    Args:
        conditions: Condition names, one per participant, each a key of
            ``CONDITION_METADATA``.

    Returns:
        Array of shape (len(conditions), 3), with one row per entry of
        ``conditions``. The columns are current-block volatility,
        previous-block volatility and their product.

    Raises:
        KeyError: If a condition name is not in ``CONDITION_METADATA``.
        ValueError: If a condition is not a second-block condition, or has no
            previous-block type recorded.
    """
    rows = []
    for condition in conditions:
        if condition not in CONDITION_METADATA:
            raise KeyError(f"Unknown condition metadata for '{condition}'")

        metadata = CONDITION_METADATA[condition]
        if int(metadata["block_number"]) != 2:
            raise ValueError(
                f"Condition '{condition}' is not a second-block condition and cannot be used "
                "for integrated ANOVA-style fitting"
            )

        current_vol = 0.5 if metadata["block_type"] == "Volatile" else -0.5
        previous_block_type = metadata["previous_block_type"]
        if previous_block_type is None:
            raise ValueError(
                f"Condition '{condition}' is missing previous-block metadata required for "
                "integrated ANOVA-style fitting"
            )
        previous_vol = 0.5 if previous_block_type == "Volatile" else -0.5
        interaction = current_vol * previous_vol

        rows.append([current_vol, previous_vol, interaction])

    return np.asarray(rows, dtype=float)
