"""Extract model-specific posterior summaries."""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Every posterior interval in this project is a 95% equal-tailed interval: it
# runs from the 2.5th to the 97.5th percentile of the posterior samples, so 2.5%
# of the samples lie below it and 2.5% above it. The two values are written out
# in full rather than calculated from 0.95, because a calculation such as
# 100 * (1 - 0.95) / 2 gives 2.5000000000000022 and would slightly change the
# reported numbers.
POSTERIOR_INTERVAL_PERCENTILES = (2.5, 97.5)


def posterior_interval(
    samples: np.ndarray, axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the 95% equal-tailed interval of posterior samples.

    Because the interval is defined by percentiles, it stays the same when the
    samples are transformed by an increasing function such as the sigmoid. An
    interval found on the logit scale therefore maps onto the same interval on
    the 0 to 1 scale.

    Args:
        samples: Posterior samples, with the samples along ``axis``.
        axis: Axis of ``samples`` that holds the posterior samples.

    Returns:
        Tuple ``(lower, upper)`` with the 2.5th and 97.5th percentiles over the
        samples. Each has the shape of ``samples`` without ``axis``.
    """

    lower, upper = np.percentile(samples, POSTERIOR_INTERVAL_PERCENTILES, axis=axis)
    return lower, upper
