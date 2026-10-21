import numpy as np

from volatility_generalisation.modelling_code.posterior_utils import (
    posterior_interval,
)


def test_posterior_interval_is_exactly_the_2_5_and_97_5_percentiles():
    # Skewed draws, so an HDI or a different percentile pair would give other values.
    samples = np.random.default_rng(0).exponential(size=(500, 3))

    lower, upper = posterior_interval(samples, axis=0)

    expected = np.percentile(samples, [2.5, 97.5], axis=0)
    assert (lower == expected[0]).all()
    assert (upper == expected[1]).all()
