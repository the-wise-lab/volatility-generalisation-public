"""Tests for the convergence check in ``diagnose_mcmc``, using small synthetic chains."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from volatility_generalisation.modelling_code import diagnostics
from volatility_generalisation.modelling_code.diagnostics import diagnose_mcmc

N_CHAINS = 4
N_DRAWS = 1000


def _fake_mcmc(
    num_chains: int = N_CHAINS, diverging: np.ndarray | None = None
) -> SimpleNamespace:
    """Build a stand-in for the sampler object that ``diagnose_mcmc`` reads.

    Args:
        num_chains: Number of chains the stand-in reports.
        diverging: Divergence flag for each draw, or None to report no
            divergence information.

    Returns:
        An object with ``num_chains`` and, if ``diverging`` is given, a
        ``get_extra_fields`` method that returns it.
    """
    mcmc = SimpleNamespace(num_chains=num_chains)
    if diverging is not None:
        mcmc.get_extra_fields = lambda: {"diverging": diverging}
    return mcmc


def _diagnose(samples: np.ndarray, num_chains: int = N_CHAINS, **mcmc_kwargs) -> dict:
    """Run ``diagnose_mcmc`` on one parameter called ``p``.

    Args:
        samples: Draws of ``p``, with shape (chains, draws).
        num_chains: Number of chains the sampler stand-in reports.
        **mcmc_kwargs: Passed on to ``_fake_mcmc``, for example ``diverging``.

    Returns:
        The dictionary returned by ``diagnose_mcmc``.
    """
    return diagnose_mcmc(
        _fake_mcmc(num_chains, **mcmc_kwargs),
        {"p": samples},
        param_names=["p"],
        verbose=False,
    )


def test_good_chains_converge() -> None:
    rng = np.random.default_rng(0)

    result = _diagnose(rng.normal(size=(N_CHAINS, N_DRAWS)))

    assert result["converged"] is True
    assert result["warnings"] == []
    assert result["rhat"]["p"] <= 1.01
    assert result["ess"]["p"] > 400


def test_shifted_chains_fail_on_rhat() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(N_CHAINS, N_DRAWS)) + 5.0 * np.arange(N_CHAINS)[:, None]

    result = _diagnose(samples)

    assert result["converged"] is False
    assert result["rhat"]["p"] > 1.01
    assert any(warning.startswith("p: R-hat") for warning in result["warnings"])


def test_constant_chains_fail_because_rhat_is_nan() -> None:
    """A NaN R-hat must not pass just because NaN > 1.01 is False."""
    result = _diagnose(np.ones((N_CHAINS, N_DRAWS)))

    assert np.isnan(result["rhat"]["p"])
    assert result["converged"] is False
    assert any(
        "p: R-hat could not be computed" in warning for warning in result["warnings"]
    )


def test_one_chain_fails_because_rhat_cannot_be_computed() -> None:
    rng = np.random.default_rng(2)

    result = _diagnose(rng.normal(size=(1, N_DRAWS)), num_chains=1)

    assert np.isnan(result["rhat"]["p"])
    assert result["converged"] is False
    assert any(
        warning.startswith("p: R-hat could not be computed")
        for warning in result["warnings"]
    )


def test_autocorrelated_chains_with_low_finite_ess_still_converge_with_warning() -> (
    None
):
    rng = np.random.default_rng(3)
    # A slow wave with whole periods in each half of every chain has very high
    # autocorrelation, but the same shape in every chain and half, so R-hat stays near 1.
    wave = np.sin(2.0 * np.pi * 4.0 * np.arange(N_DRAWS) / N_DRAWS)
    samples = wave[None, :] + 0.05 * rng.normal(size=(N_CHAINS, N_DRAWS))

    result = _diagnose(samples)

    assert np.isfinite(result["ess"]["p"])
    assert result["ess"]["p"] < 400
    assert result["rhat"]["p"] <= 1.01
    assert result["converged"] is True
    assert any(warning.startswith("p: ESS") for warning in result["warnings"])


def test_divergences_still_fail() -> None:
    rng = np.random.default_rng(4)
    diverging = np.zeros(N_CHAINS * N_DRAWS, dtype=bool)
    diverging[0] = True

    result = _diagnose(rng.normal(size=(N_CHAINS, N_DRAWS)), diverging=diverging)

    assert result["divergences"] == 1
    assert result["converged"] is False


def test_returned_keys_are_unchanged() -> None:
    rng = np.random.default_rng(5)

    result = _diagnose(rng.normal(size=(N_CHAINS, N_DRAWS)))

    assert set(result) == {"converged", "warnings", "rhat", "ess", "divergences"}


def test_compute_ess_is_removed() -> None:
    assert not hasattr(diagnostics, "compute_ess")
