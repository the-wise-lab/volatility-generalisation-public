"""
Tests for publication figures pipeline.

Covers condition order, model-comparison filtering, learning-rate summaries,
marginal effects (with deterministic hand-computed values), PPC, and CLI smoke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# The package must be installed (``pip install -e .``) for these imports.
from volatility_generalisation.figures import publication_figures as pf
from volatility_generalisation.modelling_code import condition_metadata as cm
from volatility_generalisation.modelling_code.condition_metadata import (
    build_context_regressors,
)


def _load_cli_module():
    """Import the CLI script dynamically (it is not a package module)."""
    import importlib.util

    script_path = Path("scripts/make_publication_figures.py")
    spec = importlib.util.spec_from_file_location(
        "make_publication_figures", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers for building synthetic ArviZ InferenceData
# ---------------------------------------------------------------------------

def _build_fake_posterior(
    n_subjects_total: int = 8,
    n_chains: int = 2,
    n_draws: int = 10,
    seed: int = 42,
) -> tuple[xr.Dataset, Dict[str, List[int]], List[str]]:
    """Build a minimal posterior xarray Dataset with group and subject params.

    Returns (posterior_ds, cond_indices, condition_order).
    """
    rng = np.random.default_rng(seed)
    n_conditions = 4
    # Simulate 2 subjects per condition
    subjects_per_cond = n_subjects_total // n_conditions
    cond_names = list(pf.CONDITION_ORDER)

    cond_indices: Dict[str, List[int]] = {}
    condition_labels: List[str] = []
    for i, cond in enumerate(cond_names):
        start = i * subjects_per_cond
        end = start + subjects_per_cond
        cond_indices[cond] = list(range(start, end))
        condition_labels.extend([cond] * subjects_per_cond)

    model = pf.WINNING_MODEL

    # Build the posterior data for ArviZ's keyword-based API.
    # Group-level params (latent scale, scalar per chain/draw)
    # Shape: (n_chains, n_draws)
    posterior_data: Dict[str, np.ndarray] = {}
    for param in pf.LEARNING_RATE_PARAMS:
        for site in ["_mu", "_beta_current_vol", "_beta_previous_vol",
                     "_beta_current_previous_interaction", "_subject_sigma"]:
            key = f"{model}_{param}{site}"
            posterior_data[key] = rng.normal(0, 0.5, (n_chains, n_draws))

    # Temperature group params (needed for extract_group_params compatibility)
    for site in ["_mu", "_beta_current_vol", "_beta_previous_vol",
                 "_beta_current_previous_interaction", "_subject_sigma"]:
        key = f"{model}_temperature{site}"
        posterior_data[key] = rng.normal(0, 0.5, (n_chains, n_draws))

    # Subject-level params (natural scale)
    # Shape: (n_chains, n_draws, n_subjects_total)
    for param in pf.LEARNING_RATE_PARAMS:
        key = f"{model}_{param}_subj"
        posterior_data[key] = rng.uniform(0.1, 0.6, (n_chains, n_draws, n_subjects_total))

    temp_key = f"{model}_temperature_subj"
    posterior_data[temp_key] = rng.uniform(0.2, 0.5, (n_chains, n_draws, n_subjects_total))

    # Create DataTree using az.from_dict
    dt = az.from_dict(posterior=posterior_data)
    posterior_ds = dt["posterior"]

    return posterior_ds, cond_indices, cond_names


def _build_metadata_json(
    cond_indices: Dict[str, List[int]],
) -> Dict[str, Any]:
    """Build metadata dict matching the production ``metadata.json`` schema."""
    condition_labels: List[str] = []
    subject_ids: List[str] = []
    subject_labels: List[str] = []
    context_rows: List[List[float]] = []

    contexts = build_context_regressors(list(pf.CONDITION_ORDER))

    for idx, cond in enumerate(pf.CONDITION_ORDER):
        for si in cond_indices[cond]:
            condition_labels.append(cond)
            sid = f"subj_{si:03d}"
            subject_ids.append(sid)
            subject_labels.append(f"{cond}__{sid}")
            context_rows.append(contexts[idx].tolist())

    return {
        "model": pf.WINNING_MODEL,
        "conditions": list(pf.CONDITION_ORDER),
        "n_subjects": len(subject_ids),
        "n_conditions": len(pf.CONDITION_ORDER),
        "subject_ids": subject_ids,
        "subject_labels": subject_labels,
        "condition_labels": condition_labels,
        "context": context_rows,
    }


# ---------------------------------------------------------------------------
# Contract: condition order and display labels
# ---------------------------------------------------------------------------

class TestConditionOrderAndLabels:
    """Figure order is fixed: Stable→Stable, Volatile→Stable,
    Stable→Volatile, Volatile→Volatile."""

    def test_transition_order_and_labels(self):
        order = pf.condition_order()
        labels = pf.condition_labels()

        expected_labels = {
            "StableV4_2": "Stable → Stable",
            "StableV1": "Volatile → Stable",
            "VolatileV2": "Stable → Volatile",
            "VolatileV3_2": "Volatile → Volatile",
        }
        for cond, expected in expected_labels.items():
            assert labels[cond] == expected, f"Mismatch for {cond}"

        assert order == list(expected_labels.keys())


class TestCurrentPublicationContracts:
    def test_current_defaults_and_figure1_schema(self, tmp_path):
        cli = _load_cli_module()
        args = cli.parse_args([])
        assert args.single_lr_csv == "results/compiled/subject_parameters.csv"
        assert args.comparison_dir == "results/integrated_model_selection/model_comparison/tables"
        assert args.posterior_dir == "results/integrated_model_selection/dual_lr_single_value"

        path = tmp_path / "subject_parameters.csv"
        pd.DataFrame({
            "subject_id": ["s1", "s2"],
            "condition": ["StableV4_2", "StableV1"],
            "model": ["rescorla_wagner_single_value", "dual_lr_single_value"],
            "learning_rate": [0.2, 0.3],
        }).to_csv(path, index=False)
        loaded = pf.load_single_lr_subject_estimates(str(path))
        assert loaded["model"].unique().tolist() == ["rescorla_wagner_single_value"]

    def test_figure1_rejects_duplicate_subject_condition(self, tmp_path):
        path = tmp_path / "subject_parameters.csv"
        pd.DataFrame({
            "subject_id": ["s1", "s1"],
            "condition": ["StableV4_2", "StableV4_2"],
            "model": ["rescorla_wagner_single_value"] * 2,
            "learning_rate": [0.2, 0.3],
        }).to_csv(path, index=False)
        with pytest.raises(ValueError, match="duplicate"):
            pf.load_single_lr_subject_estimates(str(path))

    def test_figure1_stats_use_exact_two_welch_comparisons(self):
        from scipy import stats

        # Unequal group sizes and unequal spread, so that Welch's test (which
        # does not pool the variances) gives a different p value from a
        # pooled-variance test. All rates lie strictly between 0 and 1.
        values = {
            "StableV4_2": [0.20, 0.25, 0.30, 0.22],
            "StableV1": [0.30, 0.45, 0.60, 0.50, 0.70, 0.35],
            "VolatileV2": [0.30, 0.35, 0.40, 0.33, 0.37],
            "VolatileV3_2": [0.40, 0.65, 0.80, 0.55, 0.90, 0.45, 0.60],
        }
        rows = []
        for condition, rates in values.items():
            rows.extend({"condition": condition, "learning_rate": rate} for rate in rates)
        stats_table = pf.compute_figure1_bracket_stats(pd.DataFrame(rows))
        assert len(stats_table) == 2
        assert stats_table["comparison"].tolist() == [
            "Stable→Stable vs Volatile→Stable",
            "Stable→Volatile vs Volatile→Volatile",
        ]
        assert stats_table["test"].nunique() == 1
        assert all(label in {"***", "**", "*", "n.s."} for label in stats_table["p_label"])

        def welch_two_sided_p_value(rates_a, rates_b):
            """Welch's test on the logit of the rates, written out step by step.

            Args:
                rates_a: Learning rates in the first group, each between 0 and 1.
                rates_b: Learning rates in the second group, each between 0 and 1.

            Returns:
                The two-sided p-value of Welch's t-test on the logits.
            """
            logit_a = np.log(np.array(rates_a) / (1 - np.array(rates_a)))
            logit_b = np.log(np.array(rates_b) / (1 - np.array(rates_b)))
            # Squared standard error of each group mean: sample variance / n.
            squared_error_a = logit_a.var(ddof=1) / len(logit_a)
            squared_error_b = logit_b.var(ddof=1) / len(logit_b)
            t_statistic = (logit_a.mean() - logit_b.mean()) / np.sqrt(
                squared_error_a + squared_error_b
            )
            # Welch-Satterthwaite degrees of freedom.
            degrees_of_freedom = (squared_error_a + squared_error_b) ** 2 / (
                squared_error_a ** 2 / (len(logit_a) - 1)
                + squared_error_b ** 2 / (len(logit_b) - 1)
            )
            return 2 * stats.t.sf(abs(t_statistic), degrees_of_freedom)

        expected_p_values = [
            welch_two_sided_p_value(values["StableV4_2"], values["StableV1"]),
            welch_two_sided_p_value(values["VolatileV2"], values["VolatileV3_2"]),
        ]
        assert stats_table["p_value_raw"].tolist() == pytest.approx(
            expected_p_values, rel=1e-8, abs=1e-10
        )

    def test_fixed_subject_scale_group_posterior_is_supported(self):
        posterior_data = {}
        rng = np.random.default_rng(4)
        for param in pf.LEARNING_RATE_PARAMS:
            for site in pf.REQUIRED_GROUP_SITES:
                posterior_data[f"{pf.WINNING_MODEL}_{param}{site}"] = rng.normal(0, 1, (1, 2))
            posterior_data[f"{pf.WINNING_MODEL}_{param}_subj"] = rng.uniform(
                0.1, 0.9, (1, 2, 4)
            )
        posterior_ds = az.from_dict(posterior=posterior_data)["posterior"]
        extracted = pf.extract_group_params(posterior_ds)
        assert not any(key.endswith("_subject_sigma") for key in extracted)


# ---------------------------------------------------------------------------
# Contract: Figure 2 uses exact retained-model allowlist
# ---------------------------------------------------------------------------

# Synthetic model names for testing the allowlist mechanism.
_TEST_ALLOWLIST = [
    "rescorla_wagner_single_value",
    "dual_lr_single_value",
    "pearce_hall_single_value",
    "leaky_beta_single_value",
    "leaky_beta_asymmetric_single_value",
]

_TEST_EXTRA_SINGLE_VALUE = "hypothetical_extra_single_value"


def _make_comparison_df(
    models,
    rng_seed=42,
    include_dse=True,
    include_stacking_weight=True,
):
    """Helper to build a synthetic comparison DataFrame with standard columns."""
    rng = np.random.default_rng(rng_seed)
    n = len(models)
    data = {
        "model": models,
        "elpd_diff": -np.abs(rng.normal(0, 2, n)),
    }
    if include_dse:
        data["dse"] = rng.exponential(1, n)
    if include_stacking_weight:
        data["stacking_weight"] = rng.dirichlet(np.ones(n))
    return pd.DataFrame(data)


class TestModelComparisonSingleValueFilter:
    """Default Figure 2 source data keeps exactly the retained models.

    Fixes:
    - Uses exact retained-model allowlist instead of ``contains("_single_value")``.
    - Requires ``dse`` column (not ``se``).
    - Requires ``stacking_weight`` column.
    - Handles ArviZ ``az.compare`` index format and CSV unnamed-index
      roundtrip.
    """

    def test_allows_exactly_retained_models(self):
        """Only the retained allowlist models pass through."""
        # Include an extra unknown _single_value model that should be excluded
        models = _TEST_ALLOWLIST + [_TEST_EXTRA_SINGLE_VALUE]
        comparison = _make_comparison_df(models)

        filtered = pf.load_comparison_for_figure2(comparison)

        assert len(filtered) == 5, (
            f"Expected exactly 5, got {len(filtered)}: {list(filtered['model'])}"
        )
        assert set(filtered["model"]) == set(_TEST_ALLOWLIST)
        assert filtered["model"].tolist() == list(pf.FIGURE2_MODEL_ORDER)

    def test_arviz_format_model_names_in_index(self):
        """Handle ArviZ ``az.compare`` output where model names are in the index."""
        rng = np.random.default_rng(42)
        n_models = len(_TEST_ALLOWLIST) + 2
        models = _TEST_ALLOWLIST + ["rescorla_wagner", "dual_lr"]

        # Simulate az.compare output: model names in index
        data = {
            "rank": list(range(n_models)),
            "elpd_diff": -np.abs(rng.normal(0, 2, n_models)),
            "dse": rng.exponential(1, n_models),
            "weight": rng.dirichlet(np.ones(n_models)),
        }
        comparison = pd.DataFrame(data, index=models)

        filtered = pf.load_comparison_for_figure2(comparison)

        assert len(filtered) == 5
        assert set(filtered["model"]) == set(_TEST_ALLOWLIST)
        # Check columns were mapped correctly
        assert "se_diff" in filtered.columns
        assert "stacking_weight" in filtered.columns

    def test_csv_roundtrip_unnamed_index(self):
        """CSV written from az.compare format may have an unnamed index column
        named 'Unnamed: 0'; loader must handle this.

        This exactly reproduces the production round-trip:
        ``az.compare()`` returns a DataFrame with model names in the index;
        writing via ``compare_df.to_csv(path)`` (default ``index=True``)
        creates an ``Unnamed: 0`` column when read back.
        """
        rng = np.random.default_rng(42)
        models = _TEST_ALLOWLIST + ["rescorla_wagner", "dual_lr"]
        n_models = len(models)

        # Simulate az.compare output: model names in the index
        data = {
            "rank": list(range(n_models)),
            "elpd_diff": -np.abs(rng.normal(0, 2, n_models)),
            "dse": rng.exponential(1, n_models),
            "weight": rng.dirichlet(np.ones(n_models)),
        }
        original = pd.DataFrame(data, index=models)

        # Production round-trip: to_csv(index=True) → pd.read_csv
        import io
        csv_str = original.to_csv()  # index=True (default)
        reloaded = pd.read_csv(io.StringIO(csv_str))

        # Verify the unnamed index column exists
        unnamed_cols = [c for c in reloaded.columns if c.startswith("Unnamed:")]
        assert len(unnamed_cols) == 1, (
            f"Expected one 'Unnamed:*' column, got {unnamed_cols}"
        )

        filtered = pf.load_comparison_for_figure2(reloaded)
        assert len(filtered) == 5
        assert set(filtered["model"]) == set(_TEST_ALLOWLIST)
        assert "se_diff" in filtered.columns
        assert "stacking_weight" in filtered.columns

    def test_missing_dse_raises_value_error(self):
        """The ``dse`` column is required; missing it raises a clear error."""
        comparison = _make_comparison_df(_TEST_ALLOWLIST, include_dse=False)
        with pytest.raises(ValueError, match="dse"):
            pf.load_comparison_for_figure2(comparison)

    def test_missing_stacking_weight_raises_value_error(self):
        """The ``stacking_weight`` column is required; missing it raises a clear error."""
        comparison = _make_comparison_df(_TEST_ALLOWLIST, include_stacking_weight=False)
        with pytest.raises(ValueError, match="stacking_weight"):
            pf.load_comparison_for_figure2(comparison)

    def test_default_table_rejects_non_allowlist_models(self):
        """Table with no allowlist models raises an error."""
        comparison = _make_comparison_df(
            ["rescorla_wagner", "dual_lr", "leaky_beta"],
        )
        with pytest.raises(ValueError, match="retained model"):
            pf.load_comparison_for_figure2(comparison)


# ---------------------------------------------------------------------------
# Contract: Learning-rate summaries use mean and 95% CI only (Findings 2, 4, 5)
# ---------------------------------------------------------------------------

# The context codes of the four conditions, written out in the order of
# ``pf.CONDITION_ORDER`` (StableV4_2, StableV1, VolatileV2, VolatileV3_2). Each
# row is (current volatility, previous volatility, their interaction).
_CONTEXT_ROWS_IN_CONDITION_ORDER = [
    (-0.5, -0.5, 0.25),
    (-0.5, 0.5, -0.25),
    (0.5, -0.5, -0.25),
    (0.5, 0.5, 0.25),
]


def _expected_rates_by_condition(group_samples: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Work out the population learning rate of each condition for every sample.

    The rate is the sigmoid, ``1 / (1 + exp(-x))``, of the group mean plus each
    volatility weight times its context code. This is written out here, without
    the helpers of the code under test.

    Args:
        group_samples: Samples of one parameter's group-level values, with the
            keys ``mu``, ``beta_current``, ``beta_previous`` and
            ``beta_interaction``, each of shape ``(n_samples,)``.

    Returns:
        List of four arrays, one per condition in the order of
        ``_CONTEXT_ROWS_IN_CONDITION_ORDER``, each of shape ``(n_samples,)``.
    """
    rates = []
    for current, previous, interaction in _CONTEXT_ROWS_IN_CONDITION_ORDER:
        latent = (
            group_samples["mu"]
            + group_samples["beta_current"] * current
            + group_samples["beta_previous"] * previous
            + group_samples["beta_interaction"] * interaction
        )
        rates.append(1.0 / (1.0 + np.exp(-latent)))
    return rates


def _group_samples_from_posterior(posterior_ds: xr.Dataset, param: str) -> Dict[str, np.ndarray]:
    """Read one parameter's group-level samples straight from the posterior.

    Args:
        posterior_ds: Posterior with the group-level sites of the winning model.
        param: Parameter name, for example ``learning_rate_pos``.

    Returns:
        Dictionary with ``mu``, ``beta_current``, ``beta_previous`` and
        ``beta_interaction``, each flattened over chains to ``(n_samples,)``.
    """
    prefix = f"{pf.WINNING_MODEL}_{param}"
    return {
        "mu": posterior_ds[f"{prefix}_mu"].values.reshape(-1),
        "beta_current": posterior_ds[f"{prefix}_beta_current_vol"].values.reshape(-1),
        "beta_previous": posterior_ds[f"{prefix}_beta_previous_vol"].values.reshape(-1),
        "beta_interaction": posterior_ds[
            f"{prefix}_beta_current_previous_interaction"
        ].values.reshape(-1),
    }


class TestLearningRateSummaryMeanAnd95CI:
    """Figure 3 source data: subject posterior means, condition-specific
    population means, 95% intervals — no 50% interval columns.

    Fixes:
    - *subj parameters are already on natural scale (no expit applied).
    - Population summaries are computed from group-level parameters.
    """

    @pytest.fixture
    def fake_posterior_pkg(self) -> tuple[xr.Dataset, Dict[str, Any], Dict[str, List[int]]]:
        """Create synthetic posterior Dataset and metadata."""
        n_subjects_total = 8
        n_chains = 2
        n_draws = 10
        posterior_ds, cond_indices, cond_names = _build_fake_posterior(
            n_subjects_total=n_subjects_total,
            n_chains=n_chains,
            n_draws=n_draws,
        )
        metadata = _build_metadata_json(cond_indices)
        return posterior_ds, metadata, cond_indices

    def test_subject_values_are_posterior_means_on_natural_scale(self):
        """Each participant's value is the plain mean of their natural-scale samples.

        The samples are unequal and skewed (the mean differs from the median),
        so a constant, a median, an extra sigmoid or averaging over the wrong
        axis would each give different numbers.
        """
        # Samples have shape (2 chains, 3 draws, 3 participants). Each list
        # below is one participant's six samples, three per chain.
        pos_samples = [
            [[0.1, 0.2, 0.9], [0.15, 0.1, 0.2]],
            [[0.5, 0.6, 0.7], [0.4, 0.8, 0.5]],
            [[0.05, 0.05, 0.8], [0.1, 0.7, 0.3]],
        ]
        neg_samples = [
            [[0.3, 0.3, 0.6], [0.2, 0.4, 0.9]],
            [[0.1, 0.1, 0.1], [0.1, 0.1, 0.7]],
            [[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]],
        ]
        posterior_data = {
            f"{pf.WINNING_MODEL}_learning_rate_pos_subj": np.stack(
                [np.array(rows) for rows in pos_samples], axis=-1
            ),
            f"{pf.WINNING_MODEL}_learning_rate_neg_subj": np.stack(
                [np.array(rows) for rows in neg_samples], axis=-1
            ),
        }
        posterior_ds = az.from_dict(posterior=posterior_data)["posterior"]
        # Participants 0 and 1 are in the first condition, participant 2 in the second.
        cond_indices = {"StableV4_2": [0, 1], "VolatileV3_2": [2]}

        subj_by_cond = pf.extract_subject_params(posterior_ds, cond_indices)
        summary = pf.compute_learning_rate_subject_summary(subj_by_cond)

        assert summary["condition"].tolist() == ["StableV4_2", "StableV4_2", "VolatileV3_2"]
        assert summary["subject_idx"].tolist() == [0, 1, 0]
        # Means over all six samples of each participant, worked out by hand.
        # The medians would be 0.175, 0.55 and 0.2 for the positive rate.
        expected_pos = [
            (0.1 + 0.2 + 0.9 + 0.15 + 0.1 + 0.2) / 6,
            (0.5 + 0.6 + 0.7 + 0.4 + 0.8 + 0.5) / 6,
            (0.05 + 0.05 + 0.8 + 0.1 + 0.7 + 0.3) / 6,
        ]
        expected_neg = [
            (0.3 + 0.3 + 0.6 + 0.2 + 0.4 + 0.9) / 6,
            (0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.7) / 6,
            (0.25 + 0.5 + 0.75 + 0.25 + 0.5 + 0.75) / 6,
        ]
        assert summary["learning_rate_pos_subject_mean"].tolist() == pytest.approx(
            expected_pos, rel=1e-8, abs=1e-10
        )
        assert summary["learning_rate_neg_subject_mean"].tolist() == pytest.approx(
            expected_neg, rel=1e-8, abs=1e-10
        )

    def test_population_summary_is_condition_specific(
        self, fake_posterior_pkg
    ):
        """Each condition's population mean matches the sigmoid of its own context."""
        posterior_ds, metadata, cond_indices = fake_posterior_pkg
        group_params = pf.extract_group_params(posterior_ds)
        first_key = next(iter(group_params.keys()))
        n_samples = len(group_params[first_key])

        pop_summary = pf.compute_condition_population_summary(
            group_params, n_samples
        )

        for param in ("learning_rate_pos", "learning_rate_neg"):
            expected_rates = _expected_rates_by_condition(
                _group_samples_from_posterior(posterior_ds, param)
            )
            for cond, rates in zip(pf.CONDITION_ORDER, expected_rates):
                assert cond in pop_summary, f"Missing condition {cond}"
                for key in [f"{param}_population_mean",
                            f"{param}_ci_lower",
                            f"{param}_ci_upper"]:
                    assert key in pop_summary[cond], (
                        f"Missing {key} for {cond}"
                    )
                assert pop_summary[cond][f"{param}_population_mean"] == pytest.approx(
                    rates.mean(), rel=1e-8, abs=1e-10
                )

    def test_no_50_percent_interval_columns(self, fake_posterior_pkg):
        """Neither subject nor population summary should emit 50% CI columns."""
        posterior_ds, metadata, cond_indices = fake_posterior_pkg
        subj_by_cond = pf.extract_subject_params(posterior_ds, cond_indices)
        group_params = pf.extract_group_params(posterior_ds)
        first_key = next(iter(group_params.keys()))
        n_samples = len(group_params[first_key])

        subj = pf.compute_learning_rate_subject_summary(subj_by_cond)
        pop = pf.compute_condition_population_summary(group_params, n_samples)

        for col in subj.columns:
            assert "50" not in col, f"Found 50% interval column: {col}"
        for cond, cond_stats in pop.items():
            for key in cond_stats:
                assert "50" not in key, f"Found 50% interval key: {key}"

    def test_population_summary_uses_95_percent_interval(self, fake_posterior_pkg):
        """Verify the mean and the 2.5/97.5 percentiles of each population rate.

        Args:
            fake_posterior_pkg: Synthetic posterior and condition metadata.
        """
        posterior_ds, metadata, cond_indices = fake_posterior_pkg
        group_params = pf.extract_group_params(posterior_ds)
        first_key = next(iter(group_params.keys()))
        n_samples = len(group_params[first_key])

        pop_summary = pf.compute_condition_population_summary(
            group_params, n_samples, model_name=pf.WINNING_MODEL
        )

        for param in pf.LEARNING_RATE_PARAMS:
            expected_rates = _expected_rates_by_condition(
                _group_samples_from_posterior(posterior_ds, param)
            )
            for cond, rates in zip(pf.CONDITION_ORDER, expected_rates):
                expected_lower, expected_upper = np.percentile(rates, [2.5, 97.5])

                assert pop_summary[cond][f"{param}_population_mean"] == pytest.approx(
                    rates.mean(), rel=1e-8, abs=1e-10
                )
                assert pop_summary[cond][f"{param}_ci_lower"] == pytest.approx(
                    expected_lower, rel=1e-8, abs=1e-10
                )
                assert pop_summary[cond][f"{param}_ci_upper"] == pytest.approx(
                    expected_upper, rel=1e-8, abs=1e-10
                )


# ---------------------------------------------------------------------------
# Contract: the Figure 3 coefficients table is on the logit scale
# ---------------------------------------------------------------------------


def test_logit_coefficient_summary_matches_each_sites_own_samples():
    """Check every row against its own site's logit-scale draws.

    Each site gets different skewed draws, so swapping two sites, or reporting
    a row on the 0 to 1 scale, gives a different number and fails the test.
    """
    rng = np.random.default_rng(3)
    group_params = {}
    expected = {}
    for param_idx, param in enumerate(pf.LEARNING_RATE_PARAMS):
        for site_idx, site in enumerate(pf.REQUIRED_GROUP_SITES):
            samples = (
                rng.exponential(scale=1 + site_idx, size=400)
                + 10 * param_idx
                - site_idx
            )
            group_params[f"{pf.WINNING_MODEL}_{param}{site}"] = samples
            expected[(param, site.lstrip("_"))] = samples

    table = pf.compute_logit_coefficient_summary(group_params)

    assert len(table) == 8
    assert set(zip(table.parameter, table.coefficient)) == set(expected)
    assert list(table.columns) == [
        "figure",
        "parameter",
        "coefficient",
        "mean",
        "sd",
        "ci_lower",
        "ci_upper",
    ]
    for row in table.itertuples():
        samples = expected[(row.parameter, row.coefficient)]
        lower, upper = np.percentile(samples, [2.5, 97.5])
        assert row.mean == pytest.approx(samples.mean(), rel=1e-12)
        assert row.sd == pytest.approx(samples.std(ddof=1), rel=1e-12)
        assert row.ci_lower == pytest.approx(lower, rel=1e-12)
        assert row.ci_upper == pytest.approx(upper, rel=1e-12)


# ---------------------------------------------------------------------------
# Contract: Figure 3 forest effects are natural-scale marginal contrasts
#           (Findings 4, 5)
# ---------------------------------------------------------------------------

class TestLearningRateEffectsNaturalScale:
    """Marginal effects on natural learning-rate scale from group parameters.

    Tests include deterministic hand-computed values."""
    @pytest.fixture
    def posterior_with_known_group_params(self) -> tuple[xr.Dataset, Dict[str, List[int]]]:
        """Create posterior Dataset with known group-level parameters for hand computation."""
        n_chains = 1
        n_draws = 1
        n_subjects_total = 8
        model = pf.WINNING_MODEL

        cond_indices: Dict[str, List[int]] = {}
        subjects_per_cond = 2
        for i, cond in enumerate(pf.CONDITION_ORDER):
            start = i * subjects_per_cond
            end = start + subjects_per_cond
            cond_indices[cond] = list(range(start, end))

        # Known group params for learning_rate_pos (scalars, shaped (1, 1))
        # Use az.from_dict which expects dict of {group: {var: array}}
        posterior_data: Dict[str, np.ndarray] = {}

        posterior_data[f"{model}_learning_rate_pos_mu"] = np.array([[0.5]])
        posterior_data[f"{model}_learning_rate_pos_beta_current_vol"] = np.array([[0.8]])
        posterior_data[f"{model}_learning_rate_pos_beta_previous_vol"] = np.array([[-0.3]])
        posterior_data[f"{model}_learning_rate_pos_beta_current_previous_interaction"] = np.array([[0.2]])
        posterior_data[f"{model}_learning_rate_pos_subject_sigma"] = np.array([[0.5]])

        posterior_data[f"{model}_learning_rate_neg_mu"] = np.array([[-0.5]])
        posterior_data[f"{model}_learning_rate_neg_beta_current_vol"] = np.array([[0.4]])
        posterior_data[f"{model}_learning_rate_neg_beta_previous_vol"] = np.array([[-0.1]])
        posterior_data[f"{model}_learning_rate_neg_beta_current_previous_interaction"] = np.array([[0.05]])
        posterior_data[f"{model}_learning_rate_neg_subject_sigma"] = np.array([[0.5]])

        # Temperature group params
        posterior_data[f"{model}_temperature_mu"] = np.array([[0.0]])
        posterior_data[f"{model}_temperature_beta_current_vol"] = np.array([[0.0]])
        posterior_data[f"{model}_temperature_beta_previous_vol"] = np.array([[0.0]])
        posterior_data[f"{model}_temperature_beta_current_previous_interaction"] = np.array([[0.0]])
        posterior_data[f"{model}_temperature_subject_sigma"] = np.array([[0.5]])

        # Subject-level (natural scale, won't affect marginal effects)
        posterior_data[f"{model}_learning_rate_pos_subj"] = np.random.default_rng(1).uniform(0.2, 0.5, (1, 1, n_subjects_total))
        posterior_data[f"{model}_learning_rate_neg_subj"] = np.random.default_rng(2).uniform(0.1, 0.3, (1, 1, n_subjects_total))
        posterior_data[f"{model}_temperature_subj"] = np.random.default_rng(3).uniform(0.3, 0.4, (1, 1, n_subjects_total))

        # Create DataTree
        dt = az.from_dict(posterior=posterior_data)
        posterior_ds = dt["posterior"]

        return posterior_ds, cond_indices

    def test_marginal_effects_hand_computed_pos(self, posterior_with_known_group_params):
        """Deterministic hand-computed marginal effects for learning_rate_pos.

        With mu=0.5, beta_c=0.8, beta_p=-0.3, beta_i=0.2:

        Condition context encodings (±0.5):
          C0: StableV4_2  (cv=-0.5, pv=-0.5, iv=0.25)  [S→S]
          C1: StableV1    (cv=-0.5, pv=+0.5, iv=-0.25) [V→S]
          C2: VolatileV2  (cv=+0.5, pv=-0.5, iv=-0.25) [S→V]
          C3: VolatileV3_2 (cv=+0.5, pv=+0.5, iv=0.25) [V→V]

        Natural-scale per-condition population values:
          V_c = sigmoid(mu + beta_c*cv + beta_p*pv + beta_i*iv)

          V0 = sigmoid(0.5 + 0.8*(-0.5) + (-0.3)*(-0.5) + 0.2*0.25)
             = sigmoid(0.5 - 0.4 + 0.15 + 0.05) = sigmoid(0.3) ≈ 0.5744
          V1 = sigmoid(0.5 + 0.8*(-0.5) + (-0.3)*0.5 + 0.2*(-0.25))
             = sigmoid(0.5 - 0.4 - 0.15 - 0.05) = sigmoid(-0.1) ≈ 0.4750
          V2 = sigmoid(0.5 + 0.8*0.5 + (-0.3)*(-0.5) + 0.2*(-0.25))
             = sigmoid(0.5 + 0.4 + 0.15 - 0.05) = sigmoid(1.0) ≈ 0.7311
          V3 = sigmoid(0.5 + 0.8*0.5 + (-0.3)*0.5 + 0.2*0.25)
             = sigmoid(0.5 + 0.4 - 0.15 + 0.05) = sigmoid(0.8) ≈ 0.6900

        Marginal effects via contrast coding:
          current = Σ_c contrasts_c[c,0] * V_c = 0.5*(V2+V3) - 0.5*(V0+V1)
                  = 0.5*(0.7311+0.6900 - 0.5744-0.4750) = 0.5*0.3717 ≈ 0.1858
          previous = Σ_c contrasts_c[c,1] * V_c = 0.5*(V1+V3) - 0.5*(V0+V2)
                   = 0.5*(0.4750+0.6900 - 0.5744-0.7311) = 0.5*(-0.1405) ≈ -0.0702
          interaction = true diff-in-differences: SS - VS - SV + VV = V0 - V1 - V2 + V3
                      = 0.5744 - 0.4750 - 0.7311 + 0.6900 = 0.0583
          (NOT quarter-scaled: the raw contrast coding gives ±0.25 × diff, which is
          then multiplied by 4 to recover the true diff-in-diff.)
        """
        posterior_ds, cond_indices = posterior_with_known_group_params
        group_params = pf.extract_group_params(posterior_ds)
        n_samples = 1

        effects = pf.compute_marginal_effects(group_params, n_samples)
        ef_pos = effects["learning_rate_pos"]

        # Hand-computed values
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        V0 = sigmoid(0.5 + 0.8*(-0.5) + (-0.3)*(-0.5) + 0.2*0.25)
        V1 = sigmoid(0.5 + 0.8*(-0.5) + (-0.3)*0.5 + 0.2*(-0.25))
        V2 = sigmoid(0.5 + 0.8*0.5 + (-0.3)*(-0.5) + 0.2*(-0.25))
        V3 = sigmoid(0.5 + 0.8*0.5 + (-0.3)*0.5 + 0.2*0.25)

        expected_current = 0.5 * (V2 + V3 - V0 - V1)
        expected_previous = 0.5 * (V1 + V3 - V0 - V2)
        # True diff-in-differences: SS - VS - SV + VV (no quarter scaling)
        expected_interaction = V0 + V3 - V1 - V2

        assert ef_pos["current"].iloc[0] == pytest.approx(expected_current, rel=1e-4)
        assert ef_pos["previous"].iloc[0] == pytest.approx(expected_previous, rel=1e-4)
        assert ef_pos["interaction"].iloc[0] == pytest.approx(expected_interaction, rel=1e-4)

    def test_marginal_effects_hand_computed_neg(self, posterior_with_known_group_params):
        """Deterministic hand-computed marginal effects for learning_rate_neg.

        With mu=-0.5, beta_c=0.4, beta_p=-0.1, beta_i=0.05:

          V0 = sigmoid(-0.5 + 0.4*(-0.5) + (-0.1)*(-0.5) + 0.05*0.25)
             = sigmoid(-0.5 - 0.2 + 0.05 + 0.0125) = sigmoid(-0.6375) ≈ 0.3457
          V1 = sigmoid(-0.5 + 0.4*(-0.5) + (-0.1)*0.5 + 0.05*(-0.25))
             = sigmoid(-0.5 - 0.2 - 0.05 - 0.0125) = sigmoid(-0.7625) ≈ 0.3181
          V2 = sigmoid(-0.5 + 0.4*0.5 + (-0.1)*(-0.5) + 0.05*(-0.25))
             = sigmoid(-0.5 + 0.2 + 0.05 - 0.0125) = sigmoid(-0.2625) ≈ 0.4348
          V3 = sigmoid(-0.5 + 0.4*0.5 + (-0.1)*0.5 + 0.05*0.25)
             = sigmoid(-0.5 + 0.2 - 0.05 + 0.0125) = sigmoid(-0.3375) ≈ 0.4164
        """
        posterior_ds, cond_indices = posterior_with_known_group_params
        group_params = pf.extract_group_params(posterior_ds)
        n_samples = 1

        effects = pf.compute_marginal_effects(group_params, n_samples)
        ef_neg = effects["learning_rate_neg"]

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        V0 = sigmoid(-0.5 + 0.4*(-0.5) + (-0.1)*(-0.5) + 0.05*0.25)
        V1 = sigmoid(-0.5 + 0.4*(-0.5) + (-0.1)*0.5 + 0.05*(-0.25))
        V2 = sigmoid(-0.5 + 0.4*0.5 + (-0.1)*(-0.5) + 0.05*(-0.25))
        V3 = sigmoid(-0.5 + 0.4*0.5 + (-0.1)*0.5 + 0.05*0.25)

        expected_current = 0.5 * (V2 + V3 - V0 - V1)
        expected_previous = 0.5 * (V1 + V3 - V0 - V2)
        expected_interaction = V0 + V3 - V1 - V2  # true diff-in-diff

        assert ef_neg["current"].iloc[0] == pytest.approx(expected_current, rel=1e-4)
        assert ef_neg["previous"].iloc[0] == pytest.approx(expected_previous, rel=1e-4)
        assert ef_neg["interaction"].iloc[0] == pytest.approx(expected_interaction, rel=1e-4)

    def test_effects_on_natural_scale(self, posterior_with_known_group_params):
        """All effects should be in the [-1, 1] range (differences of [0,1] values)."""
        posterior_ds, cond_indices = posterior_with_known_group_params
        group_params = pf.extract_group_params(posterior_ds)
        n_samples = 1

        effects = pf.compute_marginal_effects(group_params, n_samples)

        for lr_type in ("learning_rate_pos", "learning_rate_neg"):
            assert lr_type in effects
            df = effects[lr_type]
            for col in ("current", "previous", "interaction"):
                assert col in df.columns
            assert df.values.min() >= -1.0
            assert df.values.max() <= 1.0

    def test_marginal_effects_have_posterior_intervals(self):
        """Effects carry the mean and 95% interval of their per-draw contrasts.

        Five distinct posterior draws are used, so the interval ends are
        interpolated between draws and cannot be read off one sample.
        """
        model = pf.WINNING_MODEL
        draws = {
            "learning_rate_pos": {
                "mu": np.array([0.5, 0.2, -0.3, 0.9, 0.0]),
                "beta_current": np.array([0.8, 0.1, 0.5, -0.4, 0.3]),
                "beta_previous": np.array([-0.3, 0.6, 0.2, 0.1, -0.5]),
                "beta_interaction": np.array([0.2, -0.2, 0.4, 0.0, 0.1]),
            },
            "learning_rate_neg": {
                "mu": np.array([-0.5, -0.1, 0.3, -0.8, 0.2]),
                "beta_current": np.array([0.4, -0.6, 0.2, 0.7, -0.1]),
                "beta_previous": np.array([-0.1, 0.3, -0.4, 0.5, 0.2]),
                "beta_interaction": np.array([0.05, 0.3, -0.2, 0.1, -0.4]),
            },
        }
        group_params = {}
        for param, samples in draws.items():
            group_params[f"{model}_{param}_mu"] = samples["mu"]
            group_params[f"{model}_{param}_beta_current_vol"] = samples["beta_current"]
            group_params[f"{model}_{param}_beta_previous_vol"] = samples["beta_previous"]
            group_params[
                f"{model}_{param}_beta_current_previous_interaction"
            ] = samples["beta_interaction"]

        effects = pf.compute_marginal_effects(group_params, n_samples=5)

        for lr_type, samples in draws.items():
            df = effects[lr_type]
            # Learning rate of each condition for each draw, then the three
            # contrasts: current = volatile-current minus stable-current
            # conditions, previous likewise, and the interaction as a
            # difference of differences (stable-stable minus volatile-stable
            # minus stable-volatile plus volatile-volatile).
            stable_stable, volatile_stable, stable_volatile, volatile_volatile = (
                _expected_rates_by_condition(samples)
            )
            expected_contrasts = {
                "current": 0.5 * (
                    stable_volatile + volatile_volatile - stable_stable - volatile_stable
                ),
                "previous": 0.5 * (
                    volatile_stable + volatile_volatile - stable_stable - stable_volatile
                ),
                "interaction": (
                    stable_stable + volatile_volatile - volatile_stable - stable_volatile
                ),
            }
            for col_base, contrast in expected_contrasts.items():
                lower, upper = np.percentile(contrast, [2.5, 97.5])
                assert df[col_base].iloc[0] == pytest.approx(
                    contrast.mean(), rel=1e-8, abs=1e-10
                ), f"{lr_type} {col_base} mean"
                assert df[f"{col_base}_ci_lower"].iloc[0] == pytest.approx(
                    lower, rel=1e-8, abs=1e-10
                ), f"{lr_type} {col_base} lower"
                assert df[f"{col_base}_ci_upper"].iloc[0] == pytest.approx(
                    upper, rel=1e-8, abs=1e-10
                ), f"{lr_type} {col_base} upper"


# ---------------------------------------------------------------------------
# Contract: PPC from posterior and prepared data  (Finding 1)
# ---------------------------------------------------------------------------

class TestPpcFromWinningModel:
    """PPC is generated from the winning integrated model and prepared
    condition data."""

    def test_ppc_requires_both_inputs(self, tmp_path):
        with pytest.raises(ValueError, match="posterior"):
            pf.compute_ppc_summary(
                posterior=None,
                prepared_data={"choices": np.ones((2, 10))},
            )

        with pytest.raises(ValueError, match="prepared"):
            pf.compute_ppc_summary(
                posterior={"some_key": np.ones((5, 2))},
                prepared_data=None,
            )

    def test_ppc_with_synthetic_data(self):

        n_subjects = 2
        n_trials = 5
        n_samples = 10

        # Create minimal posterior with winning model params
        # (natural scale — no inv-logit applied)
        posterior = {
            "learning_rate_pos": np.full(
                (n_samples, n_subjects), 0.3
            ),
            "learning_rate_neg": np.full(
                (n_samples, n_subjects), 0.2
            ),
            "temperature": np.full(
                (n_samples, n_subjects), 0.5
            ),
        }

        prepared_data = {
            "outcomes_0": np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]),
            "outcomes_1": np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]),
            "choices": np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]),
        }

        result = pf.compute_ppc_summary(posterior, prepared_data)

        assert "trial_mean_pred" in result
        assert "trial_mean_obs" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert len(result["trial_mean_pred"]) == n_trials

    def test_ppc_accepts_pre_extracted_format(self):
        """PPC should accept dict with bare param names (no model prefix)."""
        n_subjects = 2
        n_trials = 5
        n_samples = 10

        posterior = {
            "learning_rate_pos": np.full((n_samples, n_subjects), 0.3),
            "learning_rate_neg": np.full((n_samples, n_subjects), 0.2),
            "temperature": np.full((n_samples, n_subjects), 0.5),
        }
        prepared_data = {
            "outcomes_0": np.full((n_subjects, n_trials), 0.5),
            "outcomes_1": np.full((n_subjects, n_trials), 0.5),
            "choices": np.ones((n_subjects, n_trials), dtype=int),
        }
        result = pf.compute_ppc_summary(posterior, prepared_data)
        assert "trial_mean_pred" in result


# ---------------------------------------------------------------------------
# Contract: posterior loading from ArviZ format  (Finding 1)
# ---------------------------------------------------------------------------

class TestIntegratedPosteriorLoading:
    """Loading from posterior.nc + metadata.json using load_posterior_and_metadata."""

    def test_load_posterior_and_metadata_missing(self, tmp_path: Path):
        """Verify that load_posterior_and_metadata raises FileNotFoundError
        when the file doesn't exist."""
        missing_path = tmp_path / "posterior.nc"
        with pytest.raises(FileNotFoundError, match="Posterior file not found"):
            pf.load_posterior_and_metadata(missing_path)

    def test_extract_subject_params(self):
        """Verify subject params are extracted per condition."""
        posterior_ds, cond_indices, _ = _build_fake_posterior()
        subj_by_cond = pf.extract_subject_params(posterior_ds, cond_indices)

        for cond in pf.CONDITION_ORDER:
            assert cond in subj_by_cond
            for param in pf.LEARNING_RATE_PARAMS:
                assert param in subj_by_cond[cond], f"Missing {param} for {cond}"
                n_subjects_cond = len(cond_indices[cond])
                assert subj_by_cond[cond][param].shape[1] == n_subjects_cond

    def test_extract_group_params(self):
        """Verify group params are extracted."""
        posterior_ds, cond_indices, _ = _build_fake_posterior()
        group_params = pf.extract_group_params(posterior_ds)

        model = pf.WINNING_MODEL
        for param in pf.LEARNING_RATE_PARAMS:
            for site in ["_mu", "_beta_current_vol", "_beta_previous_vol",
                         "_beta_current_previous_interaction", "_subject_sigma"]:
                key = f"{model}_{param}{site}"
                assert key in group_params, f"Missing group param {key}"


# ---------------------------------------------------------------------------
# Contract: missing inputs raise clear errors  (Finding 6)
# ---------------------------------------------------------------------------

class TestMissingInputErrors:
    """Missing required condition/model inputs must raise clear errors."""

    def test_figures_flag_is_not_accepted(self):
        """The command line has no ``--figures`` flag; passing it should raise an
        ``unrecognized arguments`` error."""
        cli = _load_cli_module()
        with pytest.raises(SystemExit):
            cli.parse_args([
                "--figures", "1",
                "--single-lr-csv", "/tmp/nonexistent.csv",
                "--output-dir", "/tmp/output",
            ])


# ---------------------------------------------------------------------------
# Contract: source-table schemas with labels, panel IDs, estimates, bounds
#           (Finding 7)
# ---------------------------------------------------------------------------

class TestSourceTableSchemas:
    """Source tables include model names, condition labels, panel IDs,
    estimates, and 95% bounds."""

    def test_figure1_source_has_panel_id(self, tmp_path):
        """Figure 1 source must have a 'figure' panel ID."""
        n_subjects = 3
        df = pd.DataFrame({
            "subject_id": [f"s{i}" for i in range(n_subjects * 4)],
            "condition": [c for c in pf.CONDITION_ORDER for _ in range(n_subjects)],
            "learning_rate": np.random.uniform(0.1, 0.5, n_subjects * 4),
            "model": "rw",
        })
        cli = _load_cli_module()
        tables_dir = tmp_path / "tables"
        src = cli._write_figure1_source(df, tables_dir)
        assert "figure" in src.columns
        assert all(src["figure"] == "figure-1")
        assert src["subject_number"].min() == 1

    def test_figure2_source_has_model_names_and_intervals(self, tmp_path):
        """Figure 2 source has model, elpd_diff, se_diff, stacking_weight."""
        comparison = pd.DataFrame({
            "model": _TEST_ALLOWLIST[:2],
            "elpd_diff": [-2.0, -1.0],
            "se_diff": [0.5, 0.6],
            "stacking_weight": [0.7, 0.3],
            "model_display": [pf.MODEL_DISPLAY_NAMES[m] for m in _TEST_ALLOWLIST[:2]],
            "y_position": [0, 1],
        })
        cli = _load_cli_module()
        tables_dir = tmp_path / "tables"
        src = cli._write_figure2_source(comparison, tables_dir)
        assert "figure" in src.columns
        assert "model" in src.columns
        assert "elpd_diff" in src.columns
        assert "stacking_weight" in src.columns

    def test_figure2_ppc_source_has_condition_labels_and_bounds(self, tmp_path):
        """Figure 2 PPC source has condition_label, trial, predicted mean, CI."""
        ppc = {
            "StableV4_2": {
                "trial_mean_pred": np.array([0.6, 0.7]),
                "trial_mean_obs": np.array([0.5, 0.8]),
                "ci_lower": np.array([0.4, 0.5]),
                "ci_upper": np.array([0.8, 0.9]),
            }
        }
        cli = _load_cli_module()
        tables_dir = tmp_path / "tables"
        src = cli._write_figure2_ppc_source(ppc, tables_dir)
        assert "condition_label" in src.columns
        assert "predicted_mean" in src.columns
        assert "ci_lower" in src.columns
        assert "ci_upper" in src.columns
        assert src["condition_label"].iloc[0] == "Stable → Stable"
        assert src["observed_p_choice_0"].iloc[0] == 0.5
        assert src["trial_number"].tolist() == [1, 2]

    def test_figure3_source_has_estimates_and_bounds(self, tmp_path):
        """Figure 3 source tables have estimates and CI bounds."""
        subject_summary = pd.DataFrame({
            "subject_idx": [0, 0],
            "condition": ["StableV4_2", "StableV1"],
            "learning_rate_pos_subject_mean": [0.3, 0.4],
            "learning_rate_neg_subject_mean": [0.1, 0.2],
        })
        population_summary = {
            "StableV4_2": {
                "learning_rate_pos_population_mean": 0.35,
                "learning_rate_pos_ci_lower": 0.2,
                "learning_rate_pos_ci_upper": 0.5,
                "learning_rate_neg_population_mean": 0.15,
                "learning_rate_neg_ci_lower": 0.1,
                "learning_rate_neg_ci_upper": 0.2,
            }
        }
        effects = {
            "learning_rate_pos": pd.DataFrame({
                "current": [0.1], "previous": [-0.05], "interaction": [0.02],
                "current_ci_lower": [0.05], "current_ci_upper": [0.15],
                "previous_ci_lower": [-0.08], "previous_ci_upper": [-0.02],
                "interaction_ci_lower": [-0.01], "interaction_ci_upper": [0.05],
            }),
        }
        cli = _load_cli_module()
        tables_dir = tmp_path / "tables"
        src = cli._write_figure3_source(subject_summary, population_summary, effects, tables_dir)

        # Subject source
        assert "figure" in src.columns
        assert "learning_rate_pos_subject_mean" in src.columns
        assert src["subject_number"].tolist() == [1, 1]

        # Verify population table was written
        pop_csv = tables_dir / "figure-3-population.csv"
        assert pop_csv.is_file()
        pop_df = pd.read_csv(pop_csv)
        assert "condition" in pop_df.columns
        assert "parameter" in pop_df.columns
        assert "estimate" in pop_df.columns

        # Verify effects table was written
        effects_csv = tables_dir / "figure-3-effects.csv"
        assert effects_csv.is_file()
        ef_df = pd.read_csv(effects_csv)
        assert "parameter" in ef_df.columns
        assert "effect" in ef_df.columns
        assert "estimate" in ef_df.columns


def _capture_closed_figure(monkeypatch):
    """Capture figures passed to ``plt.close`` without closing them."""
    import matplotlib.pyplot as plt

    captured = []
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.append(fig))
    return captured


class TestPublicationPlotStyling:
    """Requested visual conventions are applied to all publication plots."""

    @staticmethod
    def _ppc():
        return {
            cond: {
                "trial_mean_pred": np.array([0.4, 0.6]),
                "trial_mean_obs": np.array([0.5, 0.5]),
                "ci_lower": np.array([0.2, 0.3]),
                "ci_upper": np.array([0.7, 0.8]),
            }
            for cond in pf.CONDITION_ORDER
        }

    def test_palette_and_model_labels(self):
        assert pf.CONDITION_COLOURS == (
            "#4169E1", "#C23B75", "#E69F00", "#00A878",
        )
        assert all("\n" in label for label in pf.MODEL_DISPLAY_NAMES.values())
        assert all("single value" not in label for label in pf.MODEL_DISPLAY_NAMES.values())

    def test_figure1_uses_short_labels_transparent_points_and_despines(self, monkeypatch):
        subject_estimates = pd.DataFrame({
            "condition": [c for c in pf.CONDITION_ORDER for _ in range(2)],
            "learning_rate": [0.2, 0.3, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55],
        })
        captured = _capture_closed_figure(monkeypatch)

        pf.plot_figure1(subject_estimates, "unused", output_formats=())

        ax = captured[0].axes[0]
        assert tuple(captured[0].get_size_inches()) == pf.FIGURE_SIZES_INCHES["figure1"]
        assert ax.get_xlabel() == "Condition"
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "Stab\n> Stab", "Vol\n> Stab", "Stab\n> Vol", "Vol\n> Vol",
        ]
        assert all(tick.get_rotation() == 0 for tick in ax.get_xticklabels())
        assert ax.collections[0].get_alpha() == pytest.approx(0.25)
        summary_markers = [line for line in ax.lines if line.get_marker() == "o"]
        assert all(line.get_color() == pf.SUMMARY_COLOUR for line in summary_markers)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

    def test_figure2_combines_blue_model_comparison_and_condition_coloured_ppc(self, monkeypatch):
        comparison = pf.load_comparison_for_figure2(
            _make_comparison_df(_TEST_ALLOWLIST)
        )
        captured = _capture_closed_figure(monkeypatch)

        pf.plot_figure2(comparison, self._ppc(), "unused", output_formats=())

        fig = captured[0]
        ax_forest, ax_weights, *ppc_axes = fig.axes
        assert tuple(fig.get_size_inches()) == (6, 6.6)
        assert [tick.get_text() for tick in ax_forest.get_yticklabels()] == [
            pf.MODEL_DISPLAY_NAMES[model] for model in pf.FIGURE2_MODEL_ORDER
        ]
        assert ax_forest.get_xlabel() == r"$\Delta$ ELPD (lower is better)"
        model_markers = [line for line in ax_forest.lines if line.get_marker() == "o"]
        assert all(line.get_color() == pf.ACCENT_BLUE for line in model_markers)
        assert all(
            np.allclose(bar.get_facecolor()[:3], (65 / 255, 105 / 255, 225 / 255))
            for bar in ax_weights.patches
        )
        for index, ax in enumerate(ppc_axes):
            predicted = next(line for line in ax.lines if line.get_label() == "Predicted")
            assert predicted.get_color() == pf.CONDITION_COLOURS[index]
            assert predicted.get_linestyle() == "-"
            assert ax.get_ylabel() == "P(Choose left)"
            assert all("Chance" not in line.get_label() for line in ax.lines)
        assert [text.get_text() for text in ppc_axes[0].get_legend().get_texts()] == [
            "Predicted", "Observed",
        ]
        assert all(not ax.spines[side].get_visible() for side in ("top", "right")
                   for ax in fig.axes)
        panel_labels = [text for ax in fig.axes for text in ax.texts]
        assert [label.get_text() for label in panel_labels] == list("abcdef")
        assert all(label.get_fontsize() == pf.PANEL_LABEL_FONT_SIZE for label in panel_labels)
        assert all(label.get_fontweight() == pf.PANEL_LABEL_FONT_WEIGHT for label in panel_labels)
        assert not ax_weights.get_yticklabels()

    def test_figure3_uses_full_lr_labels_blue_coefficients_and_figure1_style(self, monkeypatch):
        subject_summary = pd.DataFrame({
            "subject_idx": [0, 1, 0, 1, 0, 1, 0, 1],
            "condition": [c for c in pf.CONDITION_ORDER for _ in range(2)],
            "learning_rate_pos_subject_mean": np.linspace(0.2, 0.5, 8),
            "learning_rate_neg_subject_mean": np.linspace(0.1, 0.4, 8),
        })
        population_summary = {
            cond: {
                "learning_rate_pos_population_mean": 0.3,
                "learning_rate_pos_ci_lower": 0.2,
                "learning_rate_pos_ci_upper": 0.4,
                "learning_rate_neg_population_mean": 0.25,
                "learning_rate_neg_ci_lower": 0.15,
                "learning_rate_neg_ci_upper": 0.35,
            }
            for cond in pf.CONDITION_ORDER
        }
        effects = {
            param: pd.DataFrame({
                "current": [0.1], "previous": [-0.05], "interaction": [0.02],
                "current_ci_lower": [0.05], "current_ci_upper": [0.15],
                "previous_ci_lower": [-0.1], "previous_ci_upper": [0.0],
                "interaction_ci_lower": [-0.02], "interaction_ci_upper": [0.06],
            })
            for param in pf.LEARNING_RATE_PARAMS
        }
        captured = _capture_closed_figure(monkeypatch)

        pf.plot_figure3(subject_summary, population_summary, effects, "unused", output_formats=())

        fig = captured[0]
        assert tuple(fig.get_size_inches()) == pf.FIGURE_SIZES_INCHES["figure3"]
        assert fig.axes[0].get_ylabel() == "Positive learning rate"
        assert fig.axes[2].get_ylabel() == "Negative learning rate"
        assert fig.axes[1].get_xlabel() == "Coefficient\n(effect on positive learning rate)"
        assert fig.axes[3].get_xlabel() == "Coefficient\n(effect on negative learning rate)"
        assert not fig.axes[1].get_title()
        assert not fig.axes[3].get_title()
        assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == [
            "Stab\n> Stab", "Vol\n> Stab", "Stab\n> Vol", "Vol\n> Vol",
        ]
        assert all(tick.get_rotation() == 0 for tick in fig.axes[0].get_xticklabels())
        summary_markers = [line for ax in (fig.axes[0], fig.axes[2]) for line in ax.lines
                           if line.get_marker() == "o"]
        assert all(line.get_color() == pf.SUMMARY_COLOUR for line in summary_markers)
        coefficient_markers = [
            line for ax in (fig.axes[1], fig.axes[3]) for line in ax.lines
            if line.get_marker() == "o"
        ]
        assert all(line.get_color() == pf.ACCENT_BLUE for line in coefficient_markers)
        panel_labels = [text for ax in fig.axes for text in ax.texts]
        assert [label.get_text() for label in panel_labels] == list("abcd")
        assert all(label.get_fontweight() == pf.PANEL_LABEL_FONT_WEIGHT for label in panel_labels)
        assert all(not ax.spines[side].get_visible() for ax in fig.axes
                   for side in ("top", "right"))


# ---------------------------------------------------------------------------
# Contract: CLI smoke test with synthetic inputs  (Findings 1, 3, 6)
# ---------------------------------------------------------------------------

class TestPublicationCliSmoke:
    """End-to-end smoke test of the CLI using synthetic fixtures.

    Uses the real ArviZ posterior.nc format + metadata.json."""

    def _write_single_lr_csv(self, path: Path):
        """Write a minimal single-LR subject estimates CSV."""
        conditions = ["StableV4_2", "StableV1", "VolatileV2", "VolatileV3_2"]
        rows = []
        for cond in conditions:
            for i in range(3):
                rows.append({
                    "subject_id": f"{cond}_subj_{i}",
                    "subject_idx": i,
                    "condition": cond,
                    "model": "rescorla_wagner_single_value",
                    "block_type": "Stable" if "Stable" in cond else "Volatile",
                    "version": cond.split("V")[-1],
                    "block_number": 2,
                    "previous_block_type": "Stable" if "StableV4" in cond or "VolatileV2" in cond else "Volatile",
                    "converged": True,
                    "learning_rate": 0.1 + 0.05 * i,
                    "temperature": 0.5,
                })
        df = pd.DataFrame(rows)
        csv_path = path / "subject_params.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def _write_comparison_table(self, path: Path):
        """Write a model_selection_compare.csv with ArviZ az.compare format."""
        models = _TEST_ALLOWLIST + ["rescorla_wagner", "dual_lr"]
        rng = np.random.default_rng(42)
        n_models = len(models)
        df = pd.DataFrame({
            "model": models,
            "rank": list(range(n_models)),
            "elpd_diff": -np.abs(rng.normal(0, 2, n_models)),
            "dse": rng.exponential(1, n_models),
            "weight": rng.dirichlet(np.ones(n_models)),
        })
        path.mkdir(parents=True, exist_ok=True)
        table_path = path / "model_selection_compare.csv"
        df.to_csv(table_path, index=False)
        return table_path

    def _write_posterior_nc_and_metadata(self, path: Path, n_subjects: int = 12):
        """Write a synthetic posterior.nc and metadata.json using the real
        ArviZ DataTree NetCDF format."""
        n_chains = 2
        n_draws = 10
        rng = np.random.default_rng(42)
        model_name = pf.WINNING_MODEL

        # Build per-condition indices (3 subjects per condition)
        subjects_per_cond = n_subjects // 4
        cond_indices: Dict[str, List[int]] = {}
        condition_labels: List[str] = []
        for i, cond in enumerate(pf.CONDITION_ORDER):
            start = i * subjects_per_cond
            end = start + subjects_per_cond
            cond_indices[cond] = list(range(start, end))
            condition_labels.extend([cond] * subjects_per_cond)

        posterior_data: Dict[str, np.ndarray] = {}
        # Group-level params
        for param in pf.LEARNING_RATE_PARAMS:
            for site in pf.GROUP_SITES:
                key = f"{model_name}_{param}{site}"
                posterior_data[key] = rng.normal(0, 0.5, (n_chains, n_draws))
        # Temperature group params
        for site in pf.GROUP_SITES:
            key = f"{model_name}_temperature{site}"
            posterior_data[key] = rng.normal(0, 0.5, (n_chains, n_draws))
        # Subject-level params
        for param in pf.LEARNING_RATE_PARAMS:
            key = f"{model_name}_{param}_subj"
            posterior_data[key] = rng.uniform(0.1, 0.6, (n_chains, n_draws, n_subjects))
        key = f"{model_name}_temperature_subj"
        posterior_data[key] = rng.uniform(0.2, 0.5, (n_chains, n_draws, n_subjects))

        dt = az.from_dict(posterior=posterior_data)
        nc_path = path / "posterior.nc"
        dt.to_netcdf(nc_path)

        # Metadata
        contexts = build_context_regressors(list(pf.CONDITION_ORDER))
        subject_ids = []
        subject_labels = []
        context_rows = []
        for idx, cond in enumerate(pf.CONDITION_ORDER):
            for si in cond_indices[cond]:
                sid = f"subj_{si:03d}"
                subject_ids.append(sid)
                subject_labels.append(f"{cond}__{sid}")
                context_rows.append(contexts[idx].tolist())

        metadata = {
            "model": model_name,
            "conditions": list(pf.CONDITION_ORDER),
            "n_subjects": n_subjects,
            "n_conditions": 4,
            "subject_ids": subject_ids,
            "subject_labels": subject_labels,
            "condition_labels": condition_labels,
            "context": context_rows,
        }
        meta_path = path / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        return nc_path, metadata, cond_indices

    @pytest.mark.rendering
    def test_cli_smoke_all_figures(self, tmp_path: Path, monkeypatch):
        """Run the CLI with all synthetic fixtures and verify outputs for
        all three figures, including posterior.nc + metadata.json serialization."""
        # Arrange fixtures
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()

        single_lr_csv = self._write_single_lr_csv(fixtures)
        comparison_dir = fixtures / "comparison_tables"
        self._write_comparison_table(comparison_dir)
        posterior_dir = fixtures / "posterior"
        posterior_dir.mkdir()
        self._write_posterior_nc_and_metadata(posterior_dir, n_subjects=12)
        # Prepared data for all eight conditions, following the task design
        # (task performance uses all eight; Figure 2 uses the second blocks).
        prepared_dir = fixtures / "prepared"
        prepared_dir.mkdir()
        for cond, data in _performance_dataset(n_participants=3).items():
            np.savez_compressed(
                prepared_dir / f"{cond}.npz", outcome_coding="reward_safe", **data
            )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cli = _load_cli_module()
        monkeypatch.setattr(cli, "ensure_dm_sans_font", lambda: None)
        args = cli.parse_args([
            "--single-lr-csv", str(single_lr_csv),
            "--comparison-dir", str(comparison_dir),
            "--posterior-dir", str(posterior_dir),
            "--prepared-data-dir", str(prepared_dir),
            "--output-dir", str(output_dir),
        ])

        cli.main(args)

        # Verify output files
        figures_dir = output_dir / "figures"
        tables_dir = output_dir / "tables"
        assert figures_dir.is_dir()
        assert tables_dir.is_dir()

        expected_figures = [
            "figure-1-single-learning-rate",
            "figure-2-model-comparison-and-ppc",
            "figure-3-learning-rates",
        ]
        for fig_name in expected_figures:
            for ext in ["pdf", "svg", "png"]:
                assert (figures_dir / f"{fig_name}.{ext}").is_file(), (
                    f"Missing {fig_name}.{ext}"
                )

        expected_tables = [
            "task-performance-source.csv",
            "task-performance-summary.csv",
            "task-performance-schedule-check.csv",
            "figure-1-source.csv",
            "figure-1-comparisons.csv",
            "figure-2-source.csv",
            "figure-2-ppc-source.csv",
            "figure-3-source.csv",
            "figure-3-population.csv",
            "figure-3-effects.csv",
            "figure-3-coefficients.csv",
        ]
        for tbl in expected_tables:
            assert (tables_dir / tbl).is_file(), f"Missing source table {tbl}"

        # Spot-check source table contents
        fig2 = pd.read_csv(tables_dir / "figure-2-source.csv")
        assert "model" in fig2.columns
        assert "se_diff" in fig2.columns
        assert "stacking_weight" in fig2.columns
        assert set(fig2["model"].unique()) == set(_TEST_ALLOWLIST)

    def test_cli_graceful_missing_input_error(self, tmp_path: Path, monkeypatch):
        """CLI should raise clear error when required inputs are missing."""
        cli = _load_cli_module()
        monkeypatch.setattr(cli, "ensure_dm_sans_font", lambda: None)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises((FileNotFoundError, SystemExit)):
            cli.main(cli.parse_args([
                "--single-lr-csv", str(tmp_path / "nonexistent.csv"),
                "--output-dir", str(output_dir),
            ]))


# ---------------------------------------------------------------------------
# Contract: prepared .npz with object-dtype subject_ids loads without
#           pickle (npz-pickle-loading fix)
# ---------------------------------------------------------------------------

class TestPreparedDataLoadingNoPickle:
    """Figure 3 prepared-data loading must work with object-dtype
    ``subject_ids`` and must not enable pickle."""

    def test_loads_required_arrays_only(self, tmp_path: Path):
        """Only ``outcomes_0``, ``outcomes_1``, and ``choices`` are loaded."""
        n_subjects = 3
        n_trials = 10
        rng = np.random.default_rng(42)

        # Write a prepared .npz with object-dtype subject_ids
        npz_path = tmp_path / "test_condition.npz"
        np.savez_compressed(
            npz_path,
            choices=rng.binomial(1, 0.5, (n_subjects, n_trials)),
            outcomes_0=rng.binomial(1, 0.5, (n_subjects, n_trials)),
            outcomes_1=1 - rng.binomial(1, 0.5, (n_subjects, n_trials)),
            subject_ids=np.asarray(
                [f"subj_{i}" for i in range(n_subjects)], dtype=object
            ),
        )

        cli = _load_cli_module()
        prepared_data = cli._load_prepared_ppc_data(npz_path)

        # Must contain exactly the three required arrays
        assert set(prepared_data.keys()) == {"outcomes_0", "outcomes_1", "choices"}
        assert prepared_data["outcomes_0"].shape == (n_subjects, n_trials)
        assert prepared_data["outcomes_1"].shape == (n_subjects, n_trials)
        assert prepared_data["choices"].shape == (n_subjects, n_trials)

    def test_does_not_enable_pickle(self, tmp_path: Path, monkeypatch):
        """``np.load`` is not called with ``allow_pickle=True``."""
        n_subjects = 3
        n_trials = 5
        rng = np.random.default_rng(42)

        npz_path = tmp_path / "test_condition.npz"
        np.savez_compressed(
            npz_path,
            choices=rng.binomial(1, 0.5, (n_subjects, n_trials)),
            outcomes_0=rng.binomial(1, 0.5, (n_subjects, n_trials)),
            outcomes_1=1 - rng.binomial(1, 0.5, (n_subjects, n_trials)),
            subject_ids=np.asarray(
                [f"subj_{i}" for i in range(n_subjects)], dtype=object
            ),
        )

        original_load = np.load

        def tracking_load(*args, **kwargs):
            assert kwargs.get("allow_pickle") is not True, (
                "np.load called with allow_pickle=True"
            )
            return original_load(*args, **kwargs)

        monkeypatch.setattr(np, "load", tracking_load)

        cli = _load_cli_module()
        prepared_data = cli._load_prepared_ppc_data(npz_path)
        assert set(prepared_data.keys()) == {"outcomes_0", "outcomes_1", "choices"}

    def test_reads_scalar_coding_metadata_without_subject_ids(self, tmp_path: Path):
        npz_path = tmp_path / "reward_safe.npz"
        choices = np.array([[1, 0]])
        outcomes_0 = np.array([[1, 0]])
        outcomes_1 = np.array([[0, 1]])
        np.savez_compressed(
            npz_path,
            choices=choices,
            outcomes_0=outcomes_0,
            outcomes_1=outcomes_1,
            outcome_coding="reward_safe",
            subject_ids=np.asarray(["s1"], dtype=object),
        )
        cli = _load_cli_module()
        loaded = cli._load_prepared_ppc_data(npz_path)
        assert loaded["outcome_coding"] == "reward_safe"
        assert np.array_equal(loaded["choices"], choices)
        from volatility_generalisation.modelling_code.data_preparation import (
            normalize_to_reward_coded,
        )
        normalized = normalize_to_reward_coded(
            {key: loaded[key] for key in ("outcomes_0", "outcomes_1", "choices")},
            {"outcome_coding": loaded["outcome_coding"]},
        )
        assert np.array_equal(normalized["outcomes_0"], outcomes_0)
        assert np.array_equal(normalized["outcomes_1"], outcomes_1)
        assert np.array_equal(normalized["choices"], choices)

    def test_normalization_without_metadata_flips_only_outcomes(self):
        from volatility_generalisation.modelling_code.data_preparation import (
            normalize_to_reward_coded,
        )

        choices = np.array([[1, 0]])
        data = {
            "outcomes_0": np.array([[0, 1]]),
            "outcomes_1": np.array([[1, 0]]),
            "choices": choices.copy(),
        }
        normalized = normalize_to_reward_coded(data, metadata=None)
        assert np.array_equal(normalized["outcomes_0"], [[1, 0]])
        assert np.array_equal(normalized["outcomes_1"], [[0, 1]])
        assert np.array_equal(normalized["choices"], choices)

    def test_ppc_rejects_mismatched_prepared_shapes(self):
        posterior = {
            "learning_rate_pos": np.full((2, 1), 0.3),
            "learning_rate_neg": np.full((2, 1), 0.2),
            "temperature": np.full((2, 1), 0.5),
        }
        with pytest.raises(ValueError, match="matching shapes"):
            pf.compute_ppc_summary(
                posterior,
                {
                    "outcomes_0": np.zeros((1, 3)),
                    "outcomes_1": np.zeros((1, 2)),
                    "choices": np.zeros((1, 3)),
                },
            )


# ---------------------------------------------------------------------------
# Negative tests for metadata, condition, and subject-count mismatches
# (Finding 5)
# ---------------------------------------------------------------------------

class TestNegativeInputValidation:
    """Focused negative tests for missing/malformed metadata, missing
    conditions, and subject-count mismatches."""

    def test_load_posterior_missing_metadata(self, tmp_path):
        """load_posterior_and_metadata raises FileNotFoundError when
        metadata.json is missing."""
        # Create a valid posterior.nc
        posterior_dir = tmp_path / "posterior"
        posterior_dir.mkdir()
        n_subjects = 8
        n_chains = 2
        n_draws = 5
        rng = np.random.default_rng(42)
        model = pf.WINNING_MODEL
        posterior_data: Dict[str, np.ndarray] = {}
        for param in pf.LEARNING_RATE_PARAMS:
            for site in pf.GROUP_SITES:
                posterior_data[f"{model}_{param}{site}"] = rng.normal(0, 0.5, (n_chains, n_draws))
        for site in pf.GROUP_SITES:
            posterior_data[f"{model}_temperature{site}"] = rng.normal(0, 0.5, (n_chains, n_draws))
        for param in pf.LEARNING_RATE_PARAMS:
            posterior_data[f"{model}_{param}_subj"] = rng.uniform(0.1, 0.6, (n_chains, n_draws, n_subjects))
        posterior_data[f"{model}_temperature_subj"] = rng.uniform(0.2, 0.5, (n_chains, n_draws, n_subjects))
        dt = az.from_dict(posterior=posterior_data)
        dt.to_netcdf(posterior_dir / "posterior.nc")

        # No metadata.json
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            pf.load_posterior_and_metadata(posterior_dir / "posterior.nc")

    def test_load_posterior_malformed_metadata(self, tmp_path):
        """load_posterior_and_metadata raises KeyError when metadata is
        missing required fields."""
        posterior_dir = tmp_path / "posterior"
        posterior_dir.mkdir()
        n_subjects = 8
        n_chains = 2
        n_draws = 5
        rng = np.random.default_rng(42)
        model = pf.WINNING_MODEL
        posterior_data: Dict[str, np.ndarray] = {}
        for param in pf.LEARNING_RATE_PARAMS:
            for site in pf.GROUP_SITES:
                posterior_data[f"{model}_{param}{site}"] = rng.normal(0, 0.5, (n_chains, n_draws))
        for site in pf.GROUP_SITES:
            posterior_data[f"{model}_temperature{site}"] = rng.normal(0, 0.5, (n_chains, n_draws))
        for param in pf.LEARNING_RATE_PARAMS:
            posterior_data[f"{model}_{param}_subj"] = rng.uniform(0.1, 0.6, (n_chains, n_draws, n_subjects))
        posterior_data[f"{model}_temperature_subj"] = rng.uniform(0.2, 0.5, (n_chains, n_draws, n_subjects))
        dt = az.from_dict(posterior=posterior_data)
        dt.to_netcdf(posterior_dir / "posterior.nc")

        # Metadata missing 'condition_labels'
        with open(posterior_dir / "metadata.json", "w") as f:
            json.dump({"some_key": "some_value"}, f)

        with pytest.raises(KeyError, match="condition_labels"):
            pf.load_posterior_and_metadata(posterior_dir / "posterior.nc")

    def test_extract_subject_params_missing_condition(self):
        """extract_subject_params should handle missing conditions
        gracefully (they will simply have no entries)."""
        posterior_ds, _, cond_names = _build_fake_posterior(n_subjects_total=8)
        # Request an unknown condition
        cond_indices = {"UnknownCondition": [0, 1, 2]}
        subj_by_cond = pf.extract_subject_params(posterior_ds, cond_indices)
        assert "UnknownCondition" in subj_by_cond
        # The condition will have empty (0-subject) arrays for each param
        for param in pf.LEARNING_RATE_PARAMS:
            assert param in subj_by_cond["UnknownCondition"]
            assert subj_by_cond["UnknownCondition"][param].shape[1] == 3

    def test_extract_group_params_missing_required_site(self):
        """extract_group_params raises ValueError when a required group
        site is missing."""
        # Create a posterior deliberately missing one group site
        n_subjects = 8
        n_chains = 2
        n_draws = 5
        rng = np.random.default_rng(42)
        model = pf.WINNING_MODEL
        posterior_data: Dict[str, np.ndarray] = {}
        # Only add some group params — skip _beta_current_vol
        for param in pf.LEARNING_RATE_PARAMS:
            for site in ["_mu", "_beta_previous_vol",
                         "_beta_current_previous_interaction", "_subject_sigma"]:
                posterior_data[f"{model}_{param}{site}"] = rng.normal(0, 0.5, (n_chains, n_draws))
        # Subject-level params
        for param in pf.LEARNING_RATE_PARAMS:
            posterior_data[f"{model}_{param}_subj"] = rng.uniform(0.1, 0.6, (n_chains, n_draws, n_subjects))
        posterior_data[f"{model}_temperature_subj"] = rng.uniform(0.2, 0.5, (n_chains, n_draws, n_subjects))

        dt = az.from_dict(posterior=posterior_data)
        posterior_ds = dt["posterior"]

        with pytest.raises(ValueError, match="beta_current_vol"):
            pf.extract_group_params(posterior_ds)

    def test_condition_order_consistency(self):
        """Condition order and context regressors must remain consistent."""
        order = pf.condition_order()
        assert len(order) == 4
        # StableV4_2 must be first (Stable→Stable)
        assert order[0] == "StableV4_2"
        # VolatileV3_2 must be last (Volatile→Volatile)
        assert order[-1] == "VolatileV3_2"
        # Verify context regressor ordering matches condition order
        contexts = build_context_regressors(order)
        assert contexts.shape == (4, 3)
        # StableV4_2 should have current_vol=-0.5, previous_vol=-0.5
        assert contexts[0, 0] == pytest.approx(-0.5)
        assert contexts[0, 1] == pytest.approx(-0.5)
        # VolatileV3_2 should have current_vol=0.5, previous_vol=0.5
        assert contexts[3, 0] == pytest.approx(0.5)
        assert contexts[3, 1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Contract: task performance (optimal choices and safe outcomes)
# ---------------------------------------------------------------------------


def _design_outcomes(block_type, first_better_options, interval=None):
    """Build reward-coded outcomes that follow a schedule exactly.

    In each run of the schedule the better ball is safe on the first 80% of
    trials and the worse ball on the rest.

    Args:
        block_type: ``"Stable"`` or ``"Volatile"``; sets the run length.
        first_better_options: Starting better option for each participant.
        interval: Run length to use instead of the design's (for building
            data that does not follow the design).

    Returns:
        Tuple of ``outcomes_0``, ``outcomes_1`` and the true better option,
        each ``(n_participants, 120)``.
    """
    interval = interval or cm.REVERSAL_INTERVAL_TRIALS[block_type]
    trials = np.arange(cm.BLOCK_TRIAL_COUNT)
    position = trials % interval
    better_safe = (position < 0.8 * interval).astype(float)
    better = np.stack([
        ((trials // interval + first) % 2) for first in first_better_options
    ])
    outcomes_0 = np.where(better == 0, better_safe, 1.0 - better_safe)
    return outcomes_0, 1.0 - outcomes_0, better


def _performance_dataset(seed=0, n_participants=2):
    """Build prepared data for all eight conditions following the task design.

    In each task version participant ``a`` starts with option 0 better and
    always picks the better ball. The others (``b``, ``c``, ...) alternate
    between option 1 and option 0 starting as the better ball and choose at
    random.

    Args:
        seed: Seed for the random choices.
        n_participants: Number of participants in each task version.

    Returns:
        Dictionary of prepared data keyed by condition.
    """
    rng = np.random.default_rng(seed)
    letters = "abcdefghij"[:n_participants]
    dataset = {}
    for condition, metadata in cm.CONDITION_METADATA.items():
        version = str(metadata["version"]).lower()
        outcomes_0, outcomes_1, better = _design_outcomes(
            str(metadata["block_type"]), [i % 2 for i in range(n_participants)]
        )
        choices = rng.integers(0, 2, (n_participants, cm.BLOCK_TRIAL_COUNT)).astype(float)
        choices[0] = 1.0 - better[0]  # choices = 1 means option 0: picks the better ball
        dataset[condition] = {
            "choices": choices,
            "outcomes_0": outcomes_0,
            "outcomes_1": outcomes_1,
            "subject_ids": np.array(
                [f"{version}_{letter}" for letter in letters], dtype=object
            ),
        }
    return dataset


class TestTaskPerformance:
    """Task performance is scored against the rebuilt design schedule."""

    def test_better_option_by_trial_follows_design(self):
        stable = cm.better_option_by_trial("Stable", 0)
        np.testing.assert_array_equal(stable, [0] * 60 + [1] * 60)
        np.testing.assert_array_equal(cm.better_option_by_trial("Stable", 1), 1 - stable)
        volatile = cm.better_option_by_trial("Volatile", 0)
        np.testing.assert_array_equal(volatile, ([0] * 20 + [1] * 20) * 3)
        np.testing.assert_array_equal(cm.better_option_by_trial("Volatile", 1), 1 - volatile)
        with pytest.raises(ValueError, match="Unknown block type"):
            cm.better_option_by_trial("Mixed", 0)
        with pytest.raises(ValueError, match="0 or 1"):
            cm.better_option_by_trial("Stable", 2)

    def test_rebuild_recovers_starting_option(self):
        outcomes_0, outcomes_1, better = _design_outcomes("Volatile", [0, 1, 0])
        result = pf.rebuild_better_option(
            outcomes_0, outcomes_1, "Volatile", "VolatileV1", ["p1", "p2", "p3"]
        )
        np.testing.assert_array_equal(result["first_better_option"], [0, 1, 0])
        np.testing.assert_array_equal(result["better_option"], better)
        np.testing.assert_allclose(result["better_ball_safe_rate"], 0.8)
        np.testing.assert_allclose(result["lowest_segment_safe_rate"], 0.8)

    def test_rebuild_rejects_schedule_that_does_not_fit(self):
        # Reversals every 30 trials: the volatile design fits only half the trials.
        outcomes_0, outcomes_1, _ = _design_outcomes("Volatile", [0, 1], interval=30)
        with pytest.raises(ValueError, match="does not fit condition 'VolatileV2'.*p1"):
            pf.rebuild_better_option(
                outcomes_0, outcomes_1, "Volatile", "VolatileV2", ["p1", "p2"]
            )

    def test_rebuild_rejects_wrong_trial_count(self):
        outcomes_0, outcomes_1, _ = _design_outcomes("Stable", [0])
        with pytest.raises(ValueError, match="100 trials"):
            pf.rebuild_better_option(
                outcomes_0[:, :100], outcomes_1[:, :100], "Stable", "StableV1", ["p1"]
            )

    def test_measures_are_hand_computed_and_pin_choice_coding(self):
        dataset = _performance_dataset()
        # StableV1, v1_a (option 0 better on trials 1-60): option 0 for trials
        # 1-90, then option 1. Optimal on 1-60 and 91-120 = 90 trials; a flipped
        # choice coding would give 30.
        dataset["StableV1"]["choices"][0] = np.r_[np.ones(90), np.zeros(30)]
        # VolatileV1, v1_a: always option 0, the better ball on half the trials.
        dataset["VolatileV1"]["choices"][0] = np.ones(120)

        participants, schedule_check = pf.compute_task_performance(dataset)

        blocks = schedule_check.set_index(["subject_id", "condition"])
        assert blocks.loc[("v1_a", "StableV1"), "n_optimal"] == 90
        # Safe: option 0 safe on trials 1-48; option 1 safe on trials 91-108.
        assert blocks.loc[("v1_a", "StableV1"), "n_safe"] == 66
        assert blocks.loc[("v1_a", "VolatileV1"), "n_optimal"] == 60
        assert blocks.loc[("v1_a", "VolatileV1"), "n_safe"] == 60
        assert blocks.loc[("v1_b", "StableV1"), "first_better_option_number"] == 2

        person = participants.set_index("subject_id").loc["v1_a"]
        assert person["n_blocks"] == 2
        assert person["n_trials"] == 240
        assert person["proportion_optimal"] == pytest.approx(150 / 240)
        assert person["proportion_safe"] == pytest.approx(126 / 240)
        assert person["best_possible_safe"] == pytest.approx(0.8)
        # Participants who always pick the better ball score 1.0 optimal.
        assert participants.set_index("subject_id").loc["v2_a", "proportion_optimal"] == 1.0

    def test_missing_condition_and_bad_participant_ids_raise(self):
        dataset = _performance_dataset()
        del dataset["StableV4"]
        with pytest.raises(ValueError, match="missing=\\['StableV4'\\]"):
            pf.compute_task_performance(dataset)

        dataset = _performance_dataset()
        dataset["StableV2"]["subject_ids"] = np.array(["only_one"], dtype=object)
        with pytest.raises(ValueError, match="1 participant IDs for 2 rows"):
            pf.compute_task_performance(dataset)

        dataset = _performance_dataset()
        dataset["StableV2"]["subject_ids"] = np.array(["same", "same"], dtype=object)
        with pytest.raises(ValueError, match="repeated participant IDs"):
            pf.compute_task_performance(dataset)

    def test_summary_matches_independent_scipy_calls(self):
        from scipy import stats

        participants, _ = pf.compute_task_performance(_performance_dataset(seed=3))
        summary = pf.summarise_task_performance(participants).set_index("measure")

        # One row per participant, not per block.
        assert len(participants) == 8
        for measure, column in (("optimal_choices", "proportion_optimal"),
                                ("safe_outcomes", "proportion_safe")):
            values = participants[column].to_numpy()
            row = summary.loc[measure]
            test = stats.ttest_1samp(values, 0.5)
            half_width = stats.t.ppf(0.975, 7) * values.std(ddof=1) / np.sqrt(8)
            assert row["n_participants"] == 8
            assert row["df"] == 7
            assert row["mean"] == pytest.approx(values.mean())
            assert row["sd"] == pytest.approx(values.std(ddof=1))
            assert row["ci_lower"] == pytest.approx(values.mean() - half_width)
            assert row["ci_upper"] == pytest.approx(values.mean() + half_width)
            assert row["t"] == pytest.approx(test.statistic)
            assert row["p_value"] == pytest.approx(test.pvalue)

        expected_above = sum(
            stats.binomtest(int(k), int(n), 0.5, alternative="greater").pvalue < 0.05
            for k, n in zip(participants["n_optimal"], participants["n_trials"])
        )
        assert summary.loc["optimal_choices", "n_above_chance"] == expected_above
        assert summary.loc["optimal_choices", "percent_above_chance"] == pytest.approx(
            100 * expected_above / 8
        )
        assert summary.loc["safe_outcomes", "best_possible"] == pytest.approx(0.8)

    def test_loader_honours_outcome_coding(self, tmp_path):
        cli = _load_cli_module()
        data = _performance_dataset()["StableV1"]
        reward_path = tmp_path / "reward.npz"
        np.savez_compressed(reward_path, outcome_coding="reward_safe", **data)
        explosion_path = tmp_path / "explosion.npz"
        np.savez_compressed(
            explosion_path,
            outcomes_0=1.0 - data["outcomes_0"],
            outcomes_1=1.0 - data["outcomes_1"],
            choices=data["choices"],
            subject_ids=data["subject_ids"],
        )

        from_reward = cli._load_prepared_performance_data(reward_path)
        from_explosion = cli._load_prepared_performance_data(explosion_path)

        for loaded in (from_reward, from_explosion):
            np.testing.assert_array_equal(loaded["outcomes_0"], data["outcomes_0"])
            np.testing.assert_array_equal(loaded["outcomes_1"], data["outcomes_1"])
            assert list(loaded["subject_ids"]) == ["v1_a", "v1_b"]

    def test_step_writes_tables(self, tmp_path):
        cli = _load_cli_module()
        prepared_dir = tmp_path / "prepared"
        tables_dir = tmp_path / "tables"
        for directory in (prepared_dir, tables_dir):
            directory.mkdir()
        for condition, data in _performance_dataset().items():
            np.savez_compressed(
                prepared_dir / f"{condition}.npz", outcome_coding="reward_safe", **data
            )

        summary = cli._make_task_performance_outputs(prepared_dir, tables_dir)

        assert set(summary["measure"]) == {"optimal_choices", "safe_outcomes"}
        source = pd.read_csv(tables_dir / "task-performance-source.csv")
        assert len(source) == 8
        assert source["subject_number"].tolist() == list(range(1, 9))
        assert len(pd.read_csv(tables_dir / "task-performance-summary.csv")) == 2
        assert len(pd.read_csv(tables_dir / "task-performance-schedule-check.csv")) == 16

        (prepared_dir / "StableV4_2.npz").unlink()
        with pytest.raises(FileNotFoundError, match="StableV4_2"):
            cli._make_task_performance_outputs(prepared_dir, tables_dir)


class TestBundledFont:
    """The DM Sans font ships with the package and is used without the internet."""

    def test_bundled_font_matches_recorded_hash(self):
        """The bundled file is the recorded DM Sans file, with its licence."""
        import hashlib

        font_dir = pf.DM_SANS_FONT_PATH.parent
        font_bytes = pf.DM_SANS_FONT_PATH.read_bytes()
        assert hashlib.sha256(font_bytes).hexdigest() == pf.DM_SANS_FONT_SHA256
        assert pf.DM_SANS_FONT_SHA256 in (font_dir / "README.md").read_text()
        assert "SIL Open Font License" in (font_dir / "OFL.txt").read_text()

    def test_font_set_up_without_internet(self, monkeypatch):
        """With every network connection blocked, the bundled file is registered."""
        import socket
        import urllib.request

        import matplotlib
        import matplotlib.font_manager as font_manager

        def no_network(*args, **kwargs):
            """Fail any attempt to open a network connection.

            Args:
                *args: Ignored.
                **kwargs: Ignored.

            Raises:
                AssertionError: Always.
            """
            raise AssertionError("the font set-up tried to use the network")

        monkeypatch.setattr(urllib.request, "urlopen", no_network)
        monkeypatch.setattr(socket, "create_connection", no_network)
        monkeypatch.setattr(socket.socket, "connect", no_network)

        with matplotlib.rc_context():
            pf.ensure_dm_sans_font()
            assert matplotlib.rcParams["font.family"] == [pf.DM_SANS_FONT_NAME]
            found = font_manager.findfont(
                pf.DM_SANS_FONT_NAME, fallback_to_default=False
            )
        assert Path(found).resolve() == pf.DM_SANS_FONT_PATH

    def test_missing_font_file_raises_without_fallback(self, tmp_path):
        """A missing font file stops figure set-up and leaves the font unchanged."""
        import matplotlib

        with matplotlib.rc_context():
            family_before = list(matplotlib.rcParams["font.family"])
            with pytest.raises(RuntimeError, match="font file not found"):
                pf.ensure_dm_sans_font(font_path=tmp_path / "dm-sans.ttf")
            assert list(matplotlib.rcParams["font.family"]) == family_before
