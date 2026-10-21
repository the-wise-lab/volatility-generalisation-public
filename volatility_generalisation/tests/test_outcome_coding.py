"""Tests for outcome coding helpers and conventions."""

import jax.numpy as jnp
import numpy as np
import pandas as pd

from volatility_generalisation.modelling_code.data_preparation import (
    OUTCOME_CODING_REWARD_SAFE,
    OUTCOME_CODING_EXPLOSION,
    OUTCOME_CODING_KEY,
    convert_empirical_to_reward_coded,
    normalize_to_reward_coded,
    prepare_complementary_data,
    prepare_data_for_fitting,
)


# ── Helper tests ────────────────────────────────────────────────────


def test_convert_exploded_true_becomes_model_outcome_zero():
    """Raw exploded=True (explosion-coded) -> model outcome 0 (reward-coded)."""
    explosion_data = {
        "outcomes_0": np.array([[1.0]]),
        "outcomes_1": np.array([[0.0]]),
        "choices": np.array([[1.0]]),
    }
    reward_data = convert_empirical_to_reward_coded(explosion_data)
    assert reward_data["outcomes_0"][0, 0] == 0.0
    assert reward_data["outcomes_1"][0, 0] == 1.0
    assert reward_data["choices"][0, 0] == 1.0


def test_convert_exploded_false_becomes_model_outcome_one():
    """Raw exploded=False (safe) -> model outcome 1 (reward-coded)."""
    explosion_data = {
        "outcomes_0": np.array([[0.0]]),
        "outcomes_1": np.array([[1.0]]),
        "choices": np.array([[1.0]]),
    }
    reward_data = convert_empirical_to_reward_coded(explosion_data)
    assert reward_data["outcomes_0"][0, 0] == 1.0
    assert reward_data["outcomes_1"][0, 0] == 0.0
    assert reward_data["choices"][0, 0] == 1.0


def test_convert_preserves_complementarity():
    """After conversion, outcomes_0 + outcomes_1 == 1 for each trial."""
    rng = np.random.default_rng(0)
    n_subjects, n_trials = 5, 20
    explosion_data = {
        "outcomes_0": rng.integers(0, 2, size=(n_subjects, n_trials)).astype(float),
        "outcomes_1": None,
        "choices": rng.integers(0, 2, size=(n_subjects, n_trials)).astype(float),
    }
    explosion_data["outcomes_1"] = 1.0 - explosion_data["outcomes_0"]
    reward_data = convert_empirical_to_reward_coded(explosion_data)
    total = reward_data["outcomes_0"] + reward_data["outcomes_1"]
    assert np.allclose(total, 1.0), "Complementarity violated after conversion"


def test_convert_does_not_mutate_input():
    """Helper returns a new dict; input arrays are unchanged."""
    original = {
        "outcomes_0": np.array([[1.0]]),
        "outcomes_1": np.array([[0.0]]),
        "choices": np.array([[1.0]]),
    }
    copy_before = {k: v.copy() for k, v in original.items()}
    _ = convert_empirical_to_reward_coded(original)
    for key in original:
        assert np.array_equal(original[key], copy_before[key]), (
            f"Input array '{key}' was mutated"
        )


def test_convert_idempotent_for_non_outcome_keys():
    """Keys other than outcomes_0 and outcomes_1 are passed through unchanged."""
    data = {
        "outcomes_0": np.array([[1.0]]),
        "outcomes_1": np.array([[0.0]]),
        "choices": np.array([[1.0]]),
        "subject_ids": np.array(["s1"]),
    }
    result = convert_empirical_to_reward_coded(data)
    assert list(result["subject_ids"]) == ["s1"]


# ── normalize_to_reward_coded tests ─────────────────────────────────


def test_normalize_with_reward_metadata_does_not_flip():
    """When metadata says reward_safe, data is returned unchanged."""
    reward_data = {
        "outcomes_0": np.array([[1.0]]),
        "outcomes_1": np.array([[0.0]]),
        "choices": np.array([[1.0]]),
    }
    metadata = {OUTCOME_CODING_KEY: OUTCOME_CODING_REWARD_SAFE}
    result = normalize_to_reward_coded(reward_data, metadata)
    assert result["outcomes_0"][0, 0] == 1.0
    assert result["outcomes_1"][0, 0] == 0.0


def test_normalize_without_metadata_converts_explosion_coding():
    """Without metadata, data is converted from explosion to reward coding."""
    explosion_coded_data = {
        "outcomes_0": np.array([[1.0]]),
        "outcomes_1": np.array([[0.0]]),
        "choices": np.array([[1.0]]),
    }
    result = normalize_to_reward_coded(explosion_coded_data, metadata=None)
    assert result["outcomes_0"][0, 0] == 0.0
    assert result["outcomes_1"][0, 0] == 1.0


def test_normalize_with_explosion_metadata_converts():
    """When metadata says explosion, conversion is applied."""
    explosion_data = {
        "outcomes_0": np.array([[1.0]]),
        "outcomes_1": np.array([[0.0]]),
        "choices": np.array([[1.0]]),
    }
    metadata = {OUTCOME_CODING_KEY: OUTCOME_CODING_EXPLOSION}
    result = normalize_to_reward_coded(explosion_data, metadata)
    assert result["outcomes_0"][0, 0] == 0.0


def test_normalize_is_idempotent_on_reward_data():
    """Calling normalize twice on reward-coded data does not double-flip."""
    reward_data = {
        "outcomes_0": np.array([[1.0]]),
        "outcomes_1": np.array([[0.0]]),
        "choices": np.array([[1.0]]),
    }
    metadata = {OUTCOME_CODING_KEY: OUTCOME_CODING_REWARD_SAFE}
    pass1 = normalize_to_reward_coded(reward_data, metadata)
    pass2 = normalize_to_reward_coded(pass1, metadata)
    assert pass2["outcomes_0"][0, 0] == 1.0
    assert pass2["outcomes_1"][0, 0] == 0.0


# ── Choice coding unchanged ─────────────────────────────────────────


def test_choice_mapping_unchanged():
    """Raw choice 1.0 -> binary 1.0, raw choice 2.0 -> binary 0.0."""
    df = pd.DataFrame({
        "subject_id": ["s1"] * 4,
        "choice": [1.0, 2.0, 1.0, 2.0],
        "trial_num": [0, 1, 2, 3],
        "outcomes_0": [1, 0, 0, 1],
    })
    data = prepare_complementary_data(df)
    assert data["choices"][0, 0] == 1.0
    assert data["choices"][0, 1] == 0.0
    assert data["choices"][0, 2] == 1.0
    assert data["choices"][0, 3] == 0.0


def test_convert_preserves_choices():
    """Conversion does not alter the choices array."""
    data = {
        "outcomes_0": np.array([[1.0, 0.0]]),
        "outcomes_1": np.array([[0.0, 1.0]]),
        "choices": np.array([[1.0, 0.0]]),
    }
    result = convert_empirical_to_reward_coded(data)
    np.testing.assert_array_equal(result["choices"], data["choices"])


# ── Model-facing coding proof with simple simulation ────────────────


def test_reward_coded_option0_always_safe_increases_p_choose_0():
    """When option 0 is always safe (reward=1) and option 1 always
    explosive (reward=0), P(choose 0) > 0.5 after learning."""
    n_trials = 10
    outcomes_0 = np.ones(n_trials)
    outcomes_1 = np.zeros(n_trials)
    choices = np.ones(n_trials)

    from volatility_generalisation.modelling_code.model_simulation import run_model_scan

    _, outputs = run_model_scan(
        "rescorla_wagner_single_value",
        {"learning_rate": 0.5, "temperature": 0.5},
        {"outcomes_0": outcomes_0, "outcomes_1": outcomes_1, "choices": choices},
    )
    mean_p = float(np.mean(outputs["prev_pos"]))
    assert mean_p > 0.55, (
        f"Expected P(option 0) > 0.55 for always-safe option 0, "
        f"got {mean_p:.4f}"
    )


def test_reward_coded_option0_always_explosive_decreases_p_choose_0():
    """When option 0 is always explosive (reward=0) and option 1 always
    safe (reward=1), P(choose 0) < 0.5 after learning."""
    n_trials = 10
    outcomes_0 = np.zeros(n_trials)
    outcomes_1 = np.ones(n_trials)
    choices = np.ones(n_trials)

    from volatility_generalisation.modelling_code.model_simulation import run_model_scan

    _, outputs = run_model_scan(
        "rescorla_wagner_single_value",
        {"learning_rate": 0.5, "temperature": 0.5},
        {"outcomes_0": outcomes_0, "outcomes_1": outcomes_1, "choices": choices},
    )
    mean_p = float(np.mean(outputs["prev_pos"]))
    assert mean_p < 0.45, (
        f"Expected P(option 0) < 0.45 for always-explosive option 0, "
        f"got {mean_p:.4f}"
    )


# ── fit_mcmc does not mutate input ──────────────────────────────────


class _FakeNUTS:
    def __init__(self, *args, **kwargs):
        pass


class _RecordingMCMC:
    def __init__(self, *args, **kwargs):
        self.recorded_data = None

    def run(self, rng_key, **kwargs):
        data = kwargs.get("data", {})
        self.recorded_data = {
            key: np.asarray(data[key])
            for key in ("outcomes_0", "outcomes_1", "choices")
            if key in data
        }

    def get_samples(self, group_by_chain=False):
        return {}

    def get_extra_fields(self):
        return {}

    @property
    def num_chains(self):
        return 1


def test_fit_mcmc_normalizes_explosion_coded_data_without_mutating_input(monkeypatch):
    """fit_mcmc should normalize explosion-coded arrays once."""
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.hierarchical_bayes.NUTS",
        _FakeNUTS,
    )
    recorder = _RecordingMCMC()
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.hierarchical_bayes.MCMC",
        lambda *a, **kw: recorder,
    )

    n_subjects, n_trials = 2, 5
    data = {
        "outcomes_0": jnp.ones((n_subjects, n_trials)),
        "outcomes_1": jnp.zeros((n_subjects, n_trials)),
        "choices": jnp.ones((n_subjects, n_trials)),
    }
    data_copy = {k: jnp.array(v) for k, v in data.items()}

    from volatility_generalisation.modelling_code.hierarchical_bayes import fit_mcmc

    fit_mcmc(
        data=data,
        model_names="rescorla_wagner_single_value",
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=0,
    )

    for key in data:
        assert jnp.allclose(
            data[key], data_copy[key]
        ), f"fit_mcmc mutated input['{key}']"

    assert np.allclose(recorder.recorded_data["outcomes_0"], np.zeros_like(recorder.recorded_data["outcomes_0"]))
    assert np.allclose(recorder.recorded_data["outcomes_1"], np.ones_like(recorder.recorded_data["outcomes_1"]))
    assert np.allclose(recorder.recorded_data["choices"], np.ones_like(recorder.recorded_data["choices"]))


def test_fit_mcmc_respects_reward_safe_metadata(monkeypatch):
    """Metadata-tagged reward-coded arrays should not be flipped again."""
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.hierarchical_bayes.NUTS",
        _FakeNUTS,
    )
    recorder = _RecordingMCMC()
    monkeypatch.setattr(
        "volatility_generalisation.modelling_code.hierarchical_bayes.MCMC",
        lambda *a, **kw: recorder,
    )

    data = {
        "outcomes_0": jnp.ones((1, 3)),
        "outcomes_1": jnp.zeros((1, 3)),
        "choices": jnp.ones((1, 3)),
        OUTCOME_CODING_KEY: OUTCOME_CODING_REWARD_SAFE,
    }

    from volatility_generalisation.modelling_code.hierarchical_bayes import fit_mcmc

    fit_mcmc(
        data=data,
        model_names="rescorla_wagner_single_value",
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=0,
    )

    np.testing.assert_array_equal(recorder.recorded_data["outcomes_0"], np.ones_like(recorder.recorded_data["outcomes_0"]))
    np.testing.assert_array_equal(recorder.recorded_data["outcomes_1"], np.zeros_like(recorder.recorded_data["outcomes_1"]))
    np.testing.assert_array_equal(recorder.recorded_data["choices"], np.ones_like(recorder.recorded_data["choices"]))
