from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from volatility_generalisation.modelling_code.data_preparation import (
    OUTCOME_CODING_KEY,
    OUTCOME_CODING_REWARD_SAFE,
)
from volatility_generalisation.modelling_code.data_generation import (
    settings_file_record,
)
from volatility_generalisation.modelling_code.hierarchical_bayes import _normalize_fit_data
from volatility_generalisation.modelling_code.plotting import posterior_predictive_check


def _load_fit_single_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "fit_single.py"
    spec = importlib.util.spec_from_file_location("fit_single", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_parameter_summaries_flattens_chain_and_draw_dimensions() -> None:
    module = _load_fit_single_module()

    posterior_samples = {
        "dual_lr_single_value_learning_rate_pos_subj": np.array(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ],
            dtype=float,
        ),
        "dual_lr_single_value_temperature_subj": np.array(
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[0.1, 0.3], [0.5, 0.7]],
            ],
            dtype=float,
        ),
    }

    summaries = module.extract_parameter_summaries(
        posterior_samples,
        "dual_lr_single_value",
    )

    np.testing.assert_allclose(
        summaries["learning_rate_pos"]["subject_means"],
        np.array([0.4, 0.5], dtype=float),
    )
    assert summaries["learning_rate_pos"]["mean"] == np.mean([0.4, 0.5])

    np.testing.assert_allclose(
        summaries["temperature"]["subject_means"],
        np.array([0.35, 0.55], dtype=float),
    )


def test_posterior_predictive_check_flattens_grouped_chain_samples() -> None:
    data = {
        "outcomes_0": np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=float),
        "outcomes_1": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
        "choices": np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=float),
    }
    posterior_samples = {
        "rescorla_wagner_single_value_learning_rate_subj": np.array(
            [
                [[0.9, 0.9], [0.9, 0.9]],
                [[0.9, 0.9], [0.9, 0.9]],
            ],
            dtype=float,
        ),
        "rescorla_wagner_single_value_temperature_subj": np.array(
            [
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.1, 0.1], [0.1, 0.1]],
            ],
            dtype=float,
        ),
    }

    results = posterior_predictive_check(
        data=data,
        posterior_samples=posterior_samples,
        model_name="rescorla_wagner_single_value",
        n_posterior_samples=4,
        seed=0,
        verbose=False,
    )

    assert results["all_simulated_probs"].shape == (4, 2, 3)
    assert results["simulated_probs"].shape == (2, 3)
    assert results["accuracy"] > 0.7
    assert np.all(results["simulated_probs"][0, 1:] > 0.5)
    assert np.all(results["simulated_probs"][1, 1:] < 0.5)


def test_fit_and_diagnose_passes_model_config(monkeypatch) -> None:
    module = _load_fit_single_module()

    captured = {}

    class FakeMCMC:
        def get_samples(self, group_by_chain=True):
            assert group_by_chain is True
            return {
                "dual_lr_single_value_learning_rate_pos_subj": np.zeros((1, 1)),
            }

    def fake_fit_mcmc(**kwargs):
        captured.update(kwargs)
        return FakeMCMC()

    monkeypatch.setattr(module, "fit_mcmc", fake_fit_mcmc)
    monkeypatch.setattr(module, "diagnose_mcmc", lambda **kwargs: {"converged": True})

    module.fit_and_diagnose(
        data={
            "outcomes_0": np.zeros((1, 2)),
            "outcomes_1": np.zeros((1, 2)),
            "choices": np.zeros((1, 2)),
        },
        model_name="dual_lr_single_value",
        model_config="config/model_bounds.toml",
        num_warmup=10,
        num_samples=20,
        num_chains=2,
        seed=42,
    )

    assert captured["config_path"] == "config/model_bounds.toml"


def test_save_results_metadata_includes_model_config(tmp_path: Path) -> None:
    module = _load_fit_single_module()
    config_path = tmp_path / "model_bounds.toml"
    config_path.write_text("[common]\n", encoding="utf-8")

    module.save_results(
        output_dir=tmp_path,
        condition_name="StableV1",
        model_name="dual_lr_single_value",
        settings_record=settings_file_record(config_path),
        posterior_samples={"dual_lr_single_value_learning_rate_pos_subj": np.zeros((1, 1))},
        ppc_results={
            "actual_choices": np.zeros((1, 2)),
            "accuracy": 0.5,
        },
        diagnostics={"converged": True},
        parameter_summaries={"learning_rate_pos": {"mean": 0.1}},
    )

    metadata_path = tmp_path / "StableV1_dual_lr_single_value" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["config_path"] == str(config_path)
    assert metadata["config_sha256"] == hashlib.sha256(b"[common]\n").hexdigest()


def test_load_prepared_data_marks_normalized_outcomes_reward_safe(tmp_path: Path) -> None:
    module = _load_fit_single_module()

    data_path = tmp_path / "StableV1.npz"
    np.savez(
        data_path,
        outcomes_0=np.array([[1.0, 0.0]], dtype=float),
        outcomes_1=np.array([[0.0, 1.0]], dtype=float),
        choices=np.array([[1.0, 0.0]], dtype=float),
        subject_ids=np.array(["s1"], dtype=object),
    )

    loaded = module.load_prepared_data(data_path)

    assert loaded[OUTCOME_CODING_KEY] == OUTCOME_CODING_REWARD_SAFE
    np.testing.assert_array_equal(np.asarray(loaded["outcomes_0"]), np.array([[0.0, 1.0]]))
    np.testing.assert_array_equal(np.asarray(loaded["outcomes_1"]), np.array([[1.0, 0.0]]))

    renormalized = _normalize_fit_data(loaded)

    np.testing.assert_array_equal(renormalized["outcomes_0"], np.array([[0.0, 1.0]]))
    np.testing.assert_array_equal(renormalized["outcomes_1"], np.array([[1.0, 0.0]]))
