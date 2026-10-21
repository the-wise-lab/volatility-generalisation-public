import hashlib
from pathlib import Path

import pytest
import toml

from volatility_generalisation.modelling_code.data_generation import (
    DEFAULT_MODEL_BOUNDS_CONFIG,
    load_parameter_bounds,
    settings_file_record,
)


RETAINED_MODELS = {
    "rescorla_wagner_single_value": {"temperature", "learning_rate"},
    "dual_lr_single_value": {"temperature", "learning_rate_pos", "learning_rate_neg"},
    "pearce_hall_single_value": {"temperature", "kappa", "gamma"},
    "leaky_beta_single_value": {"temperature", "tau", "decay"},
    "leaky_beta_asymmetric_single_value": {"temperature", "tau_pos", "tau_neg", "decay"},
}


def test_load_parameter_bounds_defaults_to_root_config() -> None:
    bounds = load_parameter_bounds("rescorla_wagner_single_value")

    assert bounds["temperature"]["min"] == 0.02
    assert bounds["temperature"]["max"] == 0.6
    assert bounds["learning_rate"]["min"] == 0.0
    assert bounds["learning_rate"]["max"] == 1.0


def test_model_bounds_configs_contain_only_retained_model_sections() -> None:
    root_bounds = toml.load(Path("config/model_bounds.toml"))

    assert set(root_bounds.keys()) == {"common", *RETAINED_MODELS.keys()}
    for removed_name in [
        "rescorla_wagner",
        "rescorla_wagner_independent",
        "dual_lr",
        "dual_lr_independent",
        "pearce_hall",
        "pearce_hall_single_value_dual_assoc",
        "pearce_hall_independent",
        "leaky_beta",
        "leaky_beta_independent",
        "leaky_beta_asymmetric",
        "leaky_beta_asymmetric_independent",
        "leaky_beta_asymmetric_single_value_reparam",
    ]:
        assert removed_name not in root_bounds


def test_load_parameter_bounds_supports_new_models() -> None:
    for model_name, expected_keys in RETAINED_MODELS.items():
        bounds = load_parameter_bounds(model_name)
        assert set(bounds.keys()) == expected_keys


def test_load_parameter_bounds_rejects_removed_models() -> None:
    for model_name in [
        "rescorla_wagner",
        "rescorla_wagner_independent",
        "dual_lr",
        "dual_lr_independent",
        "pearce_hall",
        "pearce_hall_single_value_dual_assoc",
        "pearce_hall_independent",
        "leaky_beta",
        "leaky_beta_independent",
        "leaky_beta_asymmetric",
        "leaky_beta_asymmetric_independent",
        "leaky_beta_asymmetric_single_value_reparam",
    ]:
        try:
            load_parameter_bounds(model_name)
        except ValueError:
            continue
        raise AssertionError(f"Expected {model_name} to be rejected")


def test_settings_file_record_hashes_the_file_bytes(tmp_path: Path) -> None:
    """The record holds the path as given and the SHA-256 of the file's bytes."""
    config_path = tmp_path / "bounds.toml"
    config_path.write_bytes(b"[common]\n")

    record = settings_file_record(config_path)

    assert record == {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(b"[common]\n").hexdigest(),
    }


def test_settings_file_record_differs_for_a_different_file(tmp_path: Path) -> None:
    """Two files that differ by one character get different hashes."""
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text("[common]\nx = 1\n", encoding="utf-8")
    second.write_text("[common]\nx = 2\n", encoding="utf-8")

    assert (
        settings_file_record(first)["config_sha256"]
        != settings_file_record(second)["config_sha256"]
    )


def test_settings_file_record_defaults_to_root_config() -> None:
    """With no path, the record describes the same default file the fits load."""
    record = settings_file_record()

    assert record["config_path"] == str(DEFAULT_MODEL_BOUNDS_CONFIG)
    assert (
        record["config_sha256"]
        == hashlib.sha256(DEFAULT_MODEL_BOUNDS_CONFIG.read_bytes()).hexdigest()
    )


def test_settings_file_record_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing settings file stops the run instead of saving an empty record."""
    with pytest.raises(FileNotFoundError):
        settings_file_record(tmp_path / "missing.toml")
