"""Tests for the strict settings-file loader."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import toml

from volatility_generalisation.modelling_code.pipeline_config import (
    load_pipeline_config,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "modelling_code" / "pipeline_config.py"
)


def test_load_pipeline_config_returns_the_parsed_file(tmp_path: Path) -> None:
    """A good file comes back as a dictionary, from a text or a Path argument."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("[mcmc]\nseed = 0\n", encoding="utf-8")

    assert load_pipeline_config(config_path) == {"mcmc": {"seed": 0}}
    assert load_pipeline_config(str(config_path)) == {"mcmc": {"seed": 0}}


def test_load_pipeline_config_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing file stops with an error that names the file."""
    missing = tmp_path / "missing.toml"

    with pytest.raises(FileNotFoundError, match="missing.toml"):
        load_pipeline_config(missing)


def test_load_pipeline_config_rejects_a_broken_file(tmp_path: Path) -> None:
    """A file that is not valid TOML stops with the file name and the cause."""
    config_path = tmp_path / "broken.toml"
    config_path.write_text("[mcmc]\nseed = 1\nseed = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken.toml") as error:
        load_pipeline_config(config_path)

    assert isinstance(error.value.__cause__, toml.TomlDecodeError)


def test_load_pipeline_config_rejects_a_folder(tmp_path: Path) -> None:
    """A path that is a folder cannot be read, so it stops with the path named."""
    with pytest.raises(ValueError, match=tmp_path.name) as error:
        load_pipeline_config(tmp_path)

    assert isinstance(error.value.__cause__, OSError)


def test_pipeline_config_module_does_not_load_jax() -> None:
    """Loading the module directly does not import JAX or NumPyro."""
    code = (
        "import importlib.util, sys\n"
        f"path = {str(MODULE_PATH)!r}\n"
        "spec = importlib.util.spec_from_file_location('pipeline_config', path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert 'jax' not in sys.modules\n"
        "assert 'numpyro' not in sys.modules\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
