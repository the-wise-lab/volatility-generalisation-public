"""Read the pipeline settings file (``scripts/config.toml``).

Every script and module that needs the settings file reads it with
``load_pipeline_config``. The function stops with a clear error if the file is
missing or cannot be read, so no script quietly runs with built-in values.

This module imports only the standard library and ``toml``, so lightweight
scripts can use it without loading JAX or NumPyro.
"""

from pathlib import Path
from typing import Union

import toml


def load_pipeline_config(config_path: Union[str, Path]) -> dict:
    """Read the pipeline settings file.

    Args:
        config_path: Path to the TOML settings file.

    Returns:
        The settings as a dictionary.

    Raises:
        FileNotFoundError: If there is no file at ``config_path``.
        ValueError: If the file exists but cannot be read or is not valid TOML.
            The message names the file, and the original error is the cause.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    try:
        return toml.load(path)
    except (toml.TomlDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read settings file {path}: {exc}") from exc
