#!/usr/bin/env python3
"""Run MCMC-only parameter recovery across prepared empirical conditions.

This is the standalone parameter-recovery analysis. For each model and condition
it simulates participants with known parameter values, using the trial schedules
of the prepared empirical data, refits the model with MCMC, and saves the results
so that the fitted parameters can be compared with the true ones. Each SLURM array
task started by submit_parameter_recovery.py runs it for one model and condition.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import tempfile
from pathlib import Path


def _resolve_host_device_count() -> int:
    """Work out how many CPU cores the fit can use, on SLURM or on a local machine.

    Uses the number of CPUs SLURM gave to the task if that is set and valid.
    Otherwise uses the number of CPUs on this machine. The result is never
    below 1.

    Returns:
        Number of CPU cores to use for running MCMC chains.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    local_cpus = os.cpu_count()
    return max(1, local_cpus or 1)


HOST_DEVICE_COUNT = _resolve_host_device_count()
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpyro

numpyro.set_platform("cpu")
numpyro.set_host_device_count(HOST_DEVICE_COUNT)

job_tmp_dir = tempfile.mkdtemp(prefix="arviz_tmp_")
os.environ["ARVIZ_DATA_DIR"] = job_tmp_dir


def cleanup_arviz_tmp() -> None:
    """Delete this job's temporary ArviZ folder when the script exits."""
    shutil.rmtree(job_tmp_dir, ignore_errors=True)


atexit.register(cleanup_arviz_tmp)

import jax
import matplotlib
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

from volatility_generalisation.modelling_code.parameter_recovery import run_parameter_recovery
from volatility_generalisation.modelling_code.model_simulation import MODEL_INFO
from volatility_generalisation.modelling_code.pipeline_config import (
    load_pipeline_config,
)

DEFAULT_CONFIG = "scripts/config.toml"
DEFAULT_MODEL_CONFIG = "config/model_bounds.toml"
RETAINED_MODELS = tuple(MODEL_INFO.keys())
FALLBACK_MODELS = list(RETAINED_MODELS)
FALLBACK_CONDITIONS = [
    "VolatileV1",
    "StableV1",
    "StableV2",
    "VolatileV2",
    "VolatileV3",
    "VolatileV3_2",
    "StableV4",
    "StableV4_2",
]
DEFAULT_DATA_DIR = "data/prepared"
DEFAULT_OUTPUT_DIR = "results/parameter_recovery"

console = Console()


def setup_logging() -> logging.Logger:
    """Set up log messages so that they print to the terminal in colour.

    Returns:
        The logger named "parameter_recovery", which writes messages of level
        INFO and above.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
        ],
        force=True,
    )
    return logging.getLogger("parameter_recovery")


def render_run_configuration(
    models: list[str],
    conditions: list[str],
    args: argparse.Namespace,
) -> None:
    """Print a panel that lists the settings of this recovery run.

    Args:
        models: Names of the models that will be recovered.
        conditions: Names of the conditions that will be used.
        args: Command-line options, which supply the number of simulated
            participants, the MCMC settings, the seed and the folder paths.
    """
    config_table = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    config_table.add_column(style="bold cyan", no_wrap=True)
    config_table.add_column(style="white")
    config_table.add_row("Models", ", ".join(models))
    config_table.add_row("Conditions", ", ".join(conditions))
    config_table.add_row("Subjects", str(args.n_subjects))
    config_table.add_row("Warmup", str(args.num_warmup))
    config_table.add_row("Samples", str(args.num_samples))
    config_table.add_row("Chains", str(args.num_chains))
    config_table.add_row("Seed", str(args.seed))
    config_table.add_row("Target Accept", str(args.target_accept_prob))
    config_table.add_row("Tree Depth", str(args.max_tree_depth))
    config_table.add_row("Thinning", str(args.thinning))
    config_table.add_row("Model Config", args.model_config)
    config_table.add_row("Data", args.data_dir)
    config_table.add_row("Output", args.output_dir)

    console.print(
        Panel.fit(
            config_table,
            title="[bold green]MCMC Parameter Recovery[/bold green]",
            subtitle="[dim]Prepared empirical schedules[/dim]",
            border_style="green",
        )
    )


def load_defaults(config_path: Path) -> tuple[list[str], list[str]]:
    """Read the default models and conditions from the pipeline config file.

    Args:
        config_path: Path to the pipeline config TOML file. The file must
            exist. If it lacks a list, the built-in list is used.

    Returns:
        Tuple of the list of model names and the list of condition names.
    """
    config = load_pipeline_config(config_path)
    fitting = config.get("fitting", {})
    return fitting.get("models", FALLBACK_MODELS), fitting.get(
        "conditions", FALLBACK_CONDITIONS
    )


def parse_args() -> argparse.Namespace:
    """Read the command-line options for the parameter-recovery run.

    Covers the models, the conditions, the data and output folders, the number
    of simulated participants, the MCMC settings and the seed.

    Returns:
        The command-line options, one attribute per option.
    """
    parser = argparse.ArgumentParser(
        description="Run MCMC-only parameter recovery using prepared empirical schedules."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to pipeline config TOML")
    parser.add_argument(
        "--model-config",
        default=DEFAULT_MODEL_CONFIG,
        help="Path to model-bounds config TOML",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=RETAINED_MODELS,
        help="Models to recover",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Prepared conditions to recover; defaults to all configured conditions",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Prepared data directory")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for recovery outputs",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=5,
        help="Synthetic subjects per model-condition run",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=100,
        help="MCMC warmup samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="MCMC posterior samples",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=1,
        help="Number of MCMC chains",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=0.95,
        help="NUTS target acceptance probability",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help="Maximum NUTS tree depth",
    )
    parser.add_argument("--thinning", type=int, default=1, help="MCMC thinning factor")
    return parser.parse_args()


def validate_models(models: list[str]) -> list[str]:
    """Check that every requested model is one that can be simulated and fitted.

    Args:
        models: Model names to check.

    Returns:
        The same list of model names.

    Raises:
        ValueError: If any name is not a supported model. The message lists the
            unsupported names and the available models.
    """
    invalid_models = [model for model in models if model not in MODEL_INFO]
    if invalid_models:
        raise ValueError(
            "Unsupported model(s): "
            + ", ".join(invalid_models)
            + ". Available models: "
            + ", ".join(RETAINED_MODELS)
        )
    return models


def main() -> None:
    """Run parameter recovery for the chosen models and conditions.

    Reads the command-line options, checks the model names, and runs the
    recovery for every model and condition. It then prints a table with the
    output folder of each run. If the recovery fails, the error is logged and
    raised again, so the run stops.

    Raises:
        Exception: Any error raised by the recovery run, after it is logged.
    """
    logger = setup_logging()
    args = parse_args()
    configured_models, configured_conditions = load_defaults(Path(args.config))

    models = args.models or configured_models
    models = validate_models(list(models))
    conditions = args.conditions or configured_conditions

    render_run_configuration(models, conditions, args)
    logger.info("Host device count: %s", HOST_DEVICE_COUNT)
    logger.info("JAX local device count: %s", jax.local_device_count())

    try:
        results = run_parameter_recovery(
            models=models,
            conditions=conditions,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_subjects=args.n_subjects,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            seed=args.seed,
            target_accept_prob=args.target_accept_prob,
            max_tree_depth=args.max_tree_depth,
            thinning=args.thinning,
            config_path=args.model_config,
        )
    except Exception:
        logger.exception("Parameter recovery failed")
        raise

    logger.info("Completed parameter recovery runs")
    table = Table(
        title="[bold green]Run Outputs[/bold green]",
        box=box.ROUNDED,
        header_style="bold green",
    )
    table.add_column("Condition", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta", no_wrap=True)
    table.add_column("Output Dir", style="white")
    for result in results:
        table.add_row(result["condition"], result["model"], result["output_dir"])
    console.print(table)


if __name__ == "__main__":
    main()
