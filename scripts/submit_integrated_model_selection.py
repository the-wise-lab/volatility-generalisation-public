#!/usr/bin/env python3
"""Submit integrated model-selection jobs to SLURM.

This script starts the model-selection analysis. It submits one SLURM array job
with one task per candidate model. Each task runs run_integrated_model_selection.py,
which fits that model to all chosen conditions together. The models,
conditions, SLURM resources and MCMC settings that are not given on the command
line come from the pipeline config file, with built-in fallbacks. For the SLURM
time and memory, the config file's `integrated_selection_time` and
`integrated_selection_memory` take priority over its shared `time` and
`memory`. The data and output folders have fixed defaults and are not read from
the config file. Use `--dry-run` to print the SLURM script without submitting
it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence

from volatility_generalisation.modelling_code.integrated_model_selection import (
    get_default_conditions,
    get_default_models,
)
from volatility_generalisation.modelling_code.model_simulation import MODEL_INFO
from volatility_generalisation.modelling_code.pipeline_config import (
    load_pipeline_config,
)


DEFAULT_CONFIG = "scripts/config.toml"
DEFAULT_OUTPUT_DIR = "results/integrated_model_selection"
DEFAULT_DATA_DIR = "data/prepared"
FALLBACK_TIME = "04:00:00"
FALLBACK_MEM = "16G"
FALLBACK_CPUS = 4
FALLBACK_PARTITION = "cpu"
FALLBACK_PYTHON_COMMAND = "uv run --frozen python"
RETAINED_MODELS = tuple(MODEL_INFO.keys())
# Settings that cannot be lower than 1, with where each one is set in the config file.
AT_LEAST_ONE_OPTIONS = {
    "num_samples": "[mcmc] num_samples",
    "num_chains": "[mcmc] num_chains",
    "cpus": "[hpc] cpus_per_task",
    "thinning": "[mcmc] thinning",
    "max_tree_depth": "[mcmc] max_tree_depth",
}


def parse_args() -> argparse.Namespace:
    """Read the command-line options for submitting model-selection jobs.

    Covers the models, the conditions, the data and output folders, the SLURM
    resources, the MCMC settings, and whether to overwrite or skip existing
    results. The models, conditions, SLURM resources and MCMC settings are set to
    None when left out, so that the config file can supply them. The data and
    output folders have fixed defaults.

    Returns:
        The command-line options, one attribute per option.
    """
    parser = argparse.ArgumentParser(description="Submit integrated model-selection jobs")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", default=None, choices=RETAINED_MODELS)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time", default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--num-warmup", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-accept-prob", type=float, default=None)
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument("--thinning", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", action="store_true")
    overwrite_group.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def detect_project_dir() -> Path:
    """Find the project folder, which is the one that holds pyproject.toml.

    Checks the current folder, then its parent. If neither has pyproject.toml,
    the current folder is used.

    Returns:
        Path to the project folder.
    """
    current_dir = Path.cwd()
    if (current_dir / "pyproject.toml").exists():
        return current_dir
    if (current_dir.parent / "pyproject.toml").exists():
        return current_dir.parent
    return current_dir


def validate_settings(args: argparse.Namespace) -> None:
    """Stop if a setting that must be at least 1 is lower, or no model is chosen.

    Call this after the command line and config file have been combined, so a
    bad value in either one is caught.

    Args:
        args: Command-line options after the defaults have been filled in.

    Raises:
        ValueError: If the model list is empty, or if the number of samples,
            chains or CPUs, the thinning or the maximum tree depth is below 1.
            The message names the option.
    """
    if not args.models:
        raise ValueError(
            "The model list is empty. Give --models or list models under [fitting] "
            "in the config file."
        )
    for name, config_setting in AT_LEAST_ONE_OPTIONS.items():
        value = getattr(args, name)
        if value < 1:
            option = f"--{name.replace('_', '-')}"
            raise ValueError(
                f"{option} (or {config_setting} in the config file) must be at "
                f"least 1, but it is {value}."
            )


def validate_models(models: Sequence[str]) -> list[str]:
    """Check that every requested model is one that can be simulated and fitted.

    Args:
        models: Model names to check.

    Returns:
        The model names as a list.

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
    return list(models)


def build_job_models(
    models: Sequence[str],
    output_dir: Path,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> list[str]:
    """Pick the models that still need to be fitted.

    A model counts as already fitted if its output folder holds posterior.nc.
    With `overwrite`, fitted models are refitted. With `skip_existing`, they
    are left out. With neither, finding a fitted model is an error, so results
    are not overwritten by accident.

    Args:
        models: Names of the candidate models.
        output_dir: Folder that holds one sub-folder of results per model.
        overwrite: If True, refit models that already have results.
        skip_existing: If True, leave out models that already have results.

    Returns:
        Names of the models to submit.

    Raises:
        RuntimeError: If some models already have results and neither option
            is set, or if `skip_existing` is set and every model already has
            results.
    """
    pending = []
    existing = []
    for model in models:
        run_dir = output_dir / model
        if (run_dir / "posterior.nc").exists():
            if overwrite:
                pending.append(model)
            elif skip_existing:
                continue
            else:
                existing.append(model)
        else:
            pending.append(model)

    if existing:
        raise RuntimeError(
            "Existing integrated model-selection outputs found for: " + ", ".join(existing)
        )

    if skip_existing and not pending:
        raise RuntimeError("All selected integrated model-selection runs already have outputs.")

    return pending


def generate_job_list(models: Sequence[str], filename: Path) -> None:
    """Write the job-list file that tells each SLURM task which model to fit.

    Line N holds the model name for SLURM array task N.

    Args:
        models: Names of the models to fit.
        filename: Path of the file to write.
    """
    with open(filename, "w", encoding="utf-8") as handle:
        for model in models:
            handle.write(f"{model}\n")


def generate_slurm_script(
    job_list_file: Path,
    n_jobs: int,
    args: argparse.Namespace,
    project_dir: Path,
    python_command: str,
) -> str:
    """Write the text of the SLURM array script for the model-selection jobs.

    Each task uses a separate ArviZ cache while keeping the uv cache shared.

    Each array task reads its line of the job-list file and runs
    run_integrated_model_selection.py for that model, with the settings in
    `args`.

    Args:
        job_list_file: Path to the job-list file written by generate_job_list.
        n_jobs: Number of array tasks.
        args: Command-line options after the defaults have been filled in from
            the config. They supply the SLURM resources, the conditions, the
            folder paths and the MCMC settings.
        project_dir: Path to the project folder.
        python_command: Command used to start Python on the cluster.

    Returns:
        Text of the SLURM batch script.
    """

    worker_script = project_dir / "scripts" / "run_integrated_model_selection.py"
    conditions_flags = " ".join(args.conditions)
    return f"""#!/bin/bash
#SBATCH --job-name=integrated_model_selection
#SBATCH --output=logs/integrated_model_selection_%A_%a.out
#SBATCH --error=logs/integrated_model_selection_%A_%a.err
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --partition={args.partition}
#SBATCH --array=1-{n_jobs}

set -e

PROJECT_DIR=\"{project_dir}\"
JOB_LIST=\"{job_list_file}\"
cd \"$PROJECT_DIR\"

export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Keep uv's shared cache before giving each task a separate ArviZ cache.
export UV_CACHE_DIR="${{UV_CACHE_DIR:-${{XDG_CACHE_HOME:-$HOME/.cache}}/uv}}"
export XDG_CACHE_HOME="${{TMPDIR:-/tmp}}/cache_${{SLURM_JOB_ID}}"
export XLA_FLAGS=\"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count={args.cpus}\"

MODEL=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$JOB_LIST")

{python_command} {worker_script} \
    --model "$MODEL" \
    --conditions {conditions_flags} \
    --data-dir {args.data_dir} \
    --output-dir {args.output_dir} \
    --num-warmup {args.num_warmup} \
    --num-samples {args.num_samples} \
    --num-chains {args.num_chains} \
    --seed {args.seed} \
    --target-accept-prob {args.target_accept_prob} \
    --max-tree-depth {args.max_tree_depth} \
    --thinning {args.thinning}
"""


def backup_existing_dirs(
    model_names: Sequence[str],
    output_dir: Path,
    project_dir: Path,
) -> None:
    """Move existing model output directories to a timestamped backup.

    This prevents stale diagnostics.json or posterior.nc from persisting
    in the active result path when jobs are resubmitted with --overwrite.

    Args:
        model_names: Names of the models whose output folders should be moved.
        output_dir: Folder that holds one sub-folder of results per model.
        project_dir: Path to the project folder, used to print backup paths
            relative to it.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_base = output_dir / "_overwrite_backups" / timestamp
    archived = False
    for model in model_names:
        run_dir = output_dir / model
        if run_dir.exists():
            backup_dir = backup_base / model
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run_dir), str(backup_dir))
            rel = backup_dir.relative_to(project_dir)
            print(f"  Archived existing {model} to {rel}")
            archived = True
    if not archived:
        print("  No existing outputs to archive.")


def main() -> None:
    """Choose the models to fit and submit them to SLURM as one array job.

    Fills in the models, conditions, SLURM resources and MCMC settings left out
    on the command line from the config file, and works out which models need
    fitting. With `--overwrite`, it then moves the existing results to a
    timestamped backup. It writes the job-list file and submits the array job
    with `sbatch`. With `--dry-run` it prints the SLURM script and stops without
    changing anything. A value of 0 given on the command line is kept, not
    replaced by the config value. The SLURM time and memory come from
    `integrated_selection_time` and `integrated_selection_memory` in the config
    file, then from its shared `time` and `memory`, then from the built-in
    fallbacks.

    Raises:
        ValueError: If the model list is empty, or a setting that must be at
            least 1 is lower.
        RuntimeError: If `sbatch` fails. Errors from build_job_models are also
            raised.
    """
    args = parse_args()
    config = load_pipeline_config(args.config)
    project_dir = detect_project_dir()
    hpc_config = config.get("hpc", {})
    mcmc_config = config.get("mcmc", {})

    if args.models is None:
        args.models = get_default_models(args.config)
    args.models = validate_models(args.models)
    if args.conditions is None:
        args.conditions = get_default_conditions(args.config)
    if args.time is None:
        args.time = hpc_config.get(
            "integrated_selection_time", hpc_config.get("time", FALLBACK_TIME)
        )
    if args.mem is None:
        args.mem = hpc_config.get(
            "integrated_selection_memory", hpc_config.get("memory", FALLBACK_MEM)
        )
    if args.cpus is None:
        args.cpus = hpc_config.get("cpus_per_task", FALLBACK_CPUS)
    if args.partition is None:
        args.partition = hpc_config.get("partition", FALLBACK_PARTITION)
    if args.num_warmup is None:
        args.num_warmup = mcmc_config.get("num_warmup", 2000)
    if args.num_samples is None:
        args.num_samples = mcmc_config.get("num_samples", 4000)
    if args.num_chains is None:
        args.num_chains = mcmc_config.get("num_chains", 4)
    if args.seed is None:
        args.seed = mcmc_config.get("seed", 42)
    if args.target_accept_prob is None:
        args.target_accept_prob = mcmc_config.get("target_accept_prob", 0.95)
    if args.max_tree_depth is None:
        args.max_tree_depth = mcmc_config.get("max_tree_depth", 10)
    if args.thinning is None:
        args.thinning = mcmc_config.get("thinning", 1)
    python_command = hpc_config.get("python_command", FALLBACK_PYTHON_COMMAND)
    validate_settings(args)

    output_dir = project_dir / args.output_dir
    models = build_job_models(
        args.models,
        output_dir,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
    )

    job_list_file = project_dir / "scripts" / ".integrated_model_selection_job_list.txt"
    script = generate_slurm_script(job_list_file, len(models), args, project_dir, python_command)

    if args.dry_run:
        if args.overwrite:
            print("[Dry run, skipping backup]")
        print("Job list (not written):")
        for model in models:
            print(model)
        print()
        print(script)
        return

    # Backup existing output directories before overwriting
    if args.overwrite:
        print("Archiving existing outputs before overwrite...")
        backup_existing_dirs(args.models, output_dir, project_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_job_list(models, job_list_file)

    submission = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if submission.returncode != 0:
        raise RuntimeError(submission.stderr.strip() or "sbatch submission failed")
    print(submission.stdout.strip())


if __name__ == "__main__":
    main()
