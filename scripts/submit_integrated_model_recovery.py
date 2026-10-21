#!/usr/bin/env python3
"""Submit integrated model-recovery jobs to SLURM.

This script starts the model-recovery analysis. It builds every combination of
replicate, simulated model and fitted model, and submits them as one SLURM array
job. Each array task runs run_integrated_model_recovery_fit.py once. The
models, conditions, SLURM resources, MCMC settings and base seed that are not
given on the command line come from the pipeline config file, with built-in
fallbacks. For the SLURM time and memory, the config file's
`integrated_recovery_time` and `integrated_recovery_memory` take priority over
its shared `time` and `memory`. The number of replicates, the number of
simulated participants, the number of initial trials dropped for
cross-validation, and the data and output folders have fixed defaults and are
not read from the config file. Use `--dry-run` to print the SLURM script
without submitting it.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from volatility_generalisation.modelling_code.integrated_model_recovery import (
    DEFAULT_RESULTS_DIR,
    build_recovery_job_grid,
    recovery_output_dir,
)
from volatility_generalisation.modelling_code.integrated_model_selection import (
    get_default_conditions,
    get_default_models,
)
from volatility_generalisation.modelling_code.model_simulation import MODEL_INFO
from volatility_generalisation.modelling_code.pipeline_config import (
    load_pipeline_config,
)

DEFAULT_CONFIG = "scripts/config.toml"
DEFAULT_DATA_DIR = "data/prepared"
FALLBACK_TIME = "04:00:00"
FALLBACK_MEM = "16G"
FALLBACK_CPUS = 4
FALLBACK_PARTITION = "cpu"
FALLBACK_PYTHON_COMMAND = "uv run --frozen python"
RETAINED_MODELS = tuple(MODEL_INFO.keys())
# Settings that cannot be lower than 1, with where each one is set in the config
# file. The array concurrency has no config value, so it is checked only when it
# is given.
AT_LEAST_ONE_OPTIONS = {
    "num_samples": "[mcmc] num_samples",
    "num_chains": "[mcmc] num_chains",
    "cpus": "[hpc] cpus_per_task",
    "thinning": "[mcmc] thinning",
    "max_tree_depth": "[mcmc] max_tree_depth",
    "array_concurrency": None,
}


def parse_args() -> argparse.Namespace:
    """Read the command-line options for submitting model-recovery jobs.

    Covers the models, the conditions, the number of replicates and simulated
    participants, the SLURM resources, the MCMC settings, and whether to skip
    or overwrite existing results. The models, conditions, SLURM resources, MCMC
    settings and base seed are set to None when left out, so that the config file
    can supply them. The number of replicates (10), the number of simulated
    participants (50), the number of initial trials dropped for cross-validation
    (1) and the data and output folders have fixed defaults.

    Returns:
        The command-line options, one attribute per option.
    """
    parser = argparse.ArgumentParser(description="Submit integrated model-recovery jobs")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", default=None, choices=RETAINED_MODELS)
    parser.add_argument("--simulated-models", nargs="+", default=None, choices=RETAINED_MODELS)
    parser.add_argument("--fitted-models", nargs="+", default=None, choices=RETAINED_MODELS)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--n-subjects", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--num-warmup", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument("--target-accept-prob", type=float, default=None)
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument("--thinning", type=int, default=None)
    parser.add_argument("--loo-drop-initial-trials", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--array-concurrency", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
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


def build_pending_jobs(jobs: list[dict[str, object]], output_dir: Path, overwrite: bool) -> list[dict[str, object]]:
    """Pick the recovery jobs that do not yet have a saved fit summary.

    A job counts as done if its output folder holds fit_summary.json.

    Args:
        jobs: Recovery jobs, each a dictionary with the replicate number, the
            simulated model and the fitted model.
        output_dir: Folder that holds the recovery results.
        overwrite: If True, every job is treated as pending.

    Returns:
        The jobs that still need to run.
    """
    if overwrite:
        return jobs

    pending = []
    for job in jobs:
        run_dir = recovery_output_dir(
            output_dir,
            replicate=int(job["replicate"]),
            simulated_model=str(job["simulated_model"]),
            fitted_model=str(job["fitted_model"]),
        )
        if not (run_dir / "fit_summary.json").exists():
            pending.append(job)
    return pending


def generate_job_list(jobs: list[dict[str, object]], filename: Path) -> None:
    """Write the job-list file that tells each SLURM task what to fit.

    Line N holds the replicate number, simulated model and fitted model, separated
    by tabs, for SLURM array task N.

    Args:
        jobs: Recovery jobs, each a dictionary with the replicate number, the
            simulated model and the fitted model.
        filename: Path of the file to write.
    """
    with open(filename, "w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(
                f"{job['replicate']}\t{job['simulated_model']}\t{job['fitted_model']}\n"
            )


def validate_settings(args: argparse.Namespace) -> None:
    """Stop if a setting that must be at least 1 is lower, or no model is chosen.

    Call this after the command line and config file have been combined, so a
    bad value in either one is caught.

    Args:
        args: Command-line options after the defaults have been filled in.

    Raises:
        ValueError: If the model list is empty, or if the number of samples,
            chains or CPUs, the thinning, the maximum tree depth or the array
            concurrency is below 1. The message names the option.
    """
    if not args.model_order:
        raise ValueError(
            "The model list is empty. Give --models or list models under [fitting] "
            "in the config file."
        )
    for name, config_setting in AT_LEAST_ONE_OPTIONS.items():
        value = getattr(args, name)
        if value is not None and value < 1:
            option = f"--{name.replace('_', '-')}"
            if config_setting is not None:
                option += f" (or {config_setting} in the config file)"
            raise ValueError(f"{option} must be at least 1, but it is {value}.")


def generate_slurm_script(
    job_list_file: Path,
    n_jobs: int,
    args: argparse.Namespace,
    project_dir: Path,
    python_command: str,
) -> str:
    """Write the text of the SLURM array script for the recovery jobs.

    Each task uses a separate ArviZ cache while keeping the uv cache shared.

    Each array task reads its line of the job-list file and runs
    run_integrated_model_recovery_fit.py with that replicate, simulated model
    and fitted model, plus the settings in `args`.

    Args:
        job_list_file: Path to the job-list file written by generate_job_list.
        n_jobs: Number of array tasks.
        args: Command-line options after the defaults have been filled in from
            the config. They supply the SLURM resources, the models, the
            conditions, the folder paths and the MCMC settings.
        project_dir: Path to the project folder.
        python_command: Command used to start Python on the cluster.

    Returns:
        Text of the SLURM batch script.
    """

    worker_script = project_dir / "scripts" / "run_integrated_model_recovery_fit.py"
    conditions_flags = " ".join(args.conditions)
    model_flags = " ".join(args.model_order)
    array_spec = f"1-{n_jobs}"
    if args.array_concurrency is not None:
        array_spec = f"{array_spec}%{args.array_concurrency}"
    return f"""#!/bin/bash
#SBATCH --job-name=integrated_model_recovery
#SBATCH --output=logs/integrated_model_recovery_%A_%a.out
#SBATCH --error=logs/integrated_model_recovery_%A_%a.err
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --partition={args.partition}
#SBATCH --array={array_spec}

set -e

PROJECT_DIR="{project_dir}"
JOB_LIST="{job_list_file}"
cd "$PROJECT_DIR"

export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Keep uv's shared cache before giving each task a separate ArviZ cache.
export UV_CACHE_DIR="${{UV_CACHE_DIR:-${{XDG_CACHE_HOME:-$HOME/.cache}}/uv}}"
export XDG_CACHE_HOME="${{TMPDIR:-/tmp}}/cache_${{SLURM_JOB_ID}}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count={args.cpus}"

IFS=$'\t' read -r REPLICATE SIMULATED_MODEL FITTED_MODEL < <(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$JOB_LIST")

{python_command} {worker_script} \
    --replicate "$REPLICATE" \
    --simulated-model "$SIMULATED_MODEL" \
    --fitted-model "$FITTED_MODEL" \
    --models {model_flags} \
    --conditions {conditions_flags} \
    --data-dir {args.data_dir} \
    --output-dir {args.output_dir} \
    --n-subjects {args.n_subjects} \
    --base-seed {args.base_seed} \
    --num-warmup {args.num_warmup} \
    --num-samples {args.num_samples} \
    --num-chains {args.num_chains} \
    --target-accept-prob {args.target_accept_prob} \
    --max-tree-depth {args.max_tree_depth} \
    --thinning {args.thinning} \
    --loo-drop-initial-trials {args.loo_drop_initial_trials}
"""


def main() -> None:
    """Build the model-recovery jobs and submit them to SLURM.

    Fills in the models, conditions, SLURM resources, MCMC settings and base
    seed left out on the command line from the config file, and builds every
    replicate, simulated-model and fitted-model combination. With
    `--skip-existing`, it drops the combinations that already have results.
    Without `--skip-existing`, every combination is submitted, including ones
    that already have results, and those results are overwritten (there is no
    check, and `--overwrite` has the same effect). It then writes the job-list
    file and submits the array job with `sbatch`. With `--dry-run` it prints the
    SLURM script and stops. A value of 0 given on the command line (for example
    `--base-seed 0`) is kept, not replaced by the config value. The SLURM time
    and memory come from `integrated_recovery_time` and
    `integrated_recovery_memory` in the config file, then from its shared
    `time` and `memory`, then from the built-in fallbacks.

    Raises:
        ValueError: If the model list is empty, or a setting that must be at
            least 1 is lower.
        RuntimeError: If every selected run already has results, or if `sbatch`
            fails.
    """
    args = parse_args()
    config = load_pipeline_config(args.config)
    project_dir = detect_project_dir()
    hpc_config = config.get("hpc", {})
    mcmc_config = config.get("mcmc", {})

    if args.models is None:
        args.models = get_default_models(args.config)
    args.model_order = validate_models(args.models)
    if args.simulated_models is None:
        args.simulated_models = args.model_order
    args.simulated_models = validate_models(args.simulated_models)
    if args.fitted_models is None:
        args.fitted_models = args.model_order
    args.fitted_models = validate_models(args.fitted_models)
    if args.conditions is None:
        args.conditions = get_default_conditions(args.config)
    if args.time is None:
        args.time = hpc_config.get(
            "integrated_recovery_time", hpc_config.get("time", FALLBACK_TIME)
        )
    if args.mem is None:
        args.mem = hpc_config.get(
            "integrated_recovery_memory", hpc_config.get("memory", FALLBACK_MEM)
        )
    if args.cpus is None:
        args.cpus = hpc_config.get("cpus_per_task", FALLBACK_CPUS)
    if args.partition is None:
        args.partition = hpc_config.get("partition", FALLBACK_PARTITION)
    if args.base_seed is None:
        args.base_seed = mcmc_config.get("seed", 42)
    if args.num_warmup is None:
        args.num_warmup = mcmc_config.get("num_warmup", 2000)
    if args.num_samples is None:
        args.num_samples = mcmc_config.get("num_samples", 4000)
    if args.num_chains is None:
        args.num_chains = mcmc_config.get("num_chains", 4)
    if args.target_accept_prob is None:
        args.target_accept_prob = mcmc_config.get("target_accept_prob", 0.95)
    if args.max_tree_depth is None:
        args.max_tree_depth = mcmc_config.get("max_tree_depth", 10)
    if args.thinning is None:
        args.thinning = mcmc_config.get("thinning", 1)
    python_command = hpc_config.get("python_command", FALLBACK_PYTHON_COMMAND)
    validate_settings(args)

    output_dir = project_dir / args.output_dir
    jobs = build_recovery_job_grid(args.replicates, args.simulated_models, args.fitted_models)
    pending_jobs = build_pending_jobs(jobs, output_dir, overwrite=args.overwrite)
    if args.overwrite:
        jobs_to_run = jobs
    elif args.skip_existing:
        jobs_to_run = pending_jobs
    else:
        jobs_to_run = jobs

    if not jobs_to_run:
        raise RuntimeError("All selected integrated recovery runs already have outputs.")

    job_list_file = project_dir / "scripts" / ".integrated_model_recovery_job_list.txt"
    script = generate_slurm_script(job_list_file, len(jobs_to_run), args, project_dir, python_command)

    if args.dry_run:
        print("Job list (not written):")
        for job in jobs_to_run:
            row = (job["replicate"], job["simulated_model"], job["fitted_model"])
            print("\t".join(str(value) for value in row))
        print()
        print(script)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_job_list(jobs_to_run, job_list_file)

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
