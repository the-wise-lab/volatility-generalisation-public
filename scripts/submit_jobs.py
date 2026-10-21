#!/usr/bin/env python3
"""
User-friendly SLURM Job Submission Script

This script allows you to submit model fitting jobs to the HPC scheduler
with an easy-to-use command line interface. It automates the creation
of SLURM job scripts and handles array job indexing for you.

This script starts the fitting step of the pipeline. After prepare_data.py has
written the prepared data, it submits one SLURM array job with one task for each
model and condition pair. Each task runs fit_single.py. The models, conditions,
SLURM resources and the number of warmup steps, samples and chains that are not
given on the command line come from the pipeline config file, with built-in
fallbacks. The random seed, target acceptance, tree depth and thinning come only
from the config file. Follow the jobs with squeue and sacct.

Usage:
    # Submit all models and conditions (default)
    uv run python scripts/submit_jobs.py

    # Submit specific models
    uv run python scripts/submit_jobs.py --models rescorla_wagner_single_value pearce_hall_single_value

    # Submit specific conditions
    uv run python scripts/submit_jobs.py --conditions VolatileV1 StableV1

    # Configure resources
    uv run python scripts/submit_jobs.py --time "02:00:00" --mem "32G"

    # Use an alternate partition
    uv run python scripts/submit_jobs.py --partition <your_partition>

    # Dry run (print script but don't submit)
    uv run python scripts/submit_jobs.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os
from typing import List

from volatility_generalisation.modelling_code.model_simulation import MODEL_INFO
from volatility_generalisation.modelling_code.pipeline_config import (
    load_pipeline_config,
)

DEFAULT_CONFIG = "scripts/config.toml"
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
FALLBACK_TIME = "04:00:00"
FALLBACK_MEM = "16G"
FALLBACK_CPUS = 4
FALLBACK_PARTITION = "cpu"
FALLBACK_WARMUP = 2000
FALLBACK_SAMPLES = 4000
FALLBACK_CHAINS = 4
FALLBACK_PYTHON_COMMAND = "uv run --frozen python"


def setup_args():
    """Read the command-line options for submitting the fitting jobs.

    Covers the models and conditions to fit, the SLURM resources, the MCMC
    settings, and whether to overwrite or skip existing results. The models,
    conditions, SLURM resources and MCMC settings are set to None when left out,
    so that the config file can supply them. The random seed, target acceptance,
    tree depth and thinning are not command-line options.

    Returns:
        The command-line options, one attribute per option.
    """
    parser = argparse.ArgumentParser(description="Submit model fitting jobs to SLURM")

    # Selection arguments
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=RETAINED_MODELS,
        help="Models to fit",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None, help="Conditions to fit"
    )

    # Resource arguments
    parser.add_argument("--time", default=None, help="Wall time limit")
    parser.add_argument("--mem", default=None, help="Memory per job")
    parser.add_argument("--cpus", type=int, default=None, help="CPUs per job")
    parser.add_argument("--partition", default=None, help="SLURM partition to use")

    # MCMC arguments
    parser.add_argument("--warmup", type=int, default=None, help="Warmup steps")
    parser.add_argument("--samples", type=int, default=None, help="Sampling steps")
    parser.add_argument("--chains", type=int, default=None, help="Number of chains")

    # Operational arguments
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="Path to configuration file"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print script without submitting"
    )
    parser.add_argument(
        "--script-dir", default="scripts", help="Directory containing fit_single.py"
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    overwrite_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip model-condition runs that already have outputs",
    )

    return parser.parse_args()


def validate_models(models: List[str]) -> List[str]:
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


def generate_job_list(job_pairs: List[tuple[str, str]], filename: Path):
    """Write the job-list file that tells each SLURM task what to fit.

    Line N holds the model name and the condition name, separated by a space,
    for SLURM array task N.

    Args:
        job_pairs: List of (model, condition) pairs to fit.
        filename: Path of the file to write.
    """
    with open(filename, "w") as f:
        for model, condition in job_pairs:
            f.write(f"{model} {condition}\n")


def _run_dir(output_dir: Path, model: str, condition: str) -> Path:
    """Give the folder in which the results for one model and condition are saved.

    Args:
        output_dir: Folder that holds all fitting results.
        model: Name of the model.
        condition: Name of the condition.

    Returns:
        Path to the sub-folder named "<condition>_<model>".
    """
    return output_dir / f"{condition}_{model}"


def _run_outputs_exist(run_dir: Path) -> bool:
    """Check whether a fit has already saved any of its result files.

    The result files are metadata.json, diagnostics.json,
    parameter_summaries.json, posterior_samples.npz and ppc_results.npz.

    Args:
        run_dir: Folder for one model and condition.

    Returns:
        True if at least one of the result files exists. Otherwise False.
    """
    return any(
        (run_dir / filename).exists()
        for filename in [
            "metadata.json",
            "diagnostics.json",
            "parameter_summaries.json",
            "posterior_samples.npz",
            "ppc_results.npz",
        ]
    )


def build_job_pairs(
    models: List[str],
    conditions: List[str],
    output_dir: Path,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> List[tuple[str, str]]:
    """Pick the model and condition pairs that still need to be fitted.

    A pair counts as already fitted if its results folder holds any result file.
    With `overwrite`, fitted pairs are refitted. With `skip_existing`, they are
    left out. With neither, finding a fitted pair is an error, so results are not
    overwritten by accident.

    Args:
        models: Names of the models to fit.
        conditions: Names of the conditions to fit.
        output_dir: Folder that holds all fitting results.
        overwrite: If True, refit pairs that already have results.
        skip_existing: If True, leave out pairs that already have results.

    Returns:
        List of (model, condition) pairs to submit.

    Raises:
        RuntimeError: If some pairs already have results and neither option is
            set, or if `skip_existing` is set and every pair already has
            results.
    """
    pairs: List[tuple[str, str]] = []
    existing: List[str] = []

    for model in models:
        for condition in conditions:
            run_dir = _run_dir(output_dir, model, condition)
            if _run_outputs_exist(run_dir):
                label = f"{model} x {condition}"
                if overwrite:
                    pairs.append((model, condition))
                elif skip_existing:
                    continue
                else:
                    existing.append(label)
            else:
                pairs.append((model, condition))

    if existing:
        raise RuntimeError(
            "Existing model-fitting outputs found for: " + ", ".join(existing)
        )

    if skip_existing and not pairs:
        raise RuntimeError("All selected model-fitting runs already have outputs.")

    return pairs


def generate_slurm_script(
    job_list_file: Path,
    n_jobs: int,
    args: argparse.Namespace,
    project_dir: Path,
    modules: List[str],
    pre_run_commands: List[str],
    python_command: str,
    seed: int = 42,
    target_accept: float = 0.95,
    max_tree_depth: int = 10,
    thinning: int = 1,
) -> str:
    """Write the text of the SLURM array script for the fitting jobs.

    Each array task reads its line of the job-list file and runs fit_single.py
    for that model and condition, with a separate cache directory for each task
    to prevent ArviZ imports from sharing their warning marker. The uv cache
    stays shared. The function only builds the text; it does not create any
    folder or file.

    Args:
        job_list_file: Path to the job-list file written by generate_job_list.
        n_jobs: Number of array tasks.
        args: Command-line options after the defaults have been filled in from
            the config. They supply the SLURM time, memory, CPUs and partition,
            the number of warmup steps, samples and chains, and the folder that
            holds fit_single.py.
        project_dir: Path to the project folder.
        modules: Names of the cluster modules to load before running.
        pre_run_commands: Shell commands to run before the fit starts.
        python_command: Command used to start Python on the cluster.
        seed: Random seed for the fits.
        target_accept: Target acceptance probability for the NUTS sampler.
        max_tree_depth: Maximum tree depth for the NUTS sampler.
        thinning: Keep every n-th posterior sample.

    Returns:
        Text of the SLURM batch script.
    """

    fit_script = project_dir / args.script_dir / "fit_single.py"
    module_lines = "\n".join(f"module load {module}" for module in modules)
    pre_run_lines = "\n".join(pre_run_commands)
    if not module_lines:
        module_lines = "# No HPC modules configured."
    if not pre_run_lines:
        pre_run_lines = "# No pre-run commands configured."

    script = f"""#!/bin/bash
#SBATCH --job-name=volatility_fit

#SBATCH --output=logs/fit_%A_%a.out
#SBATCH --error=logs/fit_%A_%a.err
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --partition={args.partition}
#SBATCH --array=1-{n_jobs}

# Auto-generated by submit_jobs.py

set -e

# Configuration
PROJECT_DIR="{project_dir}"
DATA_DIR="$PROJECT_DIR/data/prepared"
OUTPUT_DIR="$PROJECT_DIR/results"
MODEL_CONFIG="$PROJECT_DIR/config/model_bounds.toml"
JOB_LIST="{job_list_file}"

# Ensure uv finds pyproject.toml and uv.lock.
cd "$PROJECT_DIR"

# ===== Load Modules / Pre-run Commands (HPC-specific) =====
{module_lines}
{pre_run_lines}

if [[ "{python_command}" == uv* ]] && ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not available on PATH. Install uv or configure [hpc] modules/pre_run_commands."
    exit 1
fi

# Get job details from the job list file (1-indexed array task id matches line number)
# sed -n '${{SLURM_ARRAY_TASK_ID}}p' prints the specific line
JOB_PARAMS=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$JOB_LIST")
read -r MODEL CONDITION <<< "$JOB_PARAMS"

echo "=========================================="
echo "Job ID: ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "Model: $MODEL"
echo "Condition: $CONDITION"
echo "=========================================="

# Environment Setup (JAX / Parallelization)
# We set thread counts to 1 because NumPyro runs chains in parallel processes.
# If we allowed each chain to use multiple threads, we would oversubscribe the CPUs.
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Keep uv's shared cache before giving each task a separate ArviZ cache.
export UV_CACHE_DIR="${{UV_CACHE_DIR:-${{XDG_CACHE_HOME:-$HOME/.cache}}/uv}}"
export XDG_CACHE_HOME="${{TMPDIR:-/tmp}}/cache_${{SLURM_JOB_ID}}"
# specific XLA flags can remain as they invoke the parallelism usage
# We set xla_force_host_platform_device_count to ensure NumPyro sees multiple devices for parallel chains
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count={args.cpus}"

# Run Fitting Script
{python_command} {fit_script} \\
    --model "$MODEL" \\
    --condition "$CONDITION" \\
    --model-config "$MODEL_CONFIG" \\
    --data-dir "$DATA_DIR" \\
    --output-dir "$OUTPUT_DIR" \\
    --num-warmup {args.warmup} \\
    --num-samples {args.samples} \\
    --num-chains {args.chains} \\
    --seed {seed} \\
    --target-accept-prob {target_accept} \\
    --max-tree-depth {max_tree_depth} \\
    --thinning {thinning}

"""
    return script


def main():
    """Build the fitting jobs and submit them to SLURM as one array job.

    Reads the command-line options and the config file. The config file must
    exist and be readable; otherwise the function raises an error before it
    writes anything. For the models, conditions, SLURM resources and the number
    of warmup steps, samples and chains, the command line takes priority, then
    the config file, then the built-in fallback for a setting the file leaves
    out. For the SLURM time and memory, the config file's fit-specific
    `fit_time` and `fit_memory` take priority over its shared `time` and
    `memory`. The random seed, target acceptance, tree depth and thinning come
    only from the config file or the built-in fallback. The function works out
    which model and condition pairs need fitting, writes the job-list file and
    the SLURM script, and submits the script with `sbatch`. With `--dry-run` it
    prints the job list and the script, and writes no file and creates no
    folder. If `sbatch` fails or is not found, it exits with a nonzero exit
    code.
    """
    args = setup_args()

    # --- Load Configuration ---
    print(f"Loading configuration from {args.config}")
    config = load_pipeline_config(args.config)

    # --- Resolve Settings (CLI > Config > Fallback) ---

    # Fitting Settings
    fit_config = config.get(
        "fitting", {}
    )  # Note: typo 'fiting' in my config file, assume 'fitting'
    # Wait, I wrote `[fiting]` in key, I should fix that typo in config.toml later.
    # I'll check both keys just in case or fix the config file now.
    if "fitting" not in config and "fiting" in config:
        fit_config = config["fiting"]

    if args.models is None:
        args.models = fit_config.get("models", FALLBACK_MODELS)
    args.models = validate_models(list(args.models))

    if args.conditions is None:
        args.conditions = fit_config.get("conditions", FALLBACK_CONDITIONS)

    # Resource Settings
    hpc_config = config.get("hpc", {})
    # Fits use the fit-specific time and memory when the config gives them,
    # otherwise the shared [hpc] values used by the other job families.
    if args.time is None:
        args.time = hpc_config.get("fit_time", hpc_config.get("time", FALLBACK_TIME))
    if args.mem is None:
        args.mem = hpc_config.get("fit_memory", hpc_config.get("memory", FALLBACK_MEM))
    if args.cpus is None:
        args.cpus = hpc_config.get("cpus_per_task", FALLBACK_CPUS)
    if args.partition is None:
        args.partition = hpc_config.get("partition", FALLBACK_PARTITION)
    python_command = hpc_config.get("python_command", FALLBACK_PYTHON_COMMAND)
    modules = hpc_config.get("modules", [])
    pre_run_commands = hpc_config.get("pre_run_commands", [])

    # MCMC Settings
    mcmc_config = config.get("mcmc", {})
    if args.warmup is None:
        args.warmup = mcmc_config.get("num_warmup", FALLBACK_WARMUP)
    if args.samples is None:
        args.samples = mcmc_config.get("num_samples", FALLBACK_SAMPLES)
    if args.chains is None:
        args.chains = mcmc_config.get("num_chains", FALLBACK_CHAINS)

    # Determine project root (assume this script is in scripts/ or project root)
    # We want the absolute path to the project root
    current_dir = Path.cwd()
    if (current_dir / "pyproject.toml").exists():
        project_dir = current_dir
    elif (current_dir.parent / "pyproject.toml").exists():
        project_dir = current_dir.parent
    else:
        print("Warning: Could not determine project root. Assuming current directory.")
        project_dir = current_dir

    output_dir = project_dir / "results"

    job_pairs = build_job_pairs(
        args.models,
        args.conditions,
        output_dir,
        overwrite=getattr(args, "overwrite", False),
        skip_existing=getattr(args, "skip_existing", False),
    )

    # 1. Choose where the job list goes (it is written below, not on a dry run)
    job_list_file = project_dir / "scripts" / ".job_list.txt"

    # Calculate number of jobs
    n_jobs = len(job_pairs)
    print(
        f"Prepared {n_jobs} jobs from {len(args.models)} models and {len(args.conditions)} conditions"
    )

    # 2. Generate SLURM Script
    mcmc_config = config.get("mcmc", {})
    seed = mcmc_config.get("seed", 42)
    target_accept = mcmc_config.get("target_accept_prob", 0.95)
    max_tree_depth = mcmc_config.get("max_tree_depth", 10)
    thinning = mcmc_config.get("thinning", 1)

    slurm_content = generate_slurm_script(
        job_list_file,
        n_jobs,
        args,
        project_dir,
        modules,
        pre_run_commands,
        python_command,
        seed=seed,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        thinning=thinning,
    )
    slurm_file = project_dir / "scripts" / ".temp_submit.sh"

    if args.dry_run:
        print("\n--- Job list (not written) ---")
        for model, condition in job_pairs:
            print(f"{model} {condition}")
        print("\n--- Generated SLURM Script ---")
        print(slurm_content)
        print("------------------------------")
        print("Dry run: no files were written.")
        print("Use without --dry-run to submit.")
    else:
        generate_job_list(job_pairs, job_list_file)
        (project_dir / "logs").mkdir(exist_ok=True)
        # Write script to file
        with open(slurm_file, "w") as f:
            f.write(slurm_content)

        # Submit
        try:
            cmd = ["sbatch", str(slurm_file)]
            print(f"Submitting: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("\nSuccess! Jobs submitted.")
            print("Check job status with: squeue -u $USER (and sacct afterwards)")
        except subprocess.CalledProcessError as e:
            sys.exit(f"Error submitting job: {e}")
        except FileNotFoundError:
            sys.exit("Error: 'sbatch' command not found. Are you on a SLURM cluster?")


if __name__ == "__main__":
    main()
