#!/usr/bin/env python3
"""Submit parameter recovery jobs to SLURM.

This script starts the standalone parameter-recovery analysis. It checks that the
prepared data files exist, then submits one SLURM array job with one task for each
model and condition pair. Each task runs run_parameter_recovery.py, which simulates
participants with known parameters and refits the model. The models, conditions,
SLURM resources and MCMC settings that are not given on the command line come
from the pipeline config file, with built-in fallbacks. For the SLURM time and
memory, the config file's `parameter_recovery_time` and
`parameter_recovery_memory` take priority over its shared `time` and `memory`.
The number of simulated participants (50) and the data and output folders have
fixed defaults and are not read from the config file. Use `--dry-run` to print
the SLURM script without submitting it.

The `--interruptible-cpu` option selects the `interruptible_cpu` partition, a
partition name on the cluster where these analyses were run. On other clusters,
choose a partition with `--partition` instead.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from volatility_generalisation.modelling_code.model_simulation import MODEL_INFO
from volatility_generalisation.modelling_code.pipeline_config import (
    load_pipeline_config,
)


DEFAULT_CONFIG = "scripts/config.toml"
DEFAULT_DATA_DIR = "data/prepared"
DEFAULT_OUTPUT_DIR = "results/parameter_recovery"
DEFAULT_N_SUBJECTS = 50

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
INTERRUPTIBLE_PARTITION = "interruptible_cpu"
FALLBACK_PYTHON_COMMAND = "uv run --frozen python"

console = Console()


def setup_args() -> argparse.Namespace:
    """Read the command-line options for submitting parameter-recovery jobs.

    Covers the models and conditions, the data and output folders, the number of
    simulated participants, the SLURM resources, the MCMC settings, and whether
    to overwrite or skip existing results. The models, conditions, SLURM
    resources and MCMC settings are set to None when left out, so that the config
    file can supply them. The number of simulated participants is also None when
    left out, but resolve_settings then uses 50 and never reads it from the
    config file. The data and output folders have fixed defaults.

    Returns:
        The command-line options, one attribute per option.
    """
    parser = argparse.ArgumentParser(description="Submit parameter recovery jobs to SLURM")

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
        help="Prepared conditions to recover",
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
        default=None,
        help="Synthetic subjects per model-condition run",
    )

    parser.add_argument("--time", default=None, help="Wall time limit")
    parser.add_argument("--mem", default=None, help="Memory per job")
    parser.add_argument("--cpus", type=int, default=None, help="CPUs per job")
    parser.add_argument("--partition", default=None, help="SLURM partition")
    parser.add_argument(
        "--interruptible-cpu",
        action="store_true",
        help=(
            "Use the interruptible_cpu SLURM partition, a partition name on the "
            "cluster where these analyses were run. On other clusters, use "
            "--partition instead"
        ),
    )

    parser.add_argument("--num-warmup", type=int, default=None, help="Warmup steps")
    parser.add_argument("--num-samples", type=int, default=None, help="Sampling steps")
    parser.add_argument("--num-chains", type=int, default=None, help="Number of chains")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=None,
        help="NUTS target acceptance probability",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=None,
        help="Maximum tree depth for NUTS",
    )
    parser.add_argument("--thinning", type=int, default=None, help="MCMC thinning factor")

    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print script without submitting")
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    overwrite_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip model-condition runs that already have outputs",
    )

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


def resolve_settings(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """Fill in every option left out on the command line.

    For the models, conditions, SLURM resources and MCMC settings that are None,
    the value comes from the config file if it is set there, and from a built-in
    fallback otherwise. For the SLURM time and memory, the config file's
    `parameter_recovery_time` and `parameter_recovery_memory` take priority
    over its shared `time` and `memory`. The number of simulated participants
    is never read from the config file: if it is None, it is set to 50. The
    `--interruptible-cpu` flag sets the partition to the interruptible one. The
    model names are checked. The input is changed in place.

    Args:
        args: Command-line options from setup_args.
        config: Contents of the pipeline config file (may be empty).

    Returns:
        The same command-line options, with the models, conditions, number of
        simulated participants, SLURM resources and MCMC settings all filled in.
    """
    fitting_config = config.get("fitting", {})
    hpc_config = config.get("hpc", {})
    mcmc_config = config.get("mcmc", {})

    if args.models is None:
        args.models = fitting_config.get("models", FALLBACK_MODELS)
    args.models = validate_models(list(args.models))
    if args.conditions is None:
        args.conditions = fitting_config.get("conditions", FALLBACK_CONDITIONS)

    if args.n_subjects is None:
        args.n_subjects = DEFAULT_N_SUBJECTS

    if args.time is None:
        args.time = hpc_config.get(
            "parameter_recovery_time", hpc_config.get("time", FALLBACK_TIME)
        )
    if args.mem is None:
        args.mem = hpc_config.get(
            "parameter_recovery_memory", hpc_config.get("memory", FALLBACK_MEM)
        )
    if args.cpus is None:
        args.cpus = hpc_config.get("cpus_per_task", FALLBACK_CPUS)
    if args.partition is None:
        args.partition = hpc_config.get("partition", FALLBACK_PARTITION)
    if getattr(args, "interruptible_cpu", False):
        args.partition = INTERRUPTIBLE_PARTITION

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

    return args


def resolve_project_path(project_dir: Path, path_str: str) -> Path:
    """Turn a folder path into an absolute path, relative to the project folder.

    Args:
        project_dir: Path to the project folder.
        path_str: Folder path, either absolute or relative to the project folder.

    Returns:
        The path if it is already absolute. Otherwise the path joined to the
        project folder.
    """
    path = Path(path_str)
    return path if path.is_absolute() else project_dir / path


def validate_prepared_condition_files(data_dir: Path, conditions: Sequence[str]) -> None:
    """Check that a prepared data file exists for every condition.

    Args:
        data_dir: Folder that holds the prepared .npz files.
        conditions: Names of the conditions. Each needs a file named
            "<condition>.npz".

    Raises:
        RuntimeError: If any file is missing. The message lists the missing
            conditions and says to run prepare_data.py.
    """
    missing_paths = [data_dir / f"{condition}.npz" for condition in conditions if not (data_dir / f"{condition}.npz").exists()]
    if missing_paths:
        missing_labels = ", ".join(path.stem for path in missing_paths)
        raise RuntimeError(
            "Missing prepared condition files in "
            f"{data_dir}: {missing_labels}. Run `uv run python scripts/prepare_data.py` "
            "or pass `--data-dir` to the directory containing the prepared `.npz` files."
        )


def _run_dir(output_dir: Path, model: str, condition: str) -> Path:
    """Give the folder in which the recovery results for one pair are saved.

    Args:
        output_dir: Folder that holds all parameter-recovery results.
        model: Name of the model.
        condition: Name of the condition.

    Returns:
        Path to the sub-folder "<condition>/<model>".
    """
    return output_dir / condition / model


def _run_outputs_exist(run_dir: Path) -> bool:
    """Check whether a recovery run has already saved any of its result files.

    The result files are parameter_recovery.csv and
    parameter_recovery_summary.csv.

    Args:
        run_dir: Folder for one model and condition.

    Returns:
        True if at least one of the result files exists. Otherwise False.
    """
    return any(
        (run_dir / filename).exists()
        for filename in [
            "parameter_recovery.csv",
            "parameter_recovery_summary.csv",
        ]
    )


def build_job_pairs(
    models: Sequence[str],
    conditions: Sequence[str],
    output_dir: Path,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> list[tuple[str, str]]:
    """Pick the model and condition pairs that still need a recovery run.

    A pair counts as done if its results folder holds any recovery result file.
    With `overwrite`, finished pairs are run again. With `skip_existing`, they
    are left out. With neither, finding a finished pair is an error, so results
    are not overwritten by accident.

    Args:
        models: Names of the models to recover.
        conditions: Names of the conditions to use.
        output_dir: Folder that holds all parameter-recovery results.
        overwrite: If True, run pairs that already have results again.
        skip_existing: If True, leave out pairs that already have results.

    Returns:
        List of (model, condition) pairs to submit.

    Raises:
        RuntimeError: If some pairs already have results and neither option is
            set, or if `skip_existing` is set and every pair already has
            results.
    """
    pairs: list[tuple[str, str]] = []
    existing: list[str] = []

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
            "Existing parameter recovery outputs found for: " + ", ".join(existing)
        )

    if skip_existing and not pairs:
        raise RuntimeError("All selected parameter recovery runs already have outputs.")

    return pairs


def generate_job_list(job_pairs: Sequence[tuple[str, str]], filename: Path) -> None:
    """Write the job-list file that tells each SLURM task what to run.

    Line N holds the model name and the condition name, separated by a space,
    for SLURM array task N.

    Args:
        job_pairs: List of (model, condition) pairs to run.
        filename: Path of the file to write.
    """
    with open(filename, "w", encoding="utf-8") as handle:
        for model, condition in job_pairs:
            handle.write(f"{model} {condition}\n")


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
    run_parameter_recovery.py for that model and condition, with the settings in
    `args`. The function only builds the text; it does not create any folder or
    file.

    Args:
        job_list_file: Path to the job-list file written by generate_job_list.
        n_jobs: Number of array tasks.
        args: Command-line options after the defaults have been filled in. They
            supply the SLURM resources, the folder paths, the settings file, the
            number of simulated participants and the MCMC settings.
        project_dir: Path to the project folder.
        python_command: Command used to start Python on the cluster.

    Returns:
        Text of the SLURM batch script.

    Raises:
        RuntimeError: If the Python command uses `uv --frozen` and `uv` cannot
            be found on the current machine.
    """

    worker_script = project_dir / "scripts" / "run_parameter_recovery.py"

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir_expr = str(data_dir) if data_dir.is_absolute() else f'$PROJECT_DIR/{data_dir.as_posix()}'
    output_dir_expr = (
        str(output_dir) if output_dir.is_absolute() else f'$PROJECT_DIR/{output_dir.as_posix()}'
    )
    # Each job reads the same settings file that was used to submit it.
    config_path = Path(args.config).resolve()

    if "uv" in python_command and "--frozen" in python_command and not shutil_which("uv"):
        raise RuntimeError(
            "uv is not available on PATH. Install uv or configure [hpc] modules/pre_run_commands."
        )

    return f"""#!/bin/bash
#SBATCH --job-name=parameter_recovery
#SBATCH --output=logs/parameter_recovery_%A_%a.out
#SBATCH --error=logs/parameter_recovery_%A_%a.err
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --partition={args.partition}
#SBATCH --array=1-{n_jobs}

# Auto-generated by submit_parameter_recovery.py

set -e

PROJECT_DIR=\"{project_dir}\"
DATA_DIR=\"{data_dir_expr}\"
OUTPUT_DIR=\"{output_dir_expr}\"
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

JOB_PARAMS=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" \"$JOB_LIST\")
read -r MODEL CONDITION <<< \"$JOB_PARAMS\"

echo \"==========================================\"
echo \"Job ID: ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}\"
echo \"Model: $MODEL\"
echo \"Condition: $CONDITION\"
echo \"==========================================\"

{python_command} {worker_script} \\
    --models \"$MODEL\" \\
    --conditions \"$CONDITION\" \\
    --data-dir \"$DATA_DIR\" \\
    --output-dir \"$OUTPUT_DIR\" \\
    --config \"{config_path}\" \\
    --n-subjects {args.n_subjects} \\
    --num-warmup {args.num_warmup} \\
    --num-samples {args.num_samples} \\
    --num-chains {args.num_chains} \\
    --seed {args.seed} \\
    --target-accept-prob {args.target_accept_prob} \\
    --max-tree-depth {args.max_tree_depth} \\
    --thinning {args.thinning}
"""


def shutil_which(command: str) -> str | None:
    """Look for a program on the current machine's search path.

    Args:
        command: Name of the program, for example "uv".

    Returns:
        The full path to the program, or None if it is not found.
    """
    from shutil import which

    return which(command)


def render_submission_summary(
    args: argparse.Namespace,
    job_pairs: Sequence[tuple[str, str]],
    job_list_file: Path,
) -> None:
    """Print a panel that summarises the jobs about to be submitted.

    Args:
        args: Command-line options after the defaults have been filled in.
            They supply the models, conditions, MCMC settings and SLURM
            resources shown in the panel.
        job_pairs: List of (model, condition) pairs to submit. Only the number
            of pairs is shown.
        job_list_file: Path to the job-list file.
    """
    table = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("Jobs", str(len(job_pairs)))
    table.add_row("Models", ", ".join(args.models))
    table.add_row("Conditions", ", ".join(args.conditions))
    table.add_row("Subjects", str(args.n_subjects))
    table.add_row("Warmup", str(args.num_warmup))
    table.add_row("Samples", str(args.num_samples))
    table.add_row("Chains", str(args.num_chains))
    table.add_row("Seed", str(args.seed))
    table.add_row("Target Accept", str(args.target_accept_prob))
    table.add_row("Tree Depth", str(args.max_tree_depth))
    table.add_row("Thinning", str(args.thinning))
    table.add_row("Time", str(args.time))
    table.add_row("Memory", str(args.mem))
    table.add_row("CPUs", str(args.cpus))
    table.add_row("Partition", str(args.partition))
    table.add_row("Output", str(args.output_dir))
    table.add_row("Job List", str(job_list_file))

    console.print(
        Panel.fit(
            table,
            title="[bold green]Parameter Recovery Submission[/bold green]",
            border_style="green",
        )
    )


def main() -> None:
    """Build the parameter-recovery jobs and submit them to SLURM.

    Reads the command-line options and the config file, checks that the prepared
    data files exist, works out which model and condition pairs need running,
    and writes the job-list file. It then prints a summary and submits the array
    job with `sbatch`. With `--dry-run` it prints the job list and the SLURM
    script and stops before submitting, and writes no file and creates no
    folder. Errors are printed as messages and end the program with a nonzero
    exit code. A missing or unreadable config file, and an unsupported model
    name, stop the program with an error before anything is written.

    Raises:
        ValueError: If a model name is not supported.
        SystemExit: If a check fails, `sbatch` fails, or `sbatch` is not found.
            The exit code is 1, or the exit code of `sbatch` if it failed.
    """
    args = setup_args()
    config = load_pipeline_config(args.config)
    try:
        args = resolve_settings(args, config)

        project_dir = detect_project_dir()
        data_dir = resolve_project_path(project_dir, args.data_dir)
        output_dir = resolve_project_path(project_dir, args.output_dir)
        validate_prepared_condition_files(data_dir, args.conditions)

        job_pairs = build_job_pairs(
            args.models,
            args.conditions,
            output_dir,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
        )

        job_list_file = project_dir / "scripts" / ".parameter_recovery_job_list.txt"

        n_jobs = len(job_pairs)
        render_submission_summary(args, job_pairs, job_list_file)

        python_command = config.get("hpc", {}).get("python_command", FALLBACK_PYTHON_COMMAND)
        slurm_content = generate_slurm_script(
            job_list_file,
            n_jobs,
            args,
            project_dir,
            python_command,
        )
        slurm_file = project_dir / "scripts" / ".temp_parameter_recovery_submit.sh"

        if args.dry_run:
            console.print("[bold cyan]Job list (not written):[/bold cyan]")
            for model, condition in job_pairs:
                console.print(f"{model} {condition}", markup=False)
            console.print(
                Panel(
                    Syntax(slurm_content, "bash", word_wrap=True),
                    title="[bold cyan]Generated SLURM Script[/bold cyan]",
                    border_style="cyan",
                )
            )
            console.print(
                f"[yellow]Dry run only.[/yellow] No files were written. "
                "Use without [bold]--dry-run[/bold] to submit."
            )
            return

        generate_job_list(job_pairs, job_list_file)
        (project_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(slurm_file, "w", encoding="utf-8") as handle:
            handle.write(slurm_content)

        cmd = ["sbatch", str(slurm_file)]
        console.print(f"[cyan]Submitting:[/cyan] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        console.print("[bold green]Parameter recovery jobs submitted.[/bold green]")
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        console.print(f"[bold red]Submission failed:[/bold red] {exc}")
        raise SystemExit(exc.returncode) from exc
    except FileNotFoundError as exc:
        console.print(
            f"[bold red]Error:[/bold red] {exc}. Are you on a SLURM cluster with the required command available?"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
