"""
Single Model Fitting Script

Fits a single model to a single condition dataset. Designed to be run
in parallel via SLURM array jobs.

This is the fitting step of the pipeline. It reads the prepared data for one
condition (the output of prepare_data.py), fits one model with MCMC, checks the
chains, runs a posterior predictive check and saves the results. The results are
later collected by compile_results.py. The exit code is 0 if the chains
converged, 2 if they did not, and 1 if the data file is missing.

Usage:
    python fit_single.py --model rescorla_wagner_single_value --condition VolatileV1 \
                         --data-dir ../data/prepared \
                         --output-dir ../results \
                         --num-warmup 2000 --num-samples 4000 --num-chains 4
"""
import os
import argparse
from pathlib import Path
import numpy as np
import tempfile
import shutil
import atexit


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

# Make sure NumPyro sees a sensible number of CPU devices on both SLURM and
# local runs before any JAX-backed inference starts.
numpyro.set_platform("cpu")
numpyro.set_host_device_count(HOST_DEVICE_COUNT)

# --- HPC / Parallelization Fixes ---
# Download ArviZ example data into this job's temporary folder.
# The batch script sets XDG_CACHE_HOME separately for the import warning marker.
job_tmp_dir = tempfile.mkdtemp(prefix="arviz_tmp_")
os.environ["ARVIZ_DATA_DIR"] = job_tmp_dir

# Register cleanup of the temp directory on script exit
def cleanup_arviz_tmp():
    """Delete this job's temporary ArviZ folder when the script exits."""
    shutil.rmtree(job_tmp_dir, ignore_errors=True)
atexit.register(cleanup_arviz_tmp)

import jax
import jax.numpy as jnp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC

# Configure JAX for optimal parallel chain execution
# This allows NumPyro to run multiple chains in parallel across CPU cores
jax.config.update("jax_enable_x64", True)  # 64-bit precision for stability

# NumPyro chain execution falls back between parallel and vectorized modes in
# fit_mcmc() depending on how many local CPU devices JAX exposes.

from volatility_generalisation.modelling_code.data_preparation import (
    OUTCOME_CODING_KEY,
    OUTCOME_CODING_REWARD_SAFE,
    normalize_to_reward_coded,
)
from volatility_generalisation.modelling_code.data_generation import (
    settings_file_record,
)
from volatility_generalisation.modelling_code.hierarchical_bayes import fit_mcmc
from volatility_generalisation.modelling_code.plotting import (
    posterior_predictive_check,
    plot_ppc_results
)
from volatility_generalisation.modelling_code.diagnostics import (
    diagnose_mcmc,
    save_diagnostics
)

console = Console()


def load_prepared_data(data_path: Path) -> dict:
    """
    Load prepared data from .npz file.
    
    Args:
        data_path: Path to .npz file
        
    Returns:
        Dictionary with JAX arrays
    """
    console.print(f"[cyan]Loading data from:[/cyan] {data_path}")
    
    data_np = np.load(data_path, allow_pickle=True)

    # Build metadata dict from npz file attributes
    metadata = {}
    if "outcome_coding" in data_np:
        metadata["outcome_coding"] = str(data_np["outcome_coding"])

    # Load as numpy dict, normalise to reward-coded (handles files with and
    # without outcome-coding metadata)
    data_dict = {
        "outcomes_0": np.asarray(data_np["outcomes_0"], dtype=float),
        "outcomes_1": np.asarray(data_np["outcomes_1"], dtype=float),
        "choices": np.asarray(data_np["choices"], dtype=float),
    }
    data_dict = normalize_to_reward_coded(data_dict, metadata=metadata)

    # Convert to JAX arrays
    data_jax = {
        'outcomes_0': jnp.array(data_dict['outcomes_0']),
        'outcomes_1': jnp.array(data_dict['outcomes_1']),
        'choices': jnp.array(data_dict['choices']),
        'subject_ids': data_np['subject_ids'].tolist() if 'subject_ids' in data_np else None,
        OUTCOME_CODING_KEY: OUTCOME_CODING_REWARD_SAFE,
    }

    console.print(f"  Data shape: {data_jax['choices'].shape}")
    console.print(f"  Subjects: {data_jax['choices'].shape[0]}, Trials: {data_jax['choices'].shape[1]}")

    return data_jax


def fit_and_diagnose(
    data: dict,
    model_name: str,
    model_config: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    target_accept_prob: float = 0.95,
    max_tree_depth: int = 10,
    thinning: int = 1,
):
    """
    Fit model and run diagnostics.
    
    Args:
        data: Prepared data dictionary
        model_name: Name of model to fit
        model_config: Path to the TOML file with the model parameter bounds
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        seed: Random seed
        target_accept_prob: NUTS target acceptance probability
        max_tree_depth: Maximum NUTS tree depth
        thinning: Thinning factor for MCMC samples
        
    Returns:
        Tuple of (mcmc, posterior_samples, diagnostics)
    """
    console.print(f"\n[bold green]Fitting {model_name} model...[/bold green]")
    console.print(f"  Parallel execution: {num_chains} chains across {jax.local_device_count()} device(s)")
    
    # Run MCMC with parallel chains
    # NumPyro automatically distributes chains across available CPU cores
    mcmc = fit_mcmc(
        data=data,
        model_names=model_name,
        config_path=model_config,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        thinning=thinning,
        progress_bar=True,
    )
    
    console.print("[green]✓ MCMC complete![/green]")
    
    # Get posterior samples
    posterior_samples = mcmc.get_samples(group_by_chain=True)
    
    # Run diagnostics
    console.print("\n[bold cyan]Running MCMC diagnostics...[/bold cyan]")
    
    # Get parameter names for this model
    param_names = [k for k in posterior_samples.keys() 
                   if k.startswith(f"{model_name}_") and k.endswith("_subj")]
    
    diagnostics = diagnose_mcmc(
        mcmc=mcmc,
        posterior_samples=posterior_samples,
        param_names=param_names,
        verbose=True
    )
    if not diagnostics['converged']:
        from volatility_generalisation.modelling_code.diagnostics import print_chain_failure_summary
        print_chain_failure_summary(posterior_samples, diagnostics)
    
    return mcmc, posterior_samples, diagnostics


def run_posterior_predictive_check(
    data: dict,
    posterior_samples: dict,
    model_name: str,
    output_dir: Path,
    condition_name: str,
    n_posterior_samples: int = 200,
):
    """
    Run posterior predictive checks and save plots.

    Args:
        data: Prepared data dictionary
        posterior_samples: MCMC posterior samples
        model_name: Name of fitted model
        output_dir: Directory to save plots
        condition_name: Name of condition for plot title
        n_posterior_samples: Number of posterior samples to use for PPD

    Returns:
        PPC results dictionary
    """
    console.print(f"\n[bold cyan]Running posterior predictive checks ({n_posterior_samples} samples)...[/bold cyan]")

    ppc_results = posterior_predictive_check(
        data=data,
        posterior_samples=posterior_samples,
        model_name=model_name,
        n_posterior_samples=n_posterior_samples,
        verbose=True,
    )

    ppc_results['subject_ids'] = data.get('subject_ids')
    console.print(f"[green]✓ PPC complete! Accuracy: {ppc_results['accuracy']:.3f}[/green]")

    # Generate and save plot
    fig = plot_ppc_results(ppc_results, condition_name=condition_name)

    plot_path = output_dir / f"ppc_{condition_name}_{model_name}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    console.print(f"[green]✓ Saved PPC plot to {plot_path}[/green]")

    return ppc_results


def extract_parameter_summaries(posterior_samples: dict, model_name: str) -> dict:
    """
    Summarise each participant-level parameter of the fitted model.

    For each parameter named `<model>_<parameter>_subj`, the function first takes
    each participant's posterior mean (the average over all samples and chains).
    It then reports the mean, median, standard deviation, minimum and maximum of
    those posterior means across participants. Parameters that are not
    participant-level are skipped.

    Args:
        posterior_samples: MCMC posterior samples
        model_name: Name of fitted model

    Returns:
        Dictionary mapping each parameter name to a dictionary with the keys
        `mean`, `median`, `std`, `min` and `max` (across participants) and
        `subject_means` (the list of each participant's posterior mean).
    """
    summaries = {}
    
    # Find all subject-level parameters for this model
    for key in posterior_samples.keys():
        if key.startswith(f"{model_name}_") and key.endswith("_subj"):
            param_name = key.replace(f"{model_name}_", "").replace("_subj", "")
            values = np.asarray(posterior_samples[key])
            if values.ndim == 1:
                values = values[:, None]
            elif values.ndim > 2:
                values = values.reshape(-1, values.shape[-1])
            
            # Compute mean across all posterior samples after flattening any
            # leading chain/draw dimensions.
            mean_values = values.mean(axis=0)
            
            summaries[param_name] = {
                'mean': float(np.mean(mean_values)),
                'median': float(np.median(mean_values)),
                'std': float(np.std(mean_values)),
                'min': float(np.min(mean_values)),
                'max': float(np.max(mean_values)),
                'subject_means': mean_values.tolist(),
            }
    
    return summaries


def save_results(
    output_dir: Path,
    condition_name: str,
    model_name: str,
    settings_record: dict,
    posterior_samples: dict,
    ppc_results: dict,
    diagnostics: dict,
    parameter_summaries: dict
):
    """
    Save all results to disk.
    
    Args:
        output_dir: Directory to save results
        condition_name: Name of condition
        model_name: Name of model
        settings_record: The bounds file's path and SHA-256 hash, from
            ``settings_file_record``, stored in the metadata.
        posterior_samples: MCMC posterior samples
        ppc_results: PPC results
        diagnostics: Diagnostics dictionary
        parameter_summaries: Parameter summaries
    """
    console.print(f"\n[bold cyan]Saving results...[/bold cyan]")
    
    # Create subdirectory for this condition-model pair
    result_dir = output_dir / f"{condition_name}_{model_name}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save posterior samples (compressed)
    samples_path = result_dir / "posterior_samples.npz"
    np.savez_compressed(samples_path, **{k: np.array(v) for k, v in posterior_samples.items()})
    console.print(f"  ✓ Posterior samples: {samples_path}")
    
    # Save PPC results (exclude non-array fields like subject_ids)
    ppc_path = result_dir / "ppc_results.npz"
    np.savez_compressed(ppc_path, **{k: np.array(v) for k, v in ppc_results.items() if k != 'subject_ids'})
    console.print(f"  ✓ PPC results: {ppc_path}")
    
    # Save diagnostics
    diagnostics_path = result_dir / "diagnostics.json"
    save_diagnostics(diagnostics, diagnostics_path)
    console.print(f"  ✓ Diagnostics: {diagnostics_path}")
    
    # Save parameter summaries
    summary_path = result_dir / "parameter_summaries.json"
    with open(summary_path, 'w') as f:
        json.dump(parameter_summaries, f, indent=2)
    console.print(f"  ✓ Parameter summaries: {summary_path}")
    
    # Save metadata
    metadata = {
        'condition': condition_name,
        'model': model_name,
        'converged': diagnostics['converged'],
        'accuracy': float(ppc_results['accuracy']),
        'n_subjects': int(ppc_results['actual_choices'].shape[0]),
        'n_trials': int(ppc_results['actual_choices'].shape[1]),
        **settings_record,
        'subject_ids': ppc_results.get('subject_ids'),
    }
    metadata_path = result_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    console.print(f"  ✓ Metadata: {metadata_path}")
    
    console.print(f"\n[bold green]✓ All results saved to {result_dir}[/bold green]")


def main():
    """Fit one model to one condition and save the results.

    Reads the command-line options, loads the prepared data for the condition,
    fits the model with MCMC, runs the chain diagnostics and the posterior
    predictive check, and saves the results. The results go in a folder named
    `<condition>_<model>` inside the output directory. The posterior predictive
    check figure is saved beside that folder, in the output directory itself, as
    `ppc_<condition>_<model>.png`. The results are saved even if the chains did
    not converge.

    Returns:
        Exit code: 0 if the chains converged, 1 if the prepared data file for
        the condition was not found, and 2 if the chains did not converge.
    """
    parser = argparse.ArgumentParser(
        description="Fit a single model to a single condition"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=[
            'rescorla_wagner_single_value',
            'dual_lr_single_value',
            'pearce_hall_single_value',
            'leaky_beta_single_value',
            'leaky_beta_asymmetric_single_value',
        ],
        help='Model to fit'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='config/model_bounds.toml',
        help='Model bounds config TOML'
    )
    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        help='Condition name (e.g., VolatileV1, StableV1)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/prepared',
        help='Directory containing prepared .npz files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=2000,
        help='Number of MCMC warmup samples'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=4000,
        help='Number of MCMC posterior samples'
    )
    parser.add_argument(
        '--num-chains',
        type=int,
        default=4,
        help='Number of MCMC chains'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--target-accept-prob',
        type=float,
        default=0.95,
        help='NUTS target acceptance probability'
    )
    parser.add_argument(
        '--thinning',
        type=int,
        default=1,
        help='Thinning factor for MCMC samples'
    )
    parser.add_argument(
        '--max-tree-depth',
        type=int,
        default=10,
        help='Maximum NUTS tree depth'
    )
    
    args = parser.parse_args()
    settings_record = settings_file_record(args.model_config)
    
    # Print header
    console.print("\n[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]   Model Fitting Pipeline[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]\n")
    
    console.print(f"[bold]Model:[/bold] {args.model}")
    console.print(f"[bold]Condition:[/bold] {args.condition}")
    console.print(f"[bold]Model config:[/bold] {args.model_config}")
    console.print(f"[bold]MCMC:[/bold] {args.num_warmup} warmup + {args.num_samples} samples × {args.num_chains} chains")
    console.print(f"[bold]Host devices:[/bold] {HOST_DEVICE_COUNT}")
    console.print(f"[bold]Seed:[/bold] {args.seed}\n")
    
    # Create the job-specific output directory immediately
    # so that monitoring scripts can detect the job is running
    output_dir_base = Path(args.output_dir)
    result_dir = output_dir_base / f"{args.condition}_{args.model}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path(args.data_dir) / f"{args.condition}.npz"
    if not data_path.exists():
        console.print(f"[bold red]Error: Data file not found: {data_path}[/bold red]")
        return 1
    
    data = load_prepared_data(data_path)
    
    # Fit model and diagnose
    mcmc, posterior_samples, diagnostics = fit_and_diagnose(
        data=data,
        model_name=args.model,
        model_config=args.model_config,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        thinning=args.thinning,
    )
    
    # Run PPC
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ppc_results = run_posterior_predictive_check(
        data=data,
        posterior_samples=posterior_samples,
        model_name=args.model,
        output_dir=output_dir,
        condition_name=args.condition
    )
    
    # Extract parameter summaries
    parameter_summaries = extract_parameter_summaries(posterior_samples, args.model)
    
    console.print(f"\n[bold cyan]Parameter Summaries:[/bold cyan]")
    for param_name, summary in parameter_summaries.items():
        console.print(f"  {param_name}: Mean={summary['mean']:.4f}, SD={summary['std']:.4f}")
    
    # Save all results
    save_results(
        output_dir=output_dir,
        condition_name=args.condition,
        model_name=args.model,
        settings_record=settings_record,
        posterior_samples=posterior_samples,
        ppc_results=ppc_results,
        diagnostics=diagnostics,
        parameter_summaries=parameter_summaries
    )
    
    # Final status
    if diagnostics['converged']:
        console.print("\n[bold green]✓ Job completed successfully![/bold green]\n")
        return 0
    else:
        console.print("\n[bold yellow]⚠ Job completed with convergence warnings[/bold yellow]\n")
        return 2


if __name__ == '__main__':
    exit(main())
