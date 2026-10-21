"""
Convergence and quality checks for MCMC samples.

Functions for checking convergence and quality of MCMC samples.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import arviz as az
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def diagnose_mcmc(
    mcmc,
    posterior_samples: Dict[str, np.ndarray],
    param_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive MCMC diagnostics using ArviZ.

    With more than one chain, this computes R-hat and the bulk effective
    sample size for each parameter (the largest R-hat and the smallest
    effective sample size across the parameter's elements). A parameter with
    R-hat above 1.01 marks the run as not converged. So does a parameter whose
    R-hat or effective sample size could not be computed (NaN in any element),
    with a warning that names the parameter. A finite effective sample size
    below 400 only adds a warning. Any divergent transitions also mark the run
    as not converged. With a single chain, R-hat cannot be computed, so every
    checked parameter marks the run as not converged, and the effective sample
    size is the number of draws. Nothing is raised here: not converging is a
    result that is recorded in the returned dictionary, not an error.

    Args:
        mcmc: NumPyro MCMC object
        posterior_samples: Dictionary of posterior samples,
            chain-structured from mcmc.get_samples(group_by_chain=True).
            Each value shape: (n_chains, n_samples, ...)
        param_names: List of parameter names to check. If None, checks every
            parameter except names starting with 'log_' and the name
            'diverging'.
        verbose: Whether to print diagnostic table

    Returns:
        Dictionary with these entries:
        - converged: False if any R-hat is above 1.01, any R-hat or effective
          sample size is NaN, or any transition diverged, otherwise True
        - warnings: list of text messages describing each problem found
        - rhat: R-hat for each checked parameter
        - ess: effective sample size for each checked parameter
        - divergences: number of divergent transitions
    """
    diagnostics = {
        'converged': True,
        'warnings': [],
        'rhat': {},
        'ess': {},
        'divergences': 0,
    }

    # Check for divergences
    if hasattr(mcmc, 'get_extra_fields'):
        extra_fields = mcmc.get_extra_fields()
        if 'diverging' in extra_fields:
            diagnostics['divergences'] = int(np.sum(extra_fields['diverging']))

    # If no param_names specified, check all non-log-likelihood parameters
    if param_names is None:
        param_names = [k for k in posterior_samples.keys()
                      if not k.startswith('log_') and k != 'diverging']

    n_chains = getattr(mcmc, 'num_chains', 1)

    if n_chains > 1:
        # Filter to requested params for arviz
        arviz_input = {
            k: np.asarray(v)
            for k, v in posterior_samples.items()
            if k in param_names
        }

        if arviz_input:
            rhat_dict = az.rhat(arviz_input)
            ess_dict = az.ess(arviz_input, method='bulk')

            for param_name in param_names:
                if param_name not in rhat_dict:
                    continue

                # NaN in any element carries through to the parameter's value.
                rhat_val = float(np.max(rhat_dict[param_name].values))
                ess_val = float(np.min(ess_dict[param_name].values))

                diagnostics['rhat'][param_name] = rhat_val
                diagnostics['ess'][param_name] = ess_val

                if np.isnan(rhat_val):
                    diagnostics["converged"] = False
                    diagnostics["warnings"].append(
                        f"{param_name}: R-hat could not be computed (NaN)"
                    )
                elif rhat_val > 1.01:
                    diagnostics['converged'] = False
                    diagnostics['warnings'].append(
                        f"{param_name}: R-hat = {rhat_val:.4f} (should be < 1.01)"
                    )

                if np.isnan(ess_val):
                    diagnostics["converged"] = False
                    diagnostics["warnings"].append(
                        f"{param_name}: ESS could not be computed (NaN)"
                    )
                elif ess_val < 400:
                    diagnostics['warnings'].append(
                        f"{param_name}: ESS = {ess_val:.0f} (should be > 400)"
                    )
    else:
        for param_name in param_names:
            if param_name not in posterior_samples:
                continue
            samples = posterior_samples[param_name]
            diagnostics['rhat'][param_name] = np.nan
            diagnostics['ess'][param_name] = samples.shape[0] * samples.shape[1]
            diagnostics["converged"] = False
            diagnostics["warnings"].append(
                f"{param_name}: R-hat could not be computed (NaN) with one chain"
            )

    # Check divergences
    if diagnostics['divergences'] > 0:
        diagnostics['converged'] = False
        diagnostics['warnings'].append(
            f"Found {diagnostics['divergences']} divergent transitions"
        )

    # Print diagnostics table if verbose
    if verbose:
        print_diagnostics(diagnostics, param_names)

    return diagnostics


def print_diagnostics(diagnostics: Dict, param_names: List[str]):
    """
    Print a formatted diagnostics table using rich.
    
    Args:
        diagnostics: Dictionary from diagnose_mcmc
        param_names: List of parameter names
    """
    console.print("\n[bold cyan]═══ MCMC Diagnostics ═══[/bold cyan]\n")
    
    # Overall status
    if diagnostics['converged']:
        console.print("[bold green]✓ Convergence: PASSED[/bold green]")
    else:
        console.print("[bold red]✗ Convergence: FAILED[/bold red]")
    
    # Divergences
    if diagnostics['divergences'] > 0:
        console.print(f"[bold red]✗ Divergences: {diagnostics['divergences']}[/bold red]")
    else:
        console.print("[bold green]✓ Divergences: 0[/bold green]")
    
    # Parameter table
    table = Table(title="\nParameter Diagnostics", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("R-hat", justify="right")
    table.add_column("ESS", justify="right")
    table.add_column("Status", justify="center")
    
    for param_name in param_names:
        if param_name in diagnostics['rhat']:
            rhat = diagnostics['rhat'][param_name]
            ess = diagnostics['ess'][param_name]
            
            # Format values
            rhat_str = f"{rhat:.4f}" if not np.isnan(rhat) else "N/A"
            ess_str = f"{ess:.0f}" if not np.isnan(ess) else "N/A"
            
            # Determine status
            status = "✓"
            style = "green"
            if np.isnan(rhat) or np.isnan(ess):
                status = "✗ NaN"
                style = "red"
            elif rhat > 1.01:
                status = "✗ R-hat"
                style = "red"
            elif not np.isnan(ess) and ess < 400:
                status = "⚠ ESS"
                style = "yellow"
            
            table.add_row(param_name, rhat_str, ess_str, f"[{style}]{status}[/{style}]")
    
    console.print(table)
    
    # Print warnings
    if diagnostics['warnings']:
        console.print("\n[bold yellow]⚠ Warnings:[/bold yellow]")
        for warning in diagnostics['warnings']:
            console.print(f"  • {warning}")
    
    console.print()


def print_chain_failure_summary(
    posterior_samples: Dict[str, np.ndarray],
    diagnostics: Dict[str, Any],
) -> None:
    """
    Print per-chain mean summaries for non-converging parameters.

    Args:
        posterior_samples: Chain-structured posterior samples
        diagnostics: Diagnostics dict from diagnose_mcmc
    """
    warnings = diagnostics.get('warnings', [])
    if not warnings:
        return

    failed = set()
    for w in warnings:
        param_name = w.split(':')[0].strip()
        if 'R-hat' in w or 'ESS' in w:
            failed.add(param_name)

    if not failed:
        return

    console.print(
        "\n[bold cyan]═══ Chain Means for Non-Converging Parameters ═══[/bold cyan]\n"
    )

    # Sort by R-hat descending (worst first)
    def _sort_key(name: str) -> float:
        """Give the sort order for a parameter, with the largest R-hat first.

        Args:
            name: Parameter name.

        Returns:
            Minus the absolute R-hat of the parameter, or 0.0 if it has no
            R-hat recorded.
        """
        r = diagnostics.get('rhat', {}).get(name)
        return 0.0 if r is None else -abs(r)

    for param_name in sorted(failed, key=_sort_key):
        if param_name not in posterior_samples:
            continue

        samples = np.asarray(posterior_samples[param_name])
        n_chains = samples.shape[0]

        chain_mean = np.mean(samples, axis=1)
        chain_std = np.std(samples, axis=1)

        if chain_mean.ndim == 1:
            parts = []
            for c in range(n_chains):
                parts.append(f"{chain_mean[c]:.4f} ± {chain_std[c]:.4f}")
            console.print(f"  {param_name}:")
            console.print(f"    chain means (mean ± std) = [{', '.join(parts)}]")
        else:
            flat_mean = chain_mean.reshape(n_chains, -1)
            flat_std = chain_std.reshape(n_chains, -1)
            for c in range(n_chains):
                lo = flat_mean[c].min()
                hi = flat_mean[c].max()
                avg_std = flat_std[c].mean()
                console.print(
                    f"  {param_name} [chain {c}]:"
                    f"  mean range [{lo:.4f}, {hi:.4f}]  avg std = {avg_std:.4f}"
                )

    console.print()


def save_diagnostics(diagnostics: Dict, output_path: str):
    """
    Save diagnostics to a JSON file.
    
    Args:
        diagnostics: Dictionary from diagnose_mcmc
        output_path: Path to save JSON file
    """
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        """Convert numpy numbers and arrays to plain Python values for JSON.

        Works through dictionaries and lists. Other values are left as they
        are.

        Args:
            obj: Value to convert.

        Returns:
            The value with numpy numbers turned into floats and numpy arrays
            turned into lists.
        """
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj
    
    diagnostics_serializable = convert_types(diagnostics)
    
    with open(output_path, 'w') as f:
        json.dump(diagnostics_serializable, f, indent=2)


def per_chain_log_lik_comparison(
    idata: az.InferenceData,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """Compare per-chain log-likelihoods to detect mode differences.

    Handles both 3D (chain, draw, subjects) and 4D (chain, draw, subject, trial)
    log-likelihood arrays. It sums the log-likelihood over participants (and
    trials) for each draw, then takes the mean and standard deviation over
    draws within each chain. A chain whose mean is far from the others may be
    stuck in a different mode.

    Args:
        idata: InferenceData with log_likelihood group
        console: Optional rich console for printing

    Returns:
        Dictionary with these entries:
        - total_ll: array of shape (n_chains, n_draws), the total
          log-likelihood of each draw
        - chain_means: list with the mean total log-likelihood of each chain
        - chain_stds: list with the standard deviation of the total
          log-likelihood of each chain
        - subj_mean_ll: array of shape (n_chains, n_draws, n_subjects) with
          each participant's log-likelihood for each draw, summed over trials
    """
    ll = idata.log_likelihood["obs"].values
    n_chains = ll.shape[0]

    # Collapse all dims beyond chain, draw
    if ll.ndim == 4:
        # (chain, draw, subject, trial) -> sum over subject+trial
        total_ll = ll.sum(axis=(2, 3))
        subj_ll = ll.sum(axis=3)  # (chain, draw, subject)
    else:
        # (chain, draw, subjects) -> sum over subjects
        total_ll = ll.sum(axis=2)
        subj_ll = ll

    chain_means = total_ll.mean(axis=1)
    chain_stds = total_ll.std(axis=1)

    if console:
        console.print(
            "\n[bold cyan]═══ Per-Chain Log-Likelihood Comparison ═══[/bold cyan]\n"
        )
        for c in range(n_chains):
            console.print(
                f"  Chain {c}: mean LL = {chain_means[c]:.1f} ± {chain_stds[c]:.1f}"
            )

        best = chain_means.argmax()
        for c in range(n_chains):
            if c != best:
                diff = chain_means[best] - chain_means[c]
                console.print(
                    f"  Chain {best} - Chain {c}: ΔLL = {diff:.1f}"
                )

        console.print()

    return {
        "total_ll": np.asarray(total_ll),
        "chain_means": chain_means.tolist(),
        "chain_stds": chain_stds.tolist(),
        "subj_mean_ll": np.asarray(subj_ll),
    }


def per_chain_ppc(
    data: Dict[str, Any],
    idata: az.InferenceData,
    model_name: str,
    n_per_chain: int = 200,
    output_dir: Optional[Union[str, Path]] = None,
    console: Optional[Console] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run posterior predictive checks separately for each chain.

    For each chain, simulates choices from that chain's posterior draws and
    compares them with the observed choices. The random seed for each chain is
    the chain number.

    Args:
        data: Data dict with outcomes_0, outcomes_1, choices (numpy or JAX arrays)
        idata: InferenceData with posterior group containing subject-level params
        model_name: Model name (for extracting subject parameters)
        n_per_chain: Number of posterior draws per chain for PPC
        output_dir: If set, save per-chain PPC plots here
        console: Optional rich console for printing

    Returns:
        Dictionary mapping chain index to the results of
        ``posterior_predictive_check`` for that chain. Empty if the posterior
        has no subject-level parameters for the model.
    """
    from .plotting import posterior_predictive_check, plot_ppc_results
    import matplotlib.pyplot as plt

    n_chains = idata.posterior.sizes["chain"]

    subj_vars = [
        v
        for v in idata.posterior.data_vars
        if v.startswith(f"{model_name}_") and v.endswith("_subj")
    ]

    if not subj_vars:
        if console:
            console.print(
                "  [yellow]No subject-level parameters found for PPC[/yellow]"
            )
        return {}

    ppc_data = {
        "outcomes_0": np.asarray(data["outcomes_0"]),
        "outcomes_1": np.asarray(data["outcomes_1"]),
        "choices": np.asarray(data["choices"]),
    }

    results: Dict[int, Dict[str, np.ndarray]] = {}
    for chain_idx in range(n_chains):
        if console:
            console.print(f"  Chain {chain_idx}: running PPC ...")

        chain_flat = {
            var: np.asarray(idata.posterior[var].values[chain_idx])
            for var in subj_vars
        }

        results[chain_idx] = posterior_predictive_check(
            data=ppc_data,
            posterior_samples=chain_flat,
            model_name=model_name,
            n_posterior_samples=n_per_chain,
            seed=chain_idx,
            verbose=True,
        )

        if output_dir:
            fig = plot_ppc_results(
                results[chain_idx], condition_name=f"Chain {chain_idx}"
            )
            fig.savefig(
                Path(output_dir) / f"ppc_chain_{chain_idx}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            if console:
                console.print(
                    f"  [dim]  Saved ppc_chain_{chain_idx}.png[/dim]"
                )

    # Summary comparison table
    if console:
        console.print(
            "\n[bold cyan]═══ Per-Chain PPC Accuracy Comparison ═══[/bold cyan]\n"
        )
        table = Table(
            title="Chain Comparison",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Chain", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("N Draws", justify="right")

        for chain_idx in range(n_chains):
            acc = results[chain_idx]["accuracy"]
            n_used = results[chain_idx].get(
                "all_simulated_probs",
                np.zeros((n_per_chain,)),
            ).shape[0]
            table.add_row(str(chain_idx), f"{acc:.4f}", str(n_used))

        console.print(table)
        console.print()

    return results


def plot_chain_pair_grid(
    idata: az.InferenceData,
    model_name: str,
    var_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    console: Optional[Console] = None,
) -> Optional[Any]:
    """Scatter matrix of group-level parameters colored by chain.

    Reveals multimodality by showing per-chain posterior distributions
    in different colours.

    Args:
        idata: InferenceData with posterior group
        model_name: Model name prefix for parameter names
        var_names: Short parameter suffixes (default: decay_mu, tau_pos_mu,
                   tau_neg_mu, temperature_mu)
        output_dir: If set, save figure here
        console: Optional rich console for printing

    Returns:
        matplotlib Figure (caller should close it), or None if no params found
    """
    import matplotlib.pyplot as plt

    if var_names is None:
        var_names = [
            "decay_mu",
            "tau_pos_mu",
            "tau_neg_mu",
            "temperature_mu",
        ]

    full_names = [f"{model_name}_{v}" for v in var_names]
    existing = [n for n in full_names if n in idata.posterior.data_vars]

    if not existing:
        if console:
            console.print("[yellow]  No matching parameters found for pair plot[/yellow]")
        return None

    n_vars = len(existing)
    n_chains = idata.posterior.sizes["chain"]

    colors = plt.colormaps["Set1"](np.linspace(0, 1, max(n_chains, 3)))

    values = {}
    for name in existing:
        arr = np.asarray(idata.posterior[name].values)
        arr = arr.squeeze()
        if arr.ndim > 2:
            arr = arr.reshape(n_chains, -1)
        values[name] = arr

    short_names = [n.replace(f"{model_name}_", "") for n in existing]

    fig, axes = plt.subplots(
        n_vars, n_vars, figsize=(2.5 * n_vars, 2.5 * n_vars)
    )
    if n_vars == 1:
        axes = np.array([[axes]])

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            if i == j:
                for c in range(n_chains):
                    ax.hist(
                        values[existing[i]][c],
                        bins=40,
                        alpha=0.4,
                        color=colors[c],
                        label=f"Chain {c}" if j == 0 else "",
                    )
                ax.set_title(short_names[i], fontsize=9)
            elif i > j:
                for c in range(n_chains):
                    ax.scatter(
                        values[existing[j]][c],
                        values[existing[i]][c],
                        s=2,
                        alpha=0.3,
                        color=colors[c],
                    )
            else:
                ax.set_visible(False)

            if j == 0 and i > 0:
                ax.set_ylabel(short_names[i], fontsize=8)
            if i == n_vars - 1 and j > 0:
                ax.set_xlabel(short_names[j], fontsize=8)

    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Per-Chain Posterior (colored by chain)", fontsize=11, y=1.02)
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / "chain_pair_grid.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        if console:
            console.print(f"  [dim]Saved {path}[/dim]")

    return fig
