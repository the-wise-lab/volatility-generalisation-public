"""
Plotting functions for model comparison and recovery analysis.

This module draws figures of simulated choices, parameter recovery and
posterior predictive checks, and runs the posterior predictive check itself.
Choices are coded 1 for option 0 and 0 for option 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.stats import gaussian_kde
from typing import Dict, Union, Optional

from .model_simulation import run_model_scan
from .posterior_utils import posterior_interval

def posterior_predictive_check(
    data: Dict[str, np.ndarray],
    posterior_samples: Dict[str, np.ndarray],
    model_name: str = 'rescorla_wagner_single_value',
    n_posterior_samples: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Perform posterior predictive checks by sampling from the full posterior.

    For each of n_posterior_samples parameter sets drawn from the posterior,
    the model is run for all subjects to generate predicted choice probabilities.
    The resulting distribution characterises the posterior predictive distribution (PPD).
    The samples are drawn without replacement, and at most all available samples
    are used. Predicted choices are 1 when the mean predicted probability of
    option 0 is at least 0.5.

    Args:
        data: Prepared data dictionary with outcomes_0, outcomes_1, choices.
            The arrays can have shape (n_subjects, n_trials) or be flattened, in
            which case they are reshaped using the number of participants in
            ``choices``.
        posterior_samples: Dictionary of posterior samples from MCMC. It must
            contain participant-level sites named ``<model_name>_<param>_subj``.
        model_name: Name of the model
        n_posterior_samples: Number of posterior samples to draw for the PPD
        seed: Random seed for selecting posterior samples and for sampling
            choices
        verbose: If True, print progress

    Returns:
        Dictionary with:
            all_simulated_probs: (n_post_samples, n_subjects, n_trials) - full PPD
            all_simulated_choices: (n_post_samples, n_subjects, n_trials) - one
                sampled choice per predicted probability
            simulated_probs: (n_subjects, n_trials) - posterior mean
            simulated_choices: (n_subjects, n_trials) - binary from posterior mean
            actual_choices: (n_subjects, n_trials) - observed choices
            accuracy: scalar overall prediction accuracy
            trial_accuracy: (n_trials,) per-trial accuracy

    Raises:
        ValueError: If ``posterior_samples`` has no participant-level sites for
            this model.
    """
    prefix = f"{model_name}_"
    suffix = "_subj"

    # Extract subject-level parameter arrays and flatten any chain dimension so
    # downstream sampling always sees (n_posterior_samples, n_subjects).
    subject_params_all = {}
    for key, val in posterior_samples.items():
        if key.startswith(prefix) and key.endswith(suffix):
            param_name = key[len(prefix):-len(suffix)]
            values = np.asarray(val)
            if values.ndim == 1:
                values = values[:, None]
            elif values.ndim > 2:
                values = values.reshape(-1, values.shape[-1])
            subject_params_all[param_name] = values

    if not subject_params_all:
        raise ValueError(
            f"No subject-level parameters found for model '{model_name}'. "
            f"Expected keys like '{model_name}_<param>_subj' in posterior_samples."
        )

    # Parse data arrays
    outcomes_0 = np.array(data['outcomes_0'])
    outcomes_1 = np.array(data['outcomes_1'])
    choices = np.array(data['choices'])

    if outcomes_0.ndim == 1:
        n_subj = choices.shape[0]
        n_tr = len(outcomes_0) // n_subj
        outcomes_0 = outcomes_0.reshape(n_subj, n_tr)
        outcomes_1 = outcomes_1.reshape(n_subj, n_tr)
        choices = choices.reshape(n_subj, n_tr)

    n_subjects, n_trials = choices.shape

    # Subsample from the MCMC posterior
    n_total = next(iter(subject_params_all.values())).shape[0]
    rng = np.random.default_rng(seed)
    n_used = min(n_posterior_samples, n_total)
    sample_indices = rng.choice(n_total, size=n_used, replace=False)

    if verbose:
        print(f"  Running PPD with {n_used} posterior samples × {n_subjects} subjects ...")

    # Run model for each posterior sample × subject
    all_simulated_probs = np.zeros((n_used, n_subjects, n_trials))

    report_every = max(1, n_used // 10)
    for s_idx, post_idx in enumerate(sample_indices):
        for subj_idx in range(n_subjects):
            params = {
                name: float(arr[post_idx, subj_idx])
                for name, arr in subject_params_all.items()
            }
            trial_data = {
                'outcomes_0': jnp.array(outcomes_0[subj_idx]),
                'outcomes_1': jnp.array(outcomes_1[subj_idx]),
                'choices': jnp.array(choices[subj_idx]),
            }
            _, outputs = run_model_scan(
                model_name=model_name,
                model_params=params,
                trial_data=trial_data,
            )
            all_simulated_probs[s_idx, subj_idx] = np.array(outputs['prev_pos'])

        if verbose and (s_idx + 1) % report_every == 0:
            print(f"    {s_idx + 1}/{n_used} samples done")

    # Sample binary choices using the convention 1 = option 0, 0 = option 1.
    # Shape: (n_used, n_subjects, n_trials) — one Bernoulli draw per posterior sample
    all_simulated_choices = rng.binomial(1, all_simulated_probs).astype(float)

    # Posterior mean prediction (from probabilities, for accuracy metric)
    mean_probs = all_simulated_probs.mean(axis=0)   # (n_subjects, n_trials)
    simulated_choices = (mean_probs >= 0.5).astype(int)
    accuracy = float((simulated_choices == choices).mean())
    trial_accuracy = (simulated_choices == choices).mean(axis=0)

    return {
        'all_simulated_probs': all_simulated_probs,     # (n_post_samples, n_subjects, n_trials)
        'all_simulated_choices': all_simulated_choices, # (n_post_samples, n_subjects, n_trials)
        'simulated_probs': mean_probs,                  # (n_subjects, n_trials) - backward compat
        'simulated_choices': simulated_choices,         # (n_subjects, n_trials)
        'actual_choices': choices,                      # (n_subjects, n_trials)
        'accuracy': accuracy,
        'trial_accuracy': trial_accuracy,
    }


def _setup_axis(ax, title: str, xlabel: Optional[str] = None, ylabel: Optional[str] = None, 
                xlim: Optional[tuple] = None, ylim: Optional[tuple] = None):
    """Set the title, axis labels and axis limits of a plot panel.

    Labels and limits are only set if they are given.

    Args:
        ax: The matplotlib axes to change.
        title: Title of the panel.
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
        xlim: Limits of the x axis, as (lower, upper).
        ylim: Limits of the y axis, as (lower, upper).
    """
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def plot_ppc_results(results: Dict[str, np.ndarray], condition_name: str = '') -> plt.Figure:
    """
    Plot posterior predictive check results with credible interval bands.

    Four panels:
      1. Mean choice probability over trials with 95% credible interval vs. observed
      2. Subject-level scatter: observed vs. predicted mean choice rate (95% CI error bars)
      3. Distribution of subject mean choice rates: observed KDE vs. PPD
      4. Trial-by-trial prediction accuracy

    The intervals are the 2.5th to 97.5th percentiles across posterior samples.
    If ``results`` has no ``all_simulated_probs``, the panels show only the mean
    prediction, with no intervals. Panel 3 falls back to histograms if the
    density estimate fails. It shows a strip of points instead when there are
    three or fewer participants or no ``all_simulated_probs``.

    Args:
        results: Dictionary from posterior_predictive_check
        condition_name: Name of condition for plot title

    Returns:
        matplotlib Figure object (caller is responsible for saving/closing)
    """
    actual_choices = np.asarray(results['actual_choices'])
    trial_accuracy = np.asarray(results['trial_accuracy'])
    accuracy = float(results['accuracy'])
    n_subjects, n_trials = actual_choices.shape
    trials = np.arange(n_trials)

    # Determine whether we have full PPD samples for uncertainty bands
    has_ppd = 'all_simulated_probs' in results and np.asarray(results['all_simulated_probs']).ndim == 3

    if has_ppd:
        all_probs = np.asarray(results['all_simulated_probs'])     # (n_samples, n_subjects, n_trials)
        # Use sampled binary choices if available, else fall back to probabilities
        if 'all_simulated_choices' in results:
            all_choices_ppd = np.asarray(results['all_simulated_choices'])
        else:
            all_choices_ppd = (all_probs > 0.5).astype(float)
        n_samples = all_probs.shape[0]

        # Trial-level: mean over subjects for each posterior sample → (n_samples, n_trials)
        trial_mean_per_sample = all_probs.mean(axis=1)
        trial_mean_pred = trial_mean_per_sample.mean(axis=0)
        trial_lower, trial_upper = posterior_interval(trial_mean_per_sample, axis=0)

        # Subject-level: mean over trials for each posterior sample → (n_samples, n_subjects)
        # Use probabilities for CI on the scatter plot, sampled choices for the distribution panel
        subj_mean_per_sample = all_probs.mean(axis=2)
        subj_mean_pred = subj_mean_per_sample.mean(axis=0)
        subj_lower, subj_upper = posterior_interval(subj_mean_per_sample, axis=0)

        # For the distribution panel: mean choice rate from sampled binary choices
        subj_choice_rate_per_sample = all_choices_ppd.mean(axis=2)  # (n_samples, n_subjects)
    else:
        # Fallback: no posterior uncertainty available
        mean_probs = np.asarray(results['simulated_probs'])
        n_samples = 1
        trial_mean_pred = mean_probs.mean(axis=0)
        trial_lower = trial_upper = trial_mean_pred
        subj_mean_pred = mean_probs.mean(axis=1)
        subj_lower = subj_upper = subj_mean_pred

    trial_mean_obs = actual_choices.mean(axis=0)   # (n_trials,)
    subj_mean_obs = actual_choices.mean(axis=1)    # (n_subjects,)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax0, ax1, ax2, ax3 = axes.flatten()

    # --- Panel 1: Trial-by-trial mean choice probability with 95% CI ---
    ax0.fill_between(trials, trial_lower, trial_upper,
                     alpha=0.25, color='steelblue', label='95% CI')
    ax0.plot(trials, trial_mean_pred, color='steelblue', linewidth=2,
             linestyle='--', label='Predicted (mean)')
    ax0.plot(trials, trial_mean_obs, color='black', linewidth=1.5, label='Observed')
    ax0.axhline(0.5, color='red', linestyle=':', alpha=0.4, label='Chance')
    _setup_axis(ax0, f'{condition_name} Mean Choice Probability Over Trials',
                xlabel='Trial', ylabel='P(Choose Option 0)', ylim=(0, 1))
    ax0.legend(fontsize=9)

    # --- Panel 2: Subject-level scatter with 95% CI error bars ---
    err_lower = subj_mean_pred - subj_lower
    err_upper = subj_upper - subj_mean_pred
    ax1.errorbar(subj_mean_obs, subj_mean_pred,
                 yerr=[err_lower, err_upper],
                 fmt='o', color='steelblue', alpha=0.7,
                 ecolor='steelblue', elinewidth=1.5, capsize=4,
                 label='Subjects (95% CI)')
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Identity')
    _setup_axis(ax1, f'{condition_name} Subject-Level Comparison',
                xlabel='Observed Mean P(Choose Option 0)',
                ylabel='Predicted Mean P(Choose Option 0)',
                xlim=(0, 1), ylim=(0, 1))
    ax1.legend(fontsize=9)

    # --- Panel 3: Distribution of subject mean choice rates (observed vs. PPD) ---
    # Uses sampled binary choices (not raw probabilities) for a fair like-for-like comparison
    xs = np.linspace(0, 1, 300)
    if has_ppd and n_subjects > 3:
        ppd_flat = subj_choice_rate_per_sample.ravel()  # (n_samples * n_subjects,)
        try:
            obs_kde = gaussian_kde(subj_mean_obs)
            ppd_kde = gaussian_kde(ppd_flat)
            ax2.fill_between(xs, ppd_kde(xs), alpha=0.25, color='steelblue')
            ax2.plot(xs, ppd_kde(xs), color='steelblue', linewidth=2,
                     linestyle='--', label=f'PPD choices (n={n_samples})')
            ax2.plot(xs, obs_kde(xs), color='black', linewidth=2.5, label='Observed')
        except Exception:
            ax2.hist(ppd_flat, bins=30, alpha=0.4, color='steelblue',
                     density=True, label=f'PPD choices (n={n_samples})')
            ax2.hist(subj_mean_obs, bins=10, alpha=0.7, color='black',
                     density=True, label='Observed')
    else:
        # Small n: dot-strip fallback
        ax2.scatter(subj_mean_obs, np.zeros(n_subjects),
                    color='black', s=60, zorder=5, label='Observed')
        if has_ppd:
            ax2.scatter(subj_choice_rate_per_sample.ravel(),
                        np.random.default_rng(0).uniform(
                            -0.02, 0.02, n_samples * n_subjects),
                        color='steelblue', s=8, alpha=0.15, label='PPD choices')
    _setup_axis(ax2, f'{condition_name} Subject Mean Choice Rate Distribution',
                xlabel='Mean P(Choose Option 0) per Subject', ylabel='Density')
    ax2.legend(fontsize=9)

    # --- Panel 4: Trial-by-trial accuracy ---
    ax3.plot(trials, trial_accuracy, color='steelblue', linewidth=1.5, alpha=0.8)
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax3.axhline(accuracy, color='darkorange', linewidth=2,
                label=f'Mean = {accuracy:.3f}')
    _setup_axis(ax3, f'{condition_name} Trial-by-Trial Accuracy',
                xlabel='Trial', ylabel='Prediction Accuracy', ylim=(0, 1))
    ax3.legend(fontsize=9)

    plt.tight_layout()
    return fig
