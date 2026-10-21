"""
Hierarchical Bayesian inference for concurrent model fitting and comparison.

This module implements a "super-model" approach where multiple learning models
are fit simultaneously to subject data, allowing for concurrent parameter estimation
and model selection (via model frequencies).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam
from typing import Dict, List, Optional, Union, Tuple, Any
import toml
from pathlib import Path
from functools import partial
from rich.console import Console

console = Console()

from .model_simulation import get_model_info, create_scan_fn, create_initial_state
from .data_generation import load_parameter_bounds
from .data_preparation import OUTCOME_CODING_KEY, normalize_to_reward_coded

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def transform_param(
    raw_val: jnp.ndarray, min_val: float, max_val: float
) -> jnp.ndarray:
    """
    Map an unbounded value into the range from min_val to max_val with a sigmoid.

    Args:
        raw_val: Unbounded value or array of values.
        min_val: Lower bound of the parameter.
        max_val: Upper bound of the parameter.

    Returns:
        The value mapped into the range from min_val to max_val. A raw value
        of 0 maps to the midpoint of the range.
    """
    return min_val + (max_val - min_val) * jax.nn.sigmoid(raw_val)


def _normalize_fit_data(data: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    """Prepare the data for fitting, with outcomes coded 1 = safe and 0 = explosion.

    Keeps only ``outcomes_0``, ``outcomes_1`` and ``choices`` and turns them
    into float arrays. If the data has an ``outcome_coding`` entry saying the
    outcomes are already reward-coded, they are left as they are. Otherwise
    the outcomes are assumed to be explosion-coded and are flipped.

    Args:
        data: Dictionary with ``outcomes_0``, ``outcomes_1`` and ``choices``,
            each of shape (n_participants, n_trials), and optionally an
            ``outcome_coding`` entry.

    Returns:
        Dictionary with float arrays ``outcomes_0``, ``outcomes_1`` and
        ``choices``, with reward-coded outcomes.
    """
    metadata = None
    if OUTCOME_CODING_KEY in data:
        metadata = {OUTCOME_CODING_KEY: str(data[OUTCOME_CODING_KEY])}

    model_data = {
        "outcomes_0": np.asarray(data["outcomes_0"], dtype=float),
        "outcomes_1": np.asarray(data["outcomes_1"], dtype=float),
        "choices": np.asarray(data["choices"], dtype=float),
    }
    return normalize_to_reward_coded(model_data, metadata=metadata)



def run_single_subject_scan(
    model_name: str,
    params: Dict[str, Any],
    outcomes_0: jnp.ndarray,
    outcomes_1: jnp.ndarray,
    choices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Run the model scan for a single subject and return choice probabilities.

    Steps through the trials of one participant, updating the model after each
    trial. Parameters whose names contain ``_init`` set the starting state and
    are not passed to the trial-by-trial update.

    Args:
        model_name: Name of the model
        params: Dictionary of parameters for this subject
        outcomes_0: Array of outcomes for option 0, one value per trial
        outcomes_1: Array of outcomes for option 1, one value per trial
        choices: Array of observed choices, one value per trial

    Returns:
        Array of shape (n_trials,) with the model's choice probability on each
        trial (probability of choosing option 0)
    """
    model_info = get_model_info(model_name)
    
    # Initial state (needs _init params)
    initial_state = create_initial_state(model_name, params)
    
    # Filter params for scan function (remove _init params)
    scan_params = {k: v for k, v in params.items() if "_init" not in k}
    
    # Create scan function
    scan_fn = create_scan_fn(model_name, scan_params, model_info)
    
    # Run scan
    trial_indices = jnp.arange(len(outcomes_0))
    xs = (trial_indices, outcomes_0, outcomes_1, choices)
    _, outputs = jax.lax.scan(scan_fn, initial_state, xs)
    
    # The first tracked output is the model's probability of choosing option 0.
    first_output_name = model_info.output_vars[0]
    return outputs[first_output_name]


def hierarchical_model(
    data: Dict[str, jnp.ndarray],
    model_names: List[str],
    config_path: Optional[Union[str, Path]] = None,
    use_bor_pxp: bool = False,
    fix_model_probs: bool = False,
    use_weighted_prior: bool = False,
):
    """
    NumPyro hierarchical model for concurrent model fitting.

    Each model has group-level means (``mu``) and spreads (``sigma``) for every
    parameter that is not fixed. Each participant's parameter is the group mean
    plus the spread times a participant offset, mapped into the parameter's
    bounds. Every participant's trials are scored under every model, and the
    model frequencies weight the models in the likelihood. The function records
    each participant's parameters (``<model>_<parameter>_subj``) and the
    probability that each model produced their data (``responsibilities``,
    shape (n_subjects, n_models)). With ``use_bor_pxp`` it also records
    ``prob_H1``, ``r_effective`` and ``pxp_sample``. It prints the prior
    settings of each parameter. It returns nothing.

    Args:
        data: Dictionary containing:
            - outcomes_0: (n_subjects, n_trials)
            - outcomes_1: (n_subjects, n_trials)
            - choices: (n_subjects, n_trials) - Binary choices where
                       1 means option 0 was chosen and 0 means option 1.
        model_names: List of model names to fit
        config_path: Path to model_bounds.toml
        use_bor_pxp: If True, uses the "Marginalized Global Hypothesis" approach to 
                     calculate Bayesian Omnibus Risk (BOR) and Protected Exceedance 
                     Probability (PXP) continuously (global mixture of H0
                     and H1).
        fix_model_probs: If True, fixes model probabilities to be uniform (1/K).
                         Used for estimating the Null Hypothesis evidence in the 
                         "Super Model" approach (super-model ELBO comparison).
        use_weighted_prior: If True (and use_bor_pxp=True), effectively mixes the
                            Group Prior with a generic Null Prior based on model probability.
                            This softer gating allows parameters to range freely if the model
                            is unlikely.

    Raises:
        ValueError: If ``fix_model_probs`` and ``use_bor_pxp`` are both True.
    """
    outcomes_0 = data["outcomes_0"]
    outcomes_1 = data["outcomes_1"]
    choices = data["choices"]
    
    # Scale choices to (0, 1) to avoid numerical issues with Beta likelihood
                                
    n_subjects, n_trials = choices.shape
    n_models = len(model_names)
    
    # 1. Model Frequencies (Dirichlet prior)
    if fix_model_probs:
        if use_bor_pxp:
            raise ValueError("Cannot fix model_probs when use_bor_pxp=True (BOR-PXP handles H0 internally)")
        model_probs = jnp.ones(n_models) / n_models
    else:
        # We use a Dirichlet prior on the probability of each model in the population
        # In the BOR/PXP framework, this corresponds to r_H1 (frequencies under H1)
        model_probs = numpyro.sample(
            "model_probs", dist.Dirichlet(jnp.ones(n_models))
        )

    # Log if we're using weighted prior
    if use_weighted_prior and use_bor_pxp:
        console.print("[bold yellow]Using Weighted Priors based on Model Probabilities.[/bold yellow]")
    
    # Store log-likelihoods for each subject under each model
    # Shape: (n_models, n_subjects)
    log_likelihoods = []

    # Load bounds
    all_model_bounds = {name: load_parameter_bounds(name, config_path) for name in model_names}
    
    # 3. Loop over models
    for k, model_name in enumerate(model_names):
        # Load parameter bounds
        bounds = all_model_bounds[model_name]
        
        # Dictionary to store parameters for this model for all subjects
        # keys: param_name, values: array of shape (n_subjects,)
        subject_params = {}

        if use_weighted_prior and use_bor_pxp:
            # Get the probability of this model (w) and non-model (1-w)
            # Note: model_probs is (n_models,). We take the k-th element.
            model_weight = model_probs[k]
            non_model_weight = 1.0 - model_weight
        
        # Define hierarchical parameters
        with numpyro.plate(f"model_{k}_params", 1): # Placeholder plate if needed
            pass
            
        for param_name, param_bounds in bounds.items():
            if "fixed" in param_bounds:
                val = jnp.full(n_subjects, param_bounds["fixed"])
                numpyro.deterministic(f"{model_name}_{param_name}_subj", val)
                subject_params[param_name] = val
                continue

            # Get prior hyperparameters (with defaults matching previous hardcoded values)
            prior_mu_mean = param_bounds.get("prior_mu_mean", 0.0)
            prior_mu_sigma = param_bounds.get("prior_mu_sigma", 0.2)
            prior_sigma_scale = param_bounds.get("prior_sigma_scale", 1.0)
            prior_offset_sigma = param_bounds.get("prior_offset_sigma", 0.5)

            # Print priors using rich
            console.print(f"[bold]{model_name} - {param_name}[/bold]")
            console.print(f"  prior_mu_mean: {prior_mu_mean}")
            console.print(f"  prior_mu_sigma: {prior_mu_sigma}")
            console.print(f"  prior_sigma_scale: {prior_sigma_scale}")
            console.print(f"  prior_offset_sigma: {prior_offset_sigma}")

            # Group-level priors (Model specific)
            mu = numpyro.sample(
                f"{model_name}_{param_name}_mu", dist.Normal(prior_mu_mean, prior_mu_sigma)
            )
            sigma = numpyro.sample(
                f"{model_name}_{param_name}_sigma", dist.HalfNormal(prior_sigma_scale)
            )
            
            if use_weighted_prior and use_bor_pxp:
                # Null Priors (Generic broad prior)
                # Centered at 0 (0.5 prob), broad spread
                null_mu = 0.0 
                null_sigma = 2.0 
                null_offset_sigma = 1.0
                
                # Construct Weighted Priors
                # We mix the moments of the distributions
                mu_mix = model_weight * mu + non_model_weight * null_mu
                sigma_mix = model_weight * sigma + non_model_weight * null_sigma
                offset_sigma_mix = model_weight * prior_offset_sigma + non_model_weight * null_offset_sigma
                
                # Subject-level deviations
                # We sample from a Normal with the mixed variance
                with numpyro.plate(f"subjects_{model_name}_{param_name}", n_subjects):
                    offset = numpyro.sample(
                        f"{model_name}_{param_name}_offset", dist.Normal(0.0, offset_sigma_mix)
                    )
                
                # Transform to constrained space
                val = transform_param(
                    mu_mix + sigma_mix * offset, param_bounds["min"], param_bounds["max"]
                )
            else:
                 # Standard Hierarchical Logic
                with numpyro.plate(f"subjects_{model_name}_{param_name}", n_subjects):
                    offset = numpyro.sample(
                        f"{model_name}_{param_name}_offset", dist.Normal(0.0, prior_offset_sigma)
                    )
                
                # Transform to constrained space
                val = transform_param(
                    mu + sigma * offset, param_bounds["min"], param_bounds["max"]
                )
            
            # Register as deterministic site for recovery
            numpyro.deterministic(f"{model_name}_{param_name}_subj", val)
            
            subject_params[param_name] = val

            
        # Run model for all subjects (vectorized)
        # Use partial to safely capture model_name
        run_scan_partial = partial(run_single_subject_scan, model_name)
            
        mean = jax.vmap(run_scan_partial)(
            subject_params, outcomes_0, outcomes_1, choices
        )
        
        # mean is the probability of choosing option 0, and the observed
        # choices use 1 = option 0 and 0 = option 1.
        mean = jnp.clip(mean, 1e-5, 1.0 - 1e-5)
        
        log_lik_trials = dist.Bernoulli(probs=mean).log_prob(choices)
        
        log_lik_subj = jnp.sum(log_lik_trials, axis=1)
        
        log_likelihoods.append(log_lik_subj)

    # 4. Mixture Likelihood
    # Stack log-likelihoods: (n_models, n_subjects)
    log_liks_stacked = jnp.stack(log_likelihoods, axis=0)
    
    if use_bor_pxp:
        # --- BOR / PXP Logic (Marginalized Global Hypothesis) ---
        
        # H1: Models have different frequencies (r ~ Dirichlet)
        # model_probs is r_H1
        log_r_H1 = jnp.log(model_probs)
        log_r_H1_expanded = log_r_H1[:, None]
        
        # Total Log-Likelihood of data under H1
        # Sum over subjects of logsumexp(log(r_H1) + LL_subject)
        log_prob_H1 = jnp.sum(
            jax.scipy.special.logsumexp(log_r_H1_expanded + log_liks_stacked, axis=0)
        )
        
        # H0: Models are equally frequent (r = 1/K)
        r_uniform = jnp.ones(n_models) / n_models
        log_r_H0 = jnp.log(r_uniform)
        log_r_H0_expanded = log_r_H0[:, None]
        
        # Total Log-Likelihood of data under H0
        log_prob_H0 = jnp.sum(
            jax.scipy.special.logsumexp(log_r_H0_expanded + log_liks_stacked, axis=0)
        )
        
        # Global Mixture (Marginalizing H)
        # Prior P(H1) = 0.5, P(H0) = 0.5
        log_prior_H1 = jnp.log(0.5)
        log_prior_H0 = jnp.log(0.5)
        
        # Final joint log-prob
        log_joint = jax.scipy.special.logsumexp(jnp.stack([
            log_prior_H1 + log_prob_H1,
            log_prior_H0 + log_prob_H0
        ]))
        
        numpyro.factor("obs_global", log_joint)
        
        # --- Deterministic Recovery ---
        
        # Posterior probability of H1 (Models differ)
        # P(H1|Data) = Softmax([Joint_H1, Joint_H0])[0]
        log_posteriors = jnp.stack([
            log_prior_H1 + log_prob_H1,
            log_prior_H0 + log_prob_H0
        ])
        prob_H1 = numpyro.deterministic("prob_H1", jax.nn.softmax(log_posteriors)[0])
        
        # Effective r (Model Frequency)
        # r_eff = P(H1|D)*r_H1 + P(H0|D)*r_H0
        r_effective = numpyro.deterministic(
            "r_effective", 
            prob_H1 * model_probs + (1 - prob_H1) * r_uniform
        )
        
        # Subject-level responsibilities
        # We use the effective frequencies to calculate the posterior probability of each model per subject
        # This is an approximation that blends the H1 and H0 assumptions
        log_r_eff = jnp.log(r_effective)[:, None]
        weighted_log_liks = log_r_eff + log_liks_stacked
        mixture_log_lik = jax.scipy.special.logsumexp(weighted_log_liks, axis=0)
        responsibilities = jnp.exp(weighted_log_liks - mixture_log_lik)
        
        numpyro.deterministic("responsibilities", responsibilities.T)
        
        # 1. Identify the "Standard XP" Winner for this sample
        # We must use 'model_probs' (the raw Dirichlet r_H1), NOT 'r_effective'.
        # r_effective squashes values but preserves rank, so checking it for wins 
        # ignores the Null risk. Checking model_probs preserves the win/loss logic 
        # conditional on H1 being true.
        
        # Create a one-hot vector for the winner (e.g., [0, 1, 0] if Model 2 wins)
        winner_idx = jnp.argmax(model_probs)
        is_winner = jax.nn.one_hot(winner_idx, n_models)
        
        # 2. Apply the Rigoux Formula (Sample-wise)
        # Formula: PXP = XP * Confidence + Chance * Risk
        # Here:
        #   XP (for this sample) = is_winner (1 or 0)
        #   Confidence = prob_H1
        #   Risk = (1 - prob_H1)
        #   Chance = 1/K
        
        chance_level = 1.0 / n_models
        
        # This variable tracks the probability mass contributed to the PXP by this specific sample
        pxp_sample = (is_winner * prob_H1) + (chance_level * (1 - prob_H1))
        
        numpyro.deterministic("pxp_sample", pxp_sample)

    else:
        # --- Standard Logic ---
        
        # Compute marginal log-likelihood for each subject:
        # log p(y|alpha) = log sum_k p(m=k|alpha) p(y|theta_k, m=k)
        #                = logsumexp(log_alpha_k + log_lik_k)
        
        log_model_probs = jnp.log(model_probs) # (n_models,)
        
        # Broadcast log_model_probs to (n_models, n_subjects)
        log_model_probs_expanded = log_model_probs[:, None]
        
        # Compute mixture log-likelihood per subject
        mixture_log_lik = jax.scipy.special.logsumexp(
            log_model_probs_expanded + log_liks_stacked, axis=0
        )
        
        # Add to potential energy
        with numpyro.plate("obs_mixture_plate", n_subjects):
            numpyro.factor("obs_mixture", mixture_log_lik)
        
        # Calculate subject-level model responsibilities (posterior probability of each model for each subject)
        # p(m=k|y) = p(y|m=k) * p(m=k) / p(y)
        # log p(m=k|y) = log_lik_k + log_model_prob_k - mixture_log_lik
        weighted_log_liks = log_model_probs_expanded + log_liks_stacked
        
        # Normalize via softmax to get actual probabilities (0 to 1)
        # Shape: (n_models, n_subjects)
        # We subtract mixture_log_lik (the log normalizer) to get log probabilities, then exp
        responsibilities = jnp.exp(weighted_log_liks - mixture_log_lik)
        
        # Save to trace for analysis
        # Transpose to (n_subjects, n_models) for easier reading
        numpyro.deterministic("responsibilities", responsibilities.T)


def hierarchical_model_single(
    data: Dict[str, jnp.ndarray],
    model_name: str,
    config_path: Optional[Union[str, Path]] = None,
    include_log_lik: bool = True,
):
    """
    NumPyro hierarchical model for single model fitting (no model comparison).

    Each parameter that is not fixed has a group-level mean (``mu``) and
    spread (``sigma``). Each participant's parameter is the group mean plus the
    spread times a participant offset, mapped into the parameter's bounds. The
    function records each participant's parameters as
    ``<model_name>_<parameter>_subj``. It scores the observed choices with the
    model's choice probabilities. It returns nothing.

    Args:
        data: Dictionary with ``outcomes_0``, ``outcomes_1`` and ``choices``,
            each of shape (n_subjects, n_trials). Choices are 1 for option 0
            and 0 for option 1.
        model_name: Name of the model to fit.
        config_path: Path to the parameter bounds file. If None, uses the
            default file.
        include_log_lik: If True, also record the log-likelihood of each trial
            as ``log_lik``, with shape (n_subjects, n_trials). It is needed for
            WAIC and LOO.
    """
    outcomes_0 = data["outcomes_0"]
    outcomes_1 = data["outcomes_1"]
    choices = data["choices"]
    
    n_subjects, n_trials = choices.shape
    
    # Load bounds
    bounds = load_parameter_bounds(model_name, config_path)
    
    # Dictionary to store parameters for this model for all subjects
    # keys: param_name, values: array of shape (n_subjects,)
    subject_params = {}
    
    for param_name, param_bounds in bounds.items():
        if "fixed" in param_bounds:
            val = jnp.full(n_subjects, param_bounds["fixed"])
            numpyro.deterministic(f"{model_name}_{param_name}_subj", val)
            subject_params[param_name] = val
            continue

        # Get prior hyperparameters (with defaults matching previous hardcoded values)
        prior_mu_mean = param_bounds.get("prior_mu_mean", 0.0)
        prior_mu_sigma = param_bounds.get("prior_mu_sigma", 0.2)
        prior_sigma_scale = param_bounds.get("prior_sigma_scale", 1.0)
        prior_offset_sigma = param_bounds.get("prior_offset_sigma", 0.5)

        # Group-level priors
        mu = numpyro.sample(
            f"{model_name}_{param_name}_mu", dist.Normal(prior_mu_mean, prior_mu_sigma)
        )
        sigma = numpyro.sample(
            f"{model_name}_{param_name}_sigma", dist.HalfNormal(prior_sigma_scale)
        )
        
        # Subject-level deviations
        with numpyro.plate(f"subjects_{model_name}_{param_name}", n_subjects):
            offset = numpyro.sample(
                f"{model_name}_{param_name}_offset", dist.Normal(0.0, prior_offset_sigma)
            )
        
        # Transform
        val = transform_param(
            mu + sigma * offset, param_bounds["min"], param_bounds["max"]
        )
        numpyro.deterministic(f"{model_name}_{param_name}_subj", val)
        
        subject_params[param_name] = val
        
    # Run model
    run_scan_partial = partial(run_single_subject_scan, model_name)
    mean = jax.vmap(run_scan_partial)(
        subject_params, outcomes_0, outcomes_1, choices
    )

    # # Check for out of bounds values
    # min_val = jnp.min(mean)
    # max_val = jnp.max(mean)
    
    # def warn_out_of_bounds(vals):
    #     mn, mx = vals
    #     jax.debug.print("WARNING: Model output out of bounds [0, 1]. Min: {}, Max: {}", mn, mx)

    # jax.lax.cond(
    #     (min_val <= 0.0) | (max_val >= 1.0),
    #     warn_out_of_bounds,
    #     lambda x: None,
    #     (min_val, max_val)
    # )

    # Clip mean to ensure numerical stability (avoid 0 or 1)
    mean = jnp.clip(mean, 1e-5, 1.0 - 1e-5)
    
    # mean is the probability of choosing option 0, while observations use
    # 1 = option 0 and 0 = option 1.
    if include_log_lik:
        # Record trial-wise log likelihood only for workflows that need WAIC/LOO.
        log_lik = dist.Bernoulli(probs=mean).log_prob(choices)
        numpyro.deterministic("log_lik", log_lik)

    with numpyro.plate("subjects", n_subjects):
        numpyro.sample("obs", dist.Bernoulli(probs=mean).to_event(1), obs=choices)

def fit_mcmc(
    data: Dict[str, jnp.ndarray],
    model_names: Union[List[str], str],
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 1,
    target_accept_prob: float = 0.95,
    max_tree_depth: int = 10,
    seed: int = 0,
    use_bor_pxp: bool = False,
    chain_method: Optional[str] = None,
    use_weighted_prior: bool = False, # Pass through
    config_path: Optional[Union[str, Path]] = None,
    include_log_lik: bool = True,
    **kwargs
):
    """
    Run MCMC inference.

    Draws posterior samples with the NUTS sampler. A single model name fits
    that model on its own (``hierarchical_model_single``). A list of names fits
    all the models together and estimates how common each one is
    (``hierarchical_model``). The data are first converted to reward coding
    (see ``_normalize_fit_data``). If ``chain_method`` is None, chains run in
    parallel when there is a device for each chain, and vectorised on one
    device otherwise. The progress bar is on unless ``progress_bar`` is passed.
    The function prints a summary of the run.

    Args:
        data: Dictionary with outcomes_0, outcomes_1, choices arrays
        model_names: Model name(s) to fit
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        target_accept_prob: Target acceptance probability for NUTS
        max_tree_depth: Maximum tree depth for NUTS
        seed: Random seed
        use_bor_pxp: Use Bayesian Omnibus Risk framework. Used only when
            fitting several models.
        chain_method: 'parallel' or 'vectorized'
        use_weighted_prior: Use weighted priors. Used only when fitting several
            models.
        config_path: Path to model_bounds.toml config file (optional)
        include_log_lik: Whether to record trial-wise log likelihood sites for
            single-model fitting
        **kwargs: Additional arguments passed to MCMC (e.g. thinning)

    Returns:
        The NumPyro MCMC object after sampling. Its ``get_samples`` method
        gives the posterior samples.
    """
    data = _normalize_fit_data(data)
    rng_key = jax.random.PRNGKey(seed)

    # Auto-detect chain method if not provided
    n_devices = jax.local_device_count()
    print(f"Detected {n_devices} device(s) for MCMC chains.")
    if chain_method is None:
        if num_chains > 1 and n_devices < num_chains:
            # If we have more chains than devices (e.g. 4 chains on 1 CPU),
            # 'parallel' would run them sequentially in batches. 
            # 'vectorized' runs them simultaneously on the single device.
            chain_method = "vectorized"
        else:
            chain_method = "parallel"

    # Print summary of the run
    print(f"\n=== Starting MCMC Inference ===")
    if isinstance(model_names, str):
        print(f"Model: {model_names}")
    else:
        print(f"Models: {', '.join(model_names)}")
        print(f"Hierarchical Mode: {'BOR/PXP' if use_bor_pxp else 'Standard'}")
    
    n_subjects, n_trials = data['choices'].shape
    print(f"Data: {n_subjects} subjects, {n_trials} trials")
    print(f"MCMC Config: {num_warmup} warmup, {num_samples} samples, {num_chains} chains")
    print(f"Chain Method: {chain_method} (available devices: {n_devices})")
    print(f"NUTS Config: target_accept={target_accept_prob}, max_tree_depth={max_tree_depth}")
    print(f"Precision: {'float64' if jax.config.jax_enable_x64 else 'float32'}")
    print(f"==============================\n")
    
    # Ensure progress bar is enabled by default unless explicitly disabled in kwargs
    if "progress_bar" not in kwargs:
        kwargs["progress_bar"] = True

    if isinstance(model_names, str):
        print(f"Initializing NUTS kernel for single model...")
        kernel = NUTS(hierarchical_model_single, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            **kwargs
        )
        print(f"Running MCMC (this may take a moment to compile)...")
        mcmc.run(
            rng_key,
            data=data,
            model_name=model_names,
            config_path=config_path,
            include_log_lik=include_log_lik,
        )
    else:
        print(f"Initializing NUTS kernel for hierarchical model comparison...")
        kernel = NUTS(hierarchical_model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            **kwargs
        )
        print(f"Running MCMC (this may take a moment to compile)...")
        # Add JIT compilation check/warmup if helpful? 
        # Actually, numpyro JITs automatically. 
        # But we can try to hint the user about what's happening.
        mcmc.run(rng_key, data=data, model_names=model_names, use_bor_pxp=use_bor_pxp, use_weighted_prior=use_weighted_prior, config_path=config_path)
        
    return mcmc
