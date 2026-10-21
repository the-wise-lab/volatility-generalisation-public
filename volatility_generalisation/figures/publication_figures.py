"""
Functions that load, summarise and plot the results for the publication figures.

Holds the condition names, labels and colours, the code that reads the estimates
and posterior samples, the summaries of learning rates and marginal effects, the
task-performance statistics, and the plotting code for the result figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from ..modelling_code.condition_metadata import (
    BLOCK_TRIAL_COUNT,
    CONDITION_METADATA,
    REVERSAL_INTERVAL_TRIALS,
    better_option_by_trial,
    build_context_regressors,
)
from ..modelling_code.posterior_utils import posterior_interval

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Agreed condition order for all four figures (transition order).
CONDITION_ORDER: Tuple[str, ...] = (
    "StableV4_2",   # Stable → Stable
    "StableV1",     # Volatile → Stable
    "VolatileV2",   # Stable → Volatile
    "VolatileV3_2",  # Volatile → Volatile
)

CONDITION_LABELS: Dict[str, str] = {
    "StableV4_2": "Stable → Stable",
    "StableV1": "Volatile → Stable",
    "VolatileV2": "Stable → Volatile",
    "VolatileV3_2": "Volatile → Volatile",
}

SHORT_CONDITION_LABELS: Dict[str, str] = {
    "StableV4_2": "Stab\n> Stab",
    "StableV1": "Vol\n> Stab",
    "VolatileV2": "Stab\n> Vol",
    "VolatileV3_2": "Vol\n> Vol",
}

# Jewel-tone palette for the four conditions. The order intentionally follows
# the selected palette options 1, 4, 2, 3.
CONDITION_COLOURS: Tuple[str, ...] = (
    "#4169E1",  # Stable → Stable  (royal blue)
    "#C23B75",  # Volatile → Stable (raspberry)
    "#E69F00",  # Stable → Volatile (amber)
    "#00A878",  # Volatile → Volatile (emerald)
)
ACCENT_BLUE = CONDITION_COLOURS[0]
SUMMARY_COLOUR = "#262626"

# Name of the main model, whose posterior Figures 2 and 3 use. It is the model
# chosen for the main analysis, not necessarily the top-ranked model in the
# model comparison.
WINNING_MODEL = "dual_lr_single_value"

# Output formats.
OUTPUT_FORMATS: Tuple[str, ...] = (".pdf", ".svg", ".png")

LEARNING_RATE_PARAMS: Tuple[str, ...] = ("learning_rate_pos", "learning_rate_neg")

# Explicit retained-model order for Figure 2.
FIGURE2_MODEL_ORDER: Tuple[str, ...] = (
    "rescorla_wagner_single_value",
    "dual_lr_single_value",
    "pearce_hall_single_value",
    "leaky_beta_single_value",
    "leaky_beta_asymmetric_single_value",
)

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "rescorla_wagner_single_value": "Rescorla-\nWagner",
    "dual_lr_single_value": "Dual learning-\nrate",
    "pearce_hall_single_value": "Pearce-\nHall",
    "leaky_beta_single_value": "Leaky\nbeta",
    "leaky_beta_asymmetric_single_value": "Asymmetric\nleaky beta",
}

FIGURE_SIZES_INCHES: Dict[str, Tuple[float, float]] = {
    "figure1": (3, 3),
    "figure2": (6, 6.6),
    "figure3": (6, 6),
}

LABEL_FONT_SIZE = 8
TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 9
PANEL_LABEL_FONT_SIZE = 12
PANEL_LABEL_FONT_WEIGHT = 900

DM_SANS_FONT_NAME = "DM Sans"
# The DM Sans file shipped with the package (see fonts/README.md for where it
# came from), and its SHA-256, so a changed file is caught by the tests.
DM_SANS_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "dm-sans.ttf"
DM_SANS_FONT_SHA256 = "b0ae3e89a7d3ef2b1838bb4b1ceb0b17cc3435af061eee809d76aa623ff116e9"

# Required group-level parameter suffixes. Subject scale is optional because
# fixed_subject_sigma removes its sampled site.
REQUIRED_GROUP_SITES: Tuple[str, ...] = (
    "_mu",
    "_beta_current_vol",
    "_beta_previous_vol",
    "_beta_current_previous_interaction",
)
OPTIONAL_GROUP_SITES: Tuple[str, ...] = ("_subject_sigma",)
GROUP_SITES: Tuple[str, ...] = REQUIRED_GROUP_SITES + OPTIONAL_GROUP_SITES


def ensure_dm_sans_font(font_path: Optional[Path] = None) -> None:
    """Set up the DM Sans font for all figures, using the file in the package.

    The function registers the font file with matplotlib and sets the default
    font family and the font sizes for text, titles, axis labels and tick
    labels. It never uses the internet and never falls back to another font.

    Args:
        font_path: Font file to register. If ``None``, uses the file shipped
            with the package (``DM_SANS_FONT_PATH``). Tests use this to check
            what happens when the file is missing.

    Raises:
        RuntimeError: If the font file is missing, or cannot be registered or
            found.
    """
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    path = Path(font_path) if font_path is not None else DM_SANS_FONT_PATH
    try:
        if not path.is_file():
            raise FileNotFoundError(f"font file not found: {path}")
        font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = [DM_SANS_FONT_NAME]
        plt.rcParams["font.size"] = LABEL_FONT_SIZE
        plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
        plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
        plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
        plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE
        font_manager.findfont(DM_SANS_FONT_NAME, fallback_to_default=False)
    except Exception as exc:
        raise RuntimeError(f"DM Sans font setup failed: {exc}") from exc

# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------

def condition_order() -> List[str]:
    """List the four second-block conditions in the order used in the figures.

    Returns:
        Condition names in transition order: Stable to Stable, Volatile to
        Stable, Stable to Volatile, Volatile to Volatile.
    """
    return list(CONDITION_ORDER)


def condition_labels() -> Dict[str, str]:
    """Give the full display label for each condition.

    Returns:
        Dictionary that maps each condition name to its label, for example
        ``"Stable → Volatile"``.
    """
    return dict(CONDITION_LABELS)


def _despine(ax) -> None:
    """Hide the top and right border lines of a plot.

    Args:
        ax: Matplotlib axes to change.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_labels(fig, axes=None, labels: Optional[Sequence[str]] = None) -> None:
    """Add bold lowercase panel labels (a, b, c, ...) to the visible plots.

    Each label sits just above and to the left of its plot. Hidden plots get no
    label, but still use up their letter. It can also be called as
    ``add_panel_labels(axes)`` or ``add_panel_labels(axes, ["a", "b"])``, with
    the axes in place of the figure.

    Args:
        fig: Matplotlib figure that owns the axes. It is not used, except that
            it is read as the axes in the short call described above.
        axes: A list or array of axes objects (any shape), in the order the
            labels are assigned.
        labels: Text for each label, in the same order as the axes. If ``None``,
            the labels are ``"a"``, ``"b"``, ``"c"`` and so on.
    """
    if axes is None or (
        labels is None
        and isinstance(axes, (list, tuple))
        and all(isinstance(label, str) for label in axes)
    ):
        if axes is not None and labels is None and isinstance(axes, (list, tuple)):
            labels = axes
        axes = fig
    del fig  # Kept in the signature so callers can pass the owning figure.
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    labels = labels or tuple(chr(ord("a") + i) for i in range(len(axes_array)))
    for ax, label in zip(axes_array, labels):
        if ax.get_visible():
            ax.text(
                -0.14, 1.04, label, transform=ax.transAxes,
                fontsize=PANEL_LABEL_FONT_SIZE, fontweight=PANEL_LABEL_FONT_WEIGHT,
                va="bottom", ha="right",
            )


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_single_lr_subject_estimates(path: str) -> pd.DataFrame:
    """Load and check the single-learning-rate estimates for each participant.

    If the file has a ``model`` column, only the rows for the
    ``rescorla_wagner_single_value`` model are kept.

    Args:
        path: Path to the CSV file, for example
            ``results/compiled/subject_parameters.csv``.

    Returns:
        Table with one row per participant and condition, with at least the
        columns ``condition``, ``learning_rate`` and ``subject_id``.

    Raises:
        ValueError: If a required column is missing, if no
            ``rescorla_wagner_single_value`` rows remain, or if a participant
            appears more than once in the same condition.
    """
    df = pd.read_csv(path)
    required = {"condition", "learning_rate", "subject_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Single-LR CSV is missing columns: {missing}"
        )
    if "model" in df.columns:
        df = df.loc[df["model"] == "rescorla_wagner_single_value"].copy()
        if df.empty:
            raise ValueError(
                "Single-LR CSV contains no rescorla_wagner_single_value rows."
            )
    duplicate_mask = df.duplicated(["subject_id", "condition"], keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["subject_id", "condition"]]
        raise ValueError(
            "Single-LR CSV contains duplicate subject/condition rows: "
            f"{duplicates.to_dict('records')}"
        )
    return df


def load_comparison_for_figure2(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the retained models from a model-comparison table, in figure order.

    Accepts both the ArviZ ``az.compare`` output format (model names in the
    table index) and a table read back from CSV, where the model names may sit
    in an unnamed first column called "Unnamed: 0". The output has the standard
    error column named ``se_diff`` and the weight column named
    ``stacking_weight``, plus ``model_display`` (label for the plot) and
    ``y_position`` (row position in the plot).

    Args:
        comparison_df: Table produced by ``az.compare`` (model names in the
            index), or a table with a ``model`` column and the columns
            ``elpd_diff``, ``dse`` (or ``se_diff``) and ``weight`` (or
            ``stacking_weight``).

    Returns:
        Table with one row for each retained model, sorted in figure order, with
        the columns ``model``, ``elpd_diff``, ``se_diff``, ``stacking_weight``,
        ``model_display`` and ``y_position``, plus any other columns of the
        input.

    Raises:
        ValueError: If a required column is missing, if a retained model is
            absent, or if a retained model appears more than once.
    """
    # Handle CSV roundtrip: unnamed index column reads as "Unnamed: 0"
    unnamed_cols = [c for c in comparison_df.columns
                    if c.startswith("Unnamed:")]
    if unnamed_cols and "model" not in comparison_df.columns:
        comparison_df = comparison_df.rename(
            columns={unnamed_cols[0]: "model"}
        )
    elif unnamed_cols:
        comparison_df = comparison_df.drop(columns=unnamed_cols)

    # Normalise: ensure a 'model' column exists
    if "model" not in comparison_df.columns:
        # ArviZ `az.compare` output — model names are in the index
        comparison_df = comparison_df.reset_index().rename(
            columns={"index": "model"}
        )

    if "se_diff" not in comparison_df.columns and "dse" not in comparison_df.columns:
        raise ValueError(
            "Comparison table is missing 'dse' or 'se_diff' (standard error of elpd_diff)."
        )
    required = {"elpd_diff"}
    missing = required - set(comparison_df.columns)
    if missing:
        raise ValueError(
            f"Comparison table is missing columns: {missing}. "
            f"Available columns: {list(comparison_df.columns)}. "
            "The 'elpd_diff' column is required."
        )

    # Map ArviZ column names: dse → se_diff, weight → stacking_weight
    if "se_diff" not in comparison_df.columns:
        comparison_df = comparison_df.rename(columns={"dse": "se_diff"})
    if "stacking_weight" not in comparison_df.columns and "weight" in comparison_df.columns:
        comparison_df = comparison_df.rename(columns={"weight": "stacking_weight"})

    # Validate that stacking_weight exists
    if "stacking_weight" not in comparison_df.columns:
        raise ValueError(
            "Comparison table is missing 'stacking_weight' (or 'weight') column. "
            "Stacking weights are required for Figure 2."
        )

    # Filter to the exact retained model set, while ignoring unrelated rows.
    allowlist = list(FIGURE2_MODEL_ORDER)
    mask = comparison_df["model"].isin(allowlist)
    filtered = comparison_df.loc[mask].copy()

    counts = filtered["model"].value_counts()
    missing_models = [model for model in allowlist if counts.get(model, 0) == 0]
    duplicate_models = [model for model in allowlist if counts.get(model, 0) > 1]
    if missing_models or duplicate_models:
        raise ValueError(
            "Figure 2 comparison table must contain each retained model exactly once. "
            f"Missing: {missing_models}; duplicated: {duplicate_models}. "
            f"Required models: {allowlist}"
        )

    filtered["model_display"] = filtered["model"].map(MODEL_DISPLAY_NAMES)
    filtered["y_position"] = filtered["model"].map(
        {model: index for index, model in enumerate(FIGURE2_MODEL_ORDER)}
    )
    return filtered.sort_values("y_position").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Posterior loading helpers  (integrated ArviZ format)
# ---------------------------------------------------------------------------

def load_posterior_and_metadata(
    posterior_path: Path,
    metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the posterior samples and metadata of a model fitted to all conditions.

    Args:
        posterior_path: Path to ``posterior.nc``, the ArviZ NetCDF file with the
            posterior samples.
        metadata_path: Path to ``metadata.json``. If ``None``, the file is
            looked for in the same folder as ``posterior_path``.

    Returns:
        Dictionary with these keys:

        - ``posterior``: the ``posterior`` group of the file, which holds the
          posterior samples.
        - ``datatree``: the whole file, as an xarray ``DataTree``.
        - ``metadata``: the contents of ``metadata.json``.
        - ``cond_indices``: dictionary that maps each condition name to the
          list of participant positions in that condition.
        - ``conditions``: the condition names, in the order they first appear,
          limited to the four second-block conditions used in the figures.

    Raises:
        FileNotFoundError: If ``posterior.nc`` or ``metadata.json`` is missing.
    """
    if not posterior_path.is_file():
        raise FileNotFoundError(
            f"Posterior file not found: {posterior_path}"
        )
    if metadata_path is None:
        metadata_path = posterior_path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. "
            "A valid metadata.json is required alongside posterior.nc."
        )

    import xarray as xr
    datatree = xr.open_datatree(posterior_path)
    # The posterior samples live in the 'posterior' group
    posterior_ds = datatree["posterior"]

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Build per-condition subject index masks from metadata
    condition_labels: List[str] = metadata["condition_labels"]
    unique_conditions: List[str] = []
    cond_indices: Dict[str, List[int]] = {}
    for i, cond in enumerate(condition_labels):
        if cond not in cond_indices:
            cond_indices[cond] = []
            unique_conditions.append(cond)
        cond_indices[cond].append(i)

    # Filter to only second-block conditions we care about
    known_conditions = [c for c in unique_conditions if c in CONDITION_ORDER]

    return {
        "posterior": posterior_ds,
        "datatree": datatree,
        "metadata": metadata,
        "cond_indices": cond_indices,
        "conditions": known_conditions,
    }


def _flatten_chains(array: np.ndarray) -> np.ndarray:
    """Join the chains of posterior samples into one long list of samples.

    Args:
        array: Posterior samples with shape ``(n_chains, n_draws, ...)``.

    Returns:
        The same values with shape ``(n_chains * n_draws, ...)``.
    """
    return array.reshape(-1, *array.shape[2:])


def extract_group_params(
    posterior_ds: xr.Dataset,
    model_name: str = WINNING_MODEL,
) -> Dict[str, np.ndarray]:
    """Get the group-level posterior samples, on the latent (unbounded) scale.

    For each learning-rate parameter, the function collects the group mean, the
    effects of current and previous volatility and their interaction, and the
    participant-level spread if it was sampled. It also collects the same values
    for the temperature parameter when they are present.

    Args:
        posterior_ds: The ``posterior`` group of the ArviZ file, which holds the
            posterior samples.
        model_name: Name of the model whose parameters are read.

    Returns:
        Dictionary that maps names such as ``{model_name}_{param}_mu``,
        ``{model_name}_{param}_beta_current_vol``,
        ``{model_name}_{param}_beta_previous_vol`` and
        ``{model_name}_{param}_beta_current_previous_interaction`` to arrays of
        shape ``(n_chains * n_draws,)``.

    Raises:
        ValueError: If a required group-level parameter is missing for either
            learning rate, or if the parameters have different numbers of
            samples.
    """
    prefix = f"{model_name}_"
    result: Dict[str, np.ndarray] = {}

    # Core population effects are always required; subject scale is optional.
    for param in LEARNING_RATE_PARAMS:
        for site in REQUIRED_GROUP_SITES:
            key = f"{prefix}{param}{site}"
            if key not in posterior_ds.data_vars:
                raise ValueError(
                    f"Required group parameter '{key}' not found in posterior. "
                    f"Available variables: {[str(v) for v in posterior_ds.data_vars]}"
                )
            arr = posterior_ds[key].values  # (n_chains, n_draws)
            result[key] = _flatten_chains(arr)
        for site in OPTIONAL_GROUP_SITES:
            key = f"{prefix}{param}{site}"
            if key in posterior_ds.data_vars:
                result[key] = _flatten_chains(posterior_ds[key].values)

    # Also extract temperature group params for compatibility
    temp_sites = [
        f"temperature{site}"
        for site in (*REQUIRED_GROUP_SITES, *OPTIONAL_GROUP_SITES)
    ]
    for ts in temp_sites:
        key = f"{prefix}{ts}"
        if key in posterior_ds.data_vars:
            arr = posterior_ds[key].values
            result[key] = _flatten_chains(arr)

    # Validate consistent sample lengths
    n_samples = None
    for key, arr in result.items():
        if n_samples is None:
            n_samples = len(arr)
        elif len(arr) != n_samples:
            raise ValueError(
                f"Inconsistent sample lengths in group params: "
                f"{key} has {len(arr)} samples, expected {n_samples}."
            )

    return result


def extract_subject_params(
    posterior_ds: xr.Dataset,
    cond_indices: Dict[str, List[int]],
    model_name: str = WINNING_MODEL,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get each participant's posterior samples, split by condition.

    The participant-level parameters (those ending in ``_subj``) are already on
    the natural, bounded scale, because the model itself passes them through the
    sigmoid function.

    Args:
        posterior_ds: The ``posterior`` group of the ArviZ file, which holds the
            posterior samples.
        cond_indices: Dictionary that maps each condition name to the list of
            participant positions in that condition.
        model_name: Name of the model whose parameters are read.

    Returns:
        Nested dictionary. The outer key is the condition name. The inner key is
        the parameter name, for example ``learning_rate_pos``. Each value is an
        array of shape ``(n_samples, n_participants_in_condition)``.

    Raises:
        ValueError: If the posterior has no participant-level samples for
            ``learning_rate_pos`` or ``learning_rate_neg``.
    """
    prefix = f"{model_name}_"
    suffix = "_subj"

    # Find all *_subj variables
    subj_vars = []
    for var_name in posterior_ds.data_vars:
        name = str(var_name)
        if name.startswith(prefix) and name.endswith(suffix):
            param_name = name[len(prefix):-len(suffix)]
            subj_vars.append((name, param_name))

    missing_subject_params = [
        param for param in LEARNING_RATE_PARAMS
        if not any(param_name == param for _, param_name in subj_vars)
    ]
    if missing_subject_params:
        raise ValueError(
            "Posterior is missing required subject-level sites: "
            f"{missing_subject_params}"
        )

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for cond, indices in cond_indices.items():
        cond_dict = {}
        for var_name, param_name in subj_vars:
            arr = posterior_ds[var_name].values  # (n_chains, n_draws, n_subjects_total)
            flat = _flatten_chains(arr)           # (n_samples, n_subjects_total)
            cond_dict[param_name] = flat[:, indices]
        result[cond] = cond_dict

    return result


# ---------------------------------------------------------------------------
# Learning-rate summary builders  (Figure 3)
# ---------------------------------------------------------------------------

def compute_learning_rate_subject_summary(
    subj_by_condition: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Average each participant's posterior learning-rate samples, by condition.

    The participant parameters are already on the natural (0 to 1) scale.

    Args:
        subj_by_condition: Nested dictionary from ``extract_subject_params``,
            with the condition name as the outer key and the parameter name as
            the inner key.

    Returns:
        Table with one row per participant and condition, and the columns
        ``subject_idx`` (position within the condition), ``condition``,
        ``learning_rate_pos_subject_mean`` and
        ``learning_rate_neg_subject_mean``.
    """
    rows = []
    for cond, params in subj_by_condition.items():
        # Determine number of subjects for this condition
        first_key = next(iter(params.keys()))
        n_subjects = params[first_key].shape[1]

        for s in range(n_subjects):
            row: Dict[str, Any] = {
                "subject_idx": s,
                "condition": cond,
            }
            for param in LEARNING_RATE_PARAMS:
                if param in params:
                    arr = params[param]          # (n_samples, n_subjects_cond)
                    row[f"{param}_subject_mean"] = float(arr[:, s].mean())
            rows.append(row)

    return pd.DataFrame(rows)


def compute_condition_population_summary(
    group_params: Dict[str, np.ndarray],
    n_samples: int,
    model_name: str = WINNING_MODEL,
) -> Dict[str, Dict[str, float]]:
    """Find the population mean and 95% interval of each learning rate, by condition.

    For every posterior sample, the group mean and the effects of current and
    previous volatility and their interaction are combined, for each condition,
    and passed through the sigmoid function. This gives the population learning
    rate on the natural (0 to 1) scale. The interval runs from the 2.5th to the
    97.5th percentile of these values across all posterior samples.

    Args:
        group_params: Group-level posterior samples from
            ``extract_group_params``.
        n_samples: Number of posterior samples. It is not used in the
            calculation, which uses every sample in ``group_params``.
        model_name: Name of the model whose parameters are read.

    Returns:
        Nested dictionary. The outer key is the condition name. The inner keys
        are ``{param}_population_mean``, ``{param}_ci_lower`` and
        ``{param}_ci_upper`` for each learning-rate parameter.
    """
    # Build context matrix (4 conditions, 3 regressors)
    contexts = build_context_regressors(list(CONDITION_ORDER))  # (4, 3)

    prefix = f"{model_name}_"
    result: Dict[str, Dict[str, float]] = {}

    for param in LEARNING_RATE_PARAMS:
        mu_key = f"{prefix}{param}_mu"
        beta_c_key = f"{prefix}{param}_beta_current_vol"
        beta_p_key = f"{prefix}{param}_beta_previous_vol"
        beta_i_key = f"{prefix}{param}_beta_current_previous_interaction"

        # Direct access — key existence is guaranteed by extract_group_params
        mu = group_params[mu_key]
        beta_c = group_params[beta_c_key]
        beta_p = group_params[beta_p_key]
        beta_i = group_params[beta_i_key]

        # For each condition, compute natural-scale population mean per sample
        for cond_idx, cond in enumerate(CONDITION_ORDER):
            cv = contexts[cond_idx, 0]
            pv = contexts[cond_idx, 1]
            iv = contexts[cond_idx, 2]

            # Natural-scale population mean for this condition
            # pop_cond = sigmoid(mu + beta_c*cv + beta_p*pv + beta_i*iv)
            pop_per_sample = _sigmoid(mu + beta_c * cv + beta_p * pv + beta_i * iv)

            cond_key = cond
            if cond_key not in result:
                result[cond_key] = {}

            result[cond_key][f"{param}_population_mean"] = float(pop_per_sample.mean())
            lower, upper = posterior_interval(pop_per_sample)
            result[cond_key][f"{param}_ci_lower"] = float(lower)
            result[cond_key][f"{param}_ci_upper"] = float(upper)

    return result


def compute_marginal_effects(
    group_params: Dict[str, np.ndarray],
    n_samples: int,
    model_name: str = WINNING_MODEL,
) -> Dict[str, pd.DataFrame]:
    """Find the effects of volatility on each learning rate, on the 0 to 1 scale.

    For each posterior sample, the function finds the learning rate in each of
    the four conditions. The effect of current volatility is the mean learning
    rate in the two conditions with a volatile current block minus the mean in
    the two with a stable current block. The effect of previous volatility is
    found in the same way from the previous block. The interaction is the
    volatile-minus-stable difference after a volatile block, minus the same
    difference after a stable block. This is done for both the positive and the
    negative learning rate.

    Args:
        group_params: Group-level posterior samples from
            ``extract_group_params``.
        n_samples: Number of posterior samples to use, taken from the start of
            each array.
        model_name: Name of the model whose parameters are read.

    Returns:
        Dictionary with the keys ``learning_rate_pos`` and ``learning_rate_neg``.
        Each value is a table with one row. It holds the posterior mean of the
        ``current``, ``previous`` and ``interaction`` effects, and the lower and
        upper ends of their 95% intervals in the columns ``current_ci_lower``,
        ``current_ci_upper``, ``previous_ci_lower``, ``previous_ci_upper``,
        ``interaction_ci_lower`` and ``interaction_ci_upper``.
    """
    # Build context matrix (4 conditions, 3 regressors)
    contexts = build_context_regressors(list(CONDITION_ORDER))  # (4, 3)

    prefix = f"{model_name}_"
    effects: Dict[str, pd.DataFrame] = {}

    for param in LEARNING_RATE_PARAMS:
        mu_key = f"{prefix}{param}_mu"
        beta_c_key = f"{prefix}{param}_beta_current_vol"
        beta_p_key = f"{prefix}{param}_beta_previous_vol"
        beta_i_key = f"{prefix}{param}_beta_current_previous_interaction"

        # Direct access — key existence is guaranteed by extract_group_params
        mu = group_params[mu_key]
        beta_c = group_params[beta_c_key]
        beta_p = group_params[beta_p_key]
        beta_i = group_params[beta_i_key]

        # For each posterior sample, compute natural-scale per-condition
        # population means, then contrast-code to get marginal effects.
        effect_samples = np.zeros((n_samples, 3))  # (samples, current/prev/interaction)

        for s in range(n_samples):
            # Per-condition natural-scale population values for this sample
            cond_vals = np.zeros(4)
            for c in range(4):
                cv = contexts[c, 0]
                pv = contexts[c, 1]
                iv = contexts[c, 2]
                cond_vals[c] = _sigmoid(
                    float(mu[s]) + float(beta_c[s]) * cv
                    + float(beta_p[s]) * pv
                    + float(beta_i[s]) * iv
                )

            # Contrast-coding: beta_j = sum_c contrasts[c,j] * cond_vals[c]
            # contexts columns: current/previous = ±0.5, interaction = ±0.25
            raw = contexts.T @ cond_vals  # (3,) — interaction is quarter-scaled
            effect_samples[s, 0] = raw[0]                     # current (half-contrast)
            effect_samples[s, 1] = raw[1]                     # previous (half-contrast)
            effect_samples[s, 2] = 4.0 * raw[2]               # interaction → true diff-in-diff

        # 95% interval of each of the three effects, over all posterior samples
        ci_lower, ci_upper = posterior_interval(effect_samples, axis=0)
        effects[param] = pd.DataFrame({
            "current": [float(effect_samples[:, 0].mean())],
            "previous": [float(effect_samples[:, 1].mean())],
            "interaction": [float(effect_samples[:, 2].mean())],
            "current_ci_lower": [float(ci_lower[0])],
            "current_ci_upper": [float(ci_upper[0])],
            "previous_ci_lower": [float(ci_lower[1])],
            "previous_ci_upper": [float(ci_upper[1])],
            "interaction_ci_lower": [float(ci_lower[2])],
            "interaction_ci_upper": [float(ci_upper[2])],
        })

    return effects


def compute_logit_coefficient_summary(
    group_params: Dict[str, np.ndarray],
    model_name: str = WINNING_MODEL,
) -> pd.DataFrame:
    """Summarise the group-level logit coefficients of each learning rate.

    The model defines each learning rate on the logit (unbounded) scale.
    Volatile blocks are coded +0.5 and stable blocks -0.5, so the group mean
    ``mu`` is the typical participant's logit learning rate averaged over the
    four conditions, each volatility ``beta`` is the volatile-minus-stable
    difference in that logit, and the interaction ``beta`` is the
    difference-in-differences. These are the
    quantities the model's priors act on, so this table reports them directly,
    without the sigmoid transformation used for the 0 to 1 scale figures.

    Args:
        group_params: Group-level posterior samples from
            ``extract_group_params``.
        model_name: Name of the model whose parameters are read.

    Returns:
        Table with one row per learning-rate parameter and group-level
        coefficient, and the columns ``figure``, ``parameter``, ``coefficient``
        (the site name without its leading underscore), ``mean``, ``sd``
        (sample standard deviation, ``ddof=1``), ``ci_lower`` and ``ci_upper``
        (95% equal-tailed interval). All values are on the logit scale.
    """
    prefix = f"{model_name}_"
    rows = []

    for param in LEARNING_RATE_PARAMS:
        for site in REQUIRED_GROUP_SITES:
            # Key existence is guaranteed by extract_group_params
            samples = group_params[f"{prefix}{param}{site}"]
            lower, upper = posterior_interval(samples)
            rows.append(
                {
                    "figure": "figure-3",
                    "parameter": param,
                    "coefficient": site.lstrip("_"),
                    "mean": float(samples.mean()),
                    "sd": float(samples.std(ddof=1)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )

    return pd.DataFrame(rows)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert unbounded values to the 0 to 1 range with the sigmoid function.

    Values are clipped to between -15 and 15 first, so that the calculation does
    not overflow.

    Args:
        x: Values on the unbounded (latent) scale.

    Returns:
        Array of the same shape as ``x``, with values between 0 and 1.
    """
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.clip(x, None, 15))),
                    np.exp(np.clip(x, -15, None)) / (1.0 + np.exp(np.clip(x, -15, None))))


# ---------------------------------------------------------------------------
# PPC summary builder  (Figure 2)
# ---------------------------------------------------------------------------

def compute_ppc_summary(
    posterior: Optional[Dict[str, np.ndarray]],
    prepared_data: Optional[Dict[str, np.ndarray]],
    model_name: str = WINNING_MODEL,
    n_posterior_samples: int = 200,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Compare the model's predicted choices with the observed choices, by trial.

    The function picks a random set of posterior samples. For each sample and
    each participant, it runs the model on that participant's outcomes and
    choices to get the predicted probability of choosing option 0 on every
    trial. These probabilities are averaged over participants for each trial, and the
    average is then summarised across samples.

    Args:
        posterior: Participant-level posterior samples, each an array of shape
            ``(n_samples, n_participants)``. They are already on the natural
            (bounded) scale, so no further transformation is applied. Keys are
            either ``{model_name}_{param}_subj`` or just the parameter name, for
            example ``learning_rate_pos``.
        prepared_data: Prepared task data with the keys ``outcomes_0``,
            ``outcomes_1`` and ``choices``. Each is an array of shape
            ``(n_participants, n_trials)``, or a one-dimensional array of all
            participants' trials joined together, which is split into
            participants using the number of participants in ``posterior``.
        model_name: Name of the model to run.
        n_posterior_samples: Number of posterior samples to use. If the
            posterior has fewer samples, all of them are used.
        seed: Random seed for choosing the posterior samples.

    Returns:
        Dictionary with these keys:

        - ``trial_mean_pred``: predicted probability of choosing option 0 on
          each trial, averaged over participants and samples, shape
          ``(n_trials,)``.
        - ``trial_mean_obs``: observed proportion of participants who chose
          option 0 on each trial (``choices`` = 1 means option 0), shape
          ``(n_trials,)``.
        - ``ci_lower``: 2.5th percentile across samples of the predicted
          probability on each trial, shape ``(n_trials,)``.
        - ``ci_upper``: 97.5th percentile across samples, shape ``(n_trials,)``.
        - ``subjects``: predicted probability of choosing option 0 for each
          participant,
          averaged over samples and trials, shape ``(n_participants,)``.
        - ``n_subjects``: number of participants.
        - ``n_trials``: number of trials per participant.

    Raises:
        ValueError: If ``posterior`` or ``prepared_data`` is ``None``, if no
            participant-level parameters are found, if the prepared arrays have
            different shapes or are not one- or two-dimensional, if they cannot be
            split into the posterior's participants, or if the posterior and the
            prepared data have different numbers of participants.
    """
    if posterior is None:
        raise ValueError("posterior must not be None")
    if prepared_data is None:
        raise ValueError("prepared_data must not be None")

    from ..modelling_code.model_simulation import run_model_scan

    # Accept both flat format ({model_name}_{param}_subj) and
    # pre-extracted format ({param} directly)
    arrays = _extract_param_arrays_flat(posterior, model_name=model_name)
    if not arrays:
        # Try pre-extracted format (keys are just param names)
        arrays = {k: np.asarray(v) for k, v in posterior.items()}
    if not arrays:
        raise ValueError(
            f"No subject-level parameters found for model '{model_name}'. "
            f"Expected keys like '{model_name}_<param>_subj'."
        )

    outcomes_0 = np.asarray(prepared_data["outcomes_0"])
    outcomes_1 = np.asarray(prepared_data["outcomes_1"])
    choices = np.asarray(prepared_data["choices"])

    if outcomes_0.shape != outcomes_1.shape or outcomes_0.shape != choices.shape:
        raise ValueError(
            "Prepared PPC arrays must have matching shapes: "
            f"outcomes_0={outcomes_0.shape}, outcomes_1={outcomes_1.shape}, "
            f"choices={choices.shape}"
        )

    first_key = next(iter(arrays.values()))
    if choices.ndim == 1:
        n_subjects = first_key.shape[1] if first_key.ndim > 1 else 1
        if n_subjects == 0 or len(choices) % n_subjects:
            raise ValueError(
                "One-dimensional prepared PPC arrays cannot be divided into "
                f"posterior subjects: choices={choices.shape}, n_subjects={n_subjects}"
            )
        n_trials = len(choices) // n_subjects
        outcomes_0 = outcomes_0.reshape(n_subjects, n_trials)
        outcomes_1 = outcomes_1.reshape(n_subjects, n_trials)
        choices = choices.reshape(n_subjects, n_trials)
    elif choices.ndim == 2:
        n_subjects, n_trials = choices.shape
    else:
        raise ValueError(f"Prepared PPC arrays must be one- or two-dimensional, got {choices.ndim}")

    # Determine sample count
    n_total = first_key.shape[0]
    if first_key.ndim > 1 and first_key.shape[1] != n_subjects:
        raise ValueError(
            "Posterior and prepared PPC data have different subject counts: "
            f"posterior={first_key.shape[1]}, prepared={n_subjects}"
        )
    n_used = min(n_posterior_samples, n_total)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(n_total, size=n_used, replace=False)

    import jax.numpy as jnp

    # Compute predicted probabilities for each sampled posterior draw
    all_probs = np.zeros((n_used, n_subjects, n_trials))

    for s_idx, post_idx in enumerate(sample_indices):
        for subj_idx in range(n_subjects):
            params = {
                name: float(arr[post_idx, subj_idx].item() if arr.ndim > 1 else arr[post_idx].item())
                for name, arr in arrays.items()
            }
            trial_data = {
                "outcomes_0": jnp.array(outcomes_0[subj_idx]),
                "outcomes_1": jnp.array(outcomes_1[subj_idx]),
                "choices": jnp.array(choices[subj_idx]),
            }
            _, outputs = run_model_scan(
                model_name=model_name,
                model_params=params,
                trial_data=trial_data,
            )
            all_probs[s_idx, subj_idx] = np.asarray(outputs["prev_pos"])

    # Per-trial group summary
    # all_probs shape: (n_used, n_subjects, n_trials)
    trial_mean_per_sample = all_probs.mean(axis=1)  # (n_used, n_trials)
    trial_mean_pred = trial_mean_per_sample.mean(axis=0)  # (n_trials,)
    ci_lower, ci_upper = posterior_interval(trial_mean_per_sample, axis=0)
    trial_mean_obs = choices.mean(axis=0)

    # Per-subject summary
    subj_mean_pred = all_probs.mean(axis=(0, 2))  # (n_subjects,)

    return {
        "trial_mean_pred": trial_mean_pred,
        "trial_mean_obs": trial_mean_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "subjects": subj_mean_pred,
        "n_subjects": n_subjects,
        "n_trials": n_trials,
    }


def _extract_param_arrays_flat(
    posterior: Dict[str, np.ndarray],
    model_name: str = WINNING_MODEL,
) -> Dict[str, np.ndarray]:
    """Pick out the participant-level parameters from a dictionary of samples.

    Only keys that match ``{model_name}_{param}_subj`` are kept. The arrays are
    expected to have shape ``(n_samples, n_participants)``. The values are
    returned unchanged, because they are already on the natural scale.
    ``compute_ppc_summary`` uses this function.

    Args:
        posterior: Dictionary that maps parameter names to posterior samples.
        model_name: Name of the model whose parameters are kept.

    Returns:
        Dictionary that maps each parameter name, without the model prefix and
        the ``_subj`` ending (for example ``learning_rate_pos``), to its array.
        It is empty if no key matches.
    """
    result = {}
    prefix = f"{model_name}_"
    suffix = "_subj"
    for key, val in posterior.items():
        if key.startswith(prefix) and key.endswith(suffix):
            param_name = key[len(prefix):-len(suffix)]
            result[param_name] = np.asarray(val)
    return result


# ---------------------------------------------------------------------------
# Figure 1: Single-learning-rate subject jitter + group mean + 95% CI
# ---------------------------------------------------------------------------

def _logit_learning_rate(values: Sequence[float], eps: float = np.finfo(float).eps) -> np.ndarray:
    """Convert learning rates to logits (log odds).

    Values are clipped to between ``eps`` and ``1 - eps`` first, so that rates of
    exactly 0 or 1 do not give infinite logits.

    Args:
        values: Learning rates between 0 and 1.
        eps: Smallest distance allowed between a value and 0 or 1.

    Returns:
        Array of logits, with the same length as ``values``.
    """
    values_array = np.asarray(values, dtype=float)
    clipped = np.clip(values_array, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _p_value_label(p_value: float) -> str:
    """Turn a p value into the significance marker shown on the figure.

    Args:
        p_value: The p value of a test.

    Returns:
        ``"***"`` if p is below 0.001, ``"**"`` if below 0.01, ``"*"`` if below
        0.05, and ``"n.s."`` (not significant) otherwise.
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def compute_figure1_bracket_stats(subject_estimates: pd.DataFrame) -> pd.DataFrame:
    """Test the two planned condition comparisons shown in Figure 1.

    Each comparison is a two-sided Welch t-test (which does not assume equal
    variances) on the logit of the participants' learning rates. The tests are
    Stable to Stable against Volatile to Stable, and Stable to Volatile against
    Volatile to Volatile. The p values are not corrected for multiple tests.

    Args:
        subject_estimates: Table with one row per participant and condition,
            with the columns ``condition`` and ``learning_rate``. Missing
            learning rates are ignored.

    Returns:
        Table with one row per comparison and the columns ``condition_a``,
        ``condition_b``, ``comparison``, ``test``, ``p_value_raw`` and
        ``p_label`` (the significance marker drawn on the figure).

    Raises:
        ValueError: If either condition of a comparison has fewer than two
            learning rates.
    """
    from scipy.stats import ttest_ind

    comparisons = (
        ("StableV4_2", "StableV1", "Stable→Stable vs Volatile→Stable"),
        ("VolatileV2", "VolatileV3_2", "Stable→Volatile vs Volatile→Volatile"),
    )
    rows = []
    for condition_a, condition_b, label in comparisons:
        values_a = subject_estimates.loc[
            subject_estimates["condition"] == condition_a, "learning_rate"
        ].dropna()
        values_b = subject_estimates.loc[
            subject_estimates["condition"] == condition_b, "learning_rate"
        ].dropna()
        if len(values_a) < 2 or len(values_b) < 2:
            raise ValueError(
                f"Figure 1 comparison '{label}' requires at least two finite "
                "values per group."
            )
        result = ttest_ind(
            _logit_learning_rate(values_a),
            _logit_learning_rate(values_b),
            equal_var=False,
            alternative="two-sided",
        )
        p_value = float(result.pvalue)
        rows.append({
            "condition_a": condition_a,
            "condition_b": condition_b,
            "comparison": label,
            "test": "two-sided independent Welch t-test on logit learning rate",
            "p_value_raw": p_value,
            "p_label": _p_value_label(p_value),
        })
    return pd.DataFrame(rows)


def _draw_figure1_bracket(ax, x_a: float, x_b: float, y: float, label: str) -> None:
    """Draw a bracket over two conditions with a significance marker on top.

    The bracket's end ticks are 1.5% of the height of the y axis.

    Args:
        ax: Matplotlib axes to draw on.
        x_a: Horizontal position of the left end of the bracket.
        x_b: Horizontal position of the right end of the bracket.
        y: Height of the bottom of the bracket, in data units.
        label: Text written above the middle of the bracket.
    """
    height = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x_a, x_a, x_b, x_b], [y, y + height, y + height, y], color="black", linewidth=0.8)
    ax.text(
        (x_a + x_b) / 2, y + height, label, ha="center", va="bottom",
        fontsize=TICK_FONT_SIZE,
    )

def plot_figure1(
    subject_estimates: pd.DataFrame,
    output_stem: str,
    output_formats: Sequence[str] = OUTPUT_FORMATS,
) -> None:
    """Draw and save Figure 1, the single-learning-rate estimates by condition.

    Each condition shows the participants' learning rates as jittered dots, the
    group mean, and an approximate 95% interval (mean plus or minus 1.96
    standard errors). Brackets above the dots mark the two planned comparisons
    from ``compute_figure1_bracket_stats``.

    Args:
        subject_estimates: Table with one row per participant and condition,
            with the columns ``condition`` and ``learning_rate``.
        output_stem: Start of the output file path. Each file extension in
            ``output_formats`` is added to it.
        output_formats: File extensions to save, for example
            ``(".pdf", ".svg", ".png")``.
    """
    import matplotlib.pyplot as plt

    # Filter to only second-block conditions in agreed order
    conditions_in_order = [c for c in CONDITION_ORDER
                           if c in subject_estimates["condition"].unique()]
    labels = [SHORT_CONDITION_LABELS[c] for c in conditions_in_order]
    colours = [CONDITION_COLOURS[CONDITION_ORDER.index(c)]
               for c in conditions_in_order]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES_INCHES["figure1"])

    # Data for each condition
    positions = []
    all_data = []
    group_means = []
    ci_lowers = []
    ci_uppers = []

    for i, cond in enumerate(conditions_in_order):
        vals = subject_estimates.loc[
            subject_estimates["condition"] == cond, "learning_rate"
        ].values
        positions.append(i + 1)
        all_data.append(vals)
        mean_val = np.mean(vals)
        sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals))
        ci = 1.96 * sem_val  # approximate 95% CI
        group_means.append(mean_val)
        ci_lowers.append(mean_val - ci)
        ci_uppers.append(mean_val + ci)

    # Subject jitter
    for i, (pos, vals) in enumerate(zip(positions, all_data)):
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            pos + jitter, vals,
            alpha=0.25, s=20, color=colours[i],
            edgecolors="none", zorder=2,
        )

    # Group means with approximate 95% confidence intervals.
    for i, (pos, m, lo, hi) in enumerate(
        zip(positions, group_means, ci_lowers, ci_uppers)
    ):
        ax.plot([pos], [m], "o", color=SUMMARY_COLOUR, markersize=8, zorder=5)
        ax.plot(
            [pos, pos], [lo, hi], "-", color=SUMMARY_COLOUR,
            linewidth=2, zorder=4,
        )

    stats = compute_figure1_bracket_stats(subject_estimates)
    finite_values = subject_estimates["learning_rate"].dropna().values
    y_max = float(np.max(finite_values))
    y_min = float(np.min(finite_values))
    y_range = max(y_max - y_min, 0.05)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.28 * y_range)
    for bracket_idx, row in stats.iterrows():
        _draw_figure1_bracket(
            ax,
            CONDITION_ORDER.index(row["condition_a"]) + 1,
            CONDITION_ORDER.index(row["condition_b"]) + 1,
            y_max + (0.08 + bracket_idx * 0.10) * y_range,
            row["p_label"],
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("Condition", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Learning rate", fontsize=LABEL_FONT_SIZE)
    ax.set_xlim(0.5, len(positions) + 0.5)
    _despine(ax)

    plt.tight_layout()

    for ext in output_formats:
        save_path = f"{output_stem}{ext}"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Model comparison and posterior predictive checks
# ---------------------------------------------------------------------------

def _plot_model_comparison(ax_forest, ax_weights, comparison: pd.DataFrame) -> None:
    """Draw the model-comparison row at the top of Figure 2.

    The left plot shows each model's difference in expected predictive accuracy
    (ELPD) from the best model, with its standard error. The right plot shows
    each model's stacking weight as a bar. Both plots list the models in the
    same order.

    Args:
        ax_forest: Matplotlib axes for the ELPD differences.
        ax_weights: Matplotlib axes for the stacking weights.
        comparison: Table from ``load_comparison_for_figure2``, with the columns
            ``elpd_diff``, ``se_diff``, ``stacking_weight``, ``model_display``
            and ``y_position``.
    """
    comp = comparison.sort_values("y_position").copy()
    y_pos = comp["y_position"].to_numpy()

    for _, row in comp.iterrows():
        ax_forest.errorbar(
            row["elpd_diff"], row["y_position"],
            xerr=row["se_diff"],
            fmt="o", color=ACCENT_BLUE, capsize=3, markersize=6,
        )
    ax_forest.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels(comp["model_display"], fontsize=TICK_FONT_SIZE)
    ax_forest.tick_params(axis="y", labelleft=True)
    ax_forest.set_xlabel(r"$\Delta$ ELPD (lower is better)", fontsize=LABEL_FONT_SIZE)
    ax_forest.invert_yaxis()

    ax_weights.barh(
        comp["y_position"], comp["stacking_weight"],
        color=ACCENT_BLUE,
    )
    ax_weights.set_yticks(y_pos)
    ax_weights.tick_params(axis="y", labelleft=False)
    ax_weights.set_xlabel("Stacking weight", fontsize=LABEL_FONT_SIZE)
    ax_weights.set_xlim(0, 1)

    _despine(ax_forest)
    _despine(ax_weights)

def _plot_ppc_panel(
    ax,
    condition: str,
    ppc: Dict[str, np.ndarray],
    reversal_trials: Optional[Dict[str, np.ndarray]] = None,
    show_legend: bool = False,
) -> None:
    """Draw one condition's comparison of predicted and observed choices.

    The plot shows the predicted probability of choosing option 0 on each trial
    with its 95% interval, the observed proportion of participants who chose
    option 0 on each trial, and optionally a dotted vertical line at each
    reversal of the better option.

    Args:
        ax: Matplotlib axes to draw on.
        condition: Condition name, which sets the colour and the title.
        ppc: Summary for this condition from ``compute_ppc_summary``. It needs
            the keys ``trial_mean_pred``, ``trial_mean_obs``, ``ci_lower`` and
            ``ci_upper``.
        reversal_trials: Dictionary that maps each condition name to the
            zero-based trial numbers at which the better option reverses. If
            ``None``, or if the condition is missing, no lines are drawn.
        show_legend: Whether to add a legend for the predicted and observed
            lines.
    """
    trials = np.arange(len(ppc["trial_mean_pred"])) + 1
    colour = CONDITION_COLOURS[CONDITION_ORDER.index(condition)]
    ax.fill_between(
        trials, ppc["ci_lower"], ppc["ci_upper"],
        alpha=0.2, color=colour, label="_nolegend_",
    )
    ax.plot(
        trials, ppc["trial_mean_pred"], color=colour, linewidth=2,
        linestyle="-", label="Predicted",
    )
    ax.plot(
        trials, ppc["trial_mean_obs"], color="black", linewidth=1.5,
        label="Observed",
    )
    if reversal_trials and condition in reversal_trials:
        for rt in reversal_trials[condition]:
            ax.axvline(rt + 1, color="gray", linestyle=":", alpha=0.4)

    ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Trial", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("P(Choose left)", fontsize=LABEL_FONT_SIZE)
    if show_legend:
        ax.legend(fontsize=TICK_FONT_SIZE)
    _despine(ax)


def plot_figure2(
    comparison: pd.DataFrame,
    ppc_by_condition: Dict[str, Dict[str, np.ndarray]],
    output_stem: str,
    output_formats: Sequence[str] = OUTPUT_FORMATS,
    reversal_trials: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Draw and save Figure 2, the model comparison and the predicted choices.

    The top row compares the retained models. Below it, one plot per condition
    compares the main model's predicted choices with the observed choices.
    Only the conditions in ``ppc_by_condition`` are drawn, in figure order, and
    any unused plots at the end are hidden.

    Args:
        comparison: Model-comparison table from ``load_comparison_for_figure2``.
        ppc_by_condition: Dictionary that maps each condition name to its
            summary from ``compute_ppc_summary``.
        output_stem: Start of the output file path. Each file extension in
            ``output_formats`` is added to it.
        output_formats: File extensions to save, for example
            ``(".pdf", ".svg", ".png")``.
        reversal_trials: Dictionary that maps each condition name to the
            zero-based trial numbers at which the better option reverses. If
            ``None``, no reversal lines are drawn.
    """
    import matplotlib.pyplot as plt

    conditions = [c for c in CONDITION_ORDER if c in ppc_by_condition]
    fig = plt.figure(figsize=FIGURE_SIZES_INCHES["figure2"])
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 0.6, 0.6])
    model_grid = grid[0, :].subgridspec(1, 2, width_ratios=[2, 1])
    ax_forest = fig.add_subplot(model_grid[0, 0])
    ax_weights = fig.add_subplot(model_grid[0, 1], sharey=ax_forest)
    _plot_model_comparison(ax_forest, ax_weights, comparison)

    ppc_axes = [fig.add_subplot(grid[row, column]) for row in (1, 2) for column in (0, 1)]
    for index, condition in enumerate(conditions):
        _plot_ppc_panel(
            ppc_axes[index], condition, ppc_by_condition[condition], reversal_trials,
            show_legend=index == 0,
        )
    for ax in ppc_axes[len(conditions):]:
        ax.axis("off")

    add_panel_labels(fig, [ax_forest, ax_weights, *ppc_axes[:len(conditions)]])

    plt.tight_layout()

    for ext in output_formats:
        fig.savefig(f"{output_stem}{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Learning-rate subject + population forest plots
# ---------------------------------------------------------------------------

def plot_figure3(
    subject_summary: pd.DataFrame,
    population_summary: Dict[str, Dict[str, float]],
    effects: Dict[str, pd.DataFrame],
    output_stem: str,
    output_formats: Sequence[str] = OUTPUT_FORMATS,
) -> None:
    """Draw and save Figure 3, the learning rates of the main model.

    There is one row for the positive learning rate and one for the negative
    learning rate. The left plot of each row shows every participant's posterior
    mean learning rate by condition, with the population mean and 95% interval
    for each condition. The right plot shows the effects of current volatility,
    previous volatility and their interaction, each with its 95% interval.

    Args:
        subject_summary: Table from ``compute_learning_rate_subject_summary``.
        population_summary: Nested dictionary from
            ``compute_condition_population_summary``, in the form
            ``{condition: {"<param>_population_mean": ...,
            "<param>_ci_lower": ..., "<param>_ci_upper": ...}}``.
        effects: Dictionary from ``compute_marginal_effects``, with one table
            per learning-rate parameter.
        output_stem: Start of the output file path. Each file extension in
            ``output_formats`` is added to it.
        output_formats: File extensions to save, for example
            ``(".pdf", ".svg", ".png")``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES_INCHES["figure3"])

    for row_idx, param in enumerate(["learning_rate_pos", "learning_rate_neg"]):
        ax_cond = axes[row_idx, 0]
        ax_forest = axes[row_idx, 1]

        param_label = {
            "learning_rate_pos": "Positive",
            "learning_rate_neg": "Negative",
        }[param]

        # ---- Left panel: per-condition subject means + population ----
        conditions_in_data = [
            c for c in CONDITION_ORDER
            if c in subject_summary["condition"].values
        ]
        col_mean = f"{param}_subject_mean"

        for ci, cond in enumerate(conditions_in_data):
            mask = subject_summary["condition"] == cond
            vals = subject_summary.loc[mask, col_mean].values
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(ci).uniform(-0.15, 0.15, size=len(vals))
            colour = CONDITION_COLOURS[CONDITION_ORDER.index(cond)]
            pos = ci + 1

            ax_cond.scatter(
                pos + jitter, vals,
                alpha=0.25, s=20, color=colour, edgecolors="none",
            )

        # Population mean + 95% CI (per-condition from group posterior)
        for ci, cond in enumerate(conditions_in_data):
            pos = ci + 1
            colour = CONDITION_COLOURS[CONDITION_ORDER.index(cond)]
            pop_mean = population_summary.get(cond, {}).get(f"{param}_population_mean", np.nan)
            ci_lower_val = population_summary.get(cond, {}).get(f"{param}_ci_lower", np.nan)
            ci_upper_val = population_summary.get(cond, {}).get(f"{param}_ci_upper", np.nan)

            if not np.isnan(pop_mean):
                ax_cond.plot(
                    [pos], [pop_mean], "o", color=SUMMARY_COLOUR,
                    markersize=8, zorder=5,
                )
                ax_cond.plot(
                    [pos, pos], [ci_lower_val, ci_upper_val], "-",
                    color=SUMMARY_COLOUR, linewidth=2, zorder=4,
                )

        ax_cond.set_xticks(range(1, len(conditions_in_data) + 1))
        ax_cond.set_xticklabels(
            [SHORT_CONDITION_LABELS[c] for c in conditions_in_data],
            fontsize=TICK_FONT_SIZE,
        )
        ax_cond.set_xlabel("Condition", fontsize=LABEL_FONT_SIZE)
        ax_cond.set_ylabel(f"{param_label} learning rate", fontsize=LABEL_FONT_SIZE)
        _despine(ax_cond)

        # ---- Right panel: forest plot of marginal effects ----
        if param in effects:
            ef = effects[param]
            ef_cols = ["current", "previous", "interaction"]
            y_vals = np.arange(len(ef_cols))
            ef_values = [ef[col].iloc[0] for col in ef_cols]
            ci_lower_vals = [ef[f"{col}_ci_lower"].iloc[0] for col in ef_cols]
            ci_upper_vals = [ef[f"{col}_ci_upper"].iloc[0] for col in ef_cols]

            for i, (val, lo, hi) in enumerate(zip(ef_values, ci_lower_vals, ci_upper_vals)):
                ax_forest.plot([val], [i], "o", color=ACCENT_BLUE, markersize=8)
                ax_forest.plot([lo, hi], [i, i], "-", color=ACCENT_BLUE, linewidth=2)

            ax_forest.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax_forest.set_yticks(y_vals)
            ax_forest.set_yticklabels(
                ["Current", "Previous", "Interaction"], fontsize=TICK_FONT_SIZE
            )
            ax_forest.set_xlabel(
                f"Coefficient\n(effect on {param_label.lower()} learning rate)",
                fontsize=LABEL_FONT_SIZE,
            )
            ax_forest.set_ylim(-0.5, len(ef_cols) - 0.5)
            ax_forest.invert_yaxis()
            _despine(ax_forest)

    add_panel_labels(fig, axes)
    plt.tight_layout()

    for ext in output_formats:
        fig.savefig(f"{output_stem}{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task performance: optimal choices and safe outcomes
# ---------------------------------------------------------------------------

# Lowest acceptable share of a block's trials on which the rebuilt better ball
# was safe. The design gives about 0.8; wrong reversal points give about 0.5.
MIN_BETTER_BALL_SAFE_RATE = 0.65

CHANCE_PROPORTION = 0.5

def rebuild_better_option(
    outcomes_0: np.ndarray,
    outcomes_1: np.ndarray,
    block_type: str,
    condition: str,
    subject_ids: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Rebuild which ball was better on each trial for every participant.

    The reversal points come from the task design (``better_option_by_trial``).
    The data can't show which ball started as the better one, so for each
    participant both starting options are tried and the one under which the
    better ball was safe more often is kept. The kept schedule must make the
    better ball safe on at least ``MIN_BETTER_BALL_SAFE_RATE`` of the block's
    trials, otherwise the design does not fit the data and an error is raised.

    Args:
        outcomes_0: Reward-coded outcomes of option 0 (1 = safe), shape
            ``(n_participants, n_trials)``.
        outcomes_1: Reward-coded outcomes of option 1, same shape.
        block_type: ``"Stable"`` or ``"Volatile"``.
        condition: Condition name, used in error messages.
        subject_ids: Participant IDs in row order, used in error messages.

    Returns:
        Dictionary with ``better_option`` (``(n_participants, n_trials)``, 0/1),
        ``first_better_option``, ``better_ball_safe_rate`` and
        ``lowest_segment_safe_rate`` (each ``(n_participants,)``).

    Raises:
        ValueError: If the block does not have ``BLOCK_TRIAL_COUNT`` trials or
            the rebuilt schedule fails the safe-rate check for any participant.
    """
    outcomes_0 = np.asarray(outcomes_0, dtype=float)
    outcomes_1 = np.asarray(outcomes_1, dtype=float)
    n_participants, n_trials = outcomes_0.shape
    if n_trials != BLOCK_TRIAL_COUNT:
        raise ValueError(
            f"Condition '{condition}' has {n_trials} trials per participant; "
            f"the task design has {BLOCK_TRIAL_COUNT}."
        )

    candidates = np.stack(
        [better_option_by_trial(block_type, first) for first in (0, 1)]
    )  # (2, n_trials)
    # Safe outcome of the better ball, for each participant and starting option.
    better_ball_safe = np.where(
        candidates[None, :, :] == 0, outcomes_0[:, None, :], outcomes_1[:, None, :]
    )  # (n_participants, 2, n_trials)
    rates = better_ball_safe.mean(axis=2)
    first_better_option = np.argmax(rates, axis=1)
    rows = np.arange(n_participants)
    kept_safe = better_ball_safe[rows, first_better_option]
    better_ball_safe_rate = rates[rows, first_better_option]

    failed = np.flatnonzero(better_ball_safe_rate < MIN_BETTER_BALL_SAFE_RATE)
    if failed.size:
        details = ", ".join(
            f"{subject_ids[i]} ({better_ball_safe_rate[i]:.3f})" for i in failed
        )
        raise ValueError(
            f"Rebuilt {block_type.lower()} schedule does not fit condition "
            f"'{condition}': the better ball was safe on fewer than "
            f"{MIN_BETTER_BALL_SAFE_RATE} of trials for {details}."
        )

    interval = REVERSAL_INTERVAL_TRIALS[block_type]
    segment_rates = kept_safe.reshape(n_participants, -1, interval).mean(axis=2)
    return {
        "better_option": candidates[first_better_option],
        "first_better_option": first_better_option,
        "better_ball_safe_rate": better_ball_safe_rate,
        "lowest_segment_safe_rate": segment_rates.min(axis=1),
    }


def compute_task_performance(
    prepared_by_condition: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every participant's choices against the rebuilt task schedule.

    On each trial the chosen option is ``1 - choices`` (``choices`` = 1 means
    option 0 was chosen). A choice is optimal when it picks the better ball,
    and safe when the chosen ball's reward-coded outcome is 1. Each
    participant's trials are pooled across the blocks they appear in.

    Args:
        prepared_by_condition: Reward-coded prepared data for each of the eight
            conditions in ``CONDITION_METADATA``. Each entry holds ``choices``,
            ``outcomes_0``, ``outcomes_1`` and ``subject_ids``.

    Returns:
        Tuple of the participant table (one row per participant) and the
        schedule-check table (one row per participant and block).

    Raises:
        ValueError: If a condition is missing or unknown, arrays disagree in
            shape, participant IDs do not match the rows or repeat, or the
            rebuilt schedule fails its check.
    """
    from scipy.stats import binomtest

    missing = [c for c in CONDITION_METADATA if c not in prepared_by_condition]
    unknown = [c for c in prepared_by_condition if c not in CONDITION_METADATA]
    if missing or unknown:
        raise ValueError(
            "Task performance needs exactly the eight task conditions; "
            f"missing={missing}, unknown={unknown}."
        )

    block_rows = []
    for condition, metadata in CONDITION_METADATA.items():
        data = prepared_by_condition[condition]
        choices = np.asarray(data["choices"], dtype=float)
        outcomes_0 = np.asarray(data["outcomes_0"], dtype=float)
        outcomes_1 = np.asarray(data["outcomes_1"], dtype=float)
        subject_ids = [str(s) for s in np.asarray(data["subject_ids"]).ravel()]
        if choices.ndim != 2 or not (
            choices.shape == outcomes_0.shape == outcomes_1.shape
        ):
            raise ValueError(
                f"Condition '{condition}' arrays must share one 2D shape: "
                f"choices={choices.shape}, outcomes_0={outcomes_0.shape}, "
                f"outcomes_1={outcomes_1.shape}."
            )
        if len(subject_ids) != choices.shape[0]:
            raise ValueError(
                f"Condition '{condition}' has {len(subject_ids)} participant IDs "
                f"for {choices.shape[0]} rows of choices."
            )
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError(f"Condition '{condition}' has repeated participant IDs.")

        block_type = str(metadata["block_type"])
        schedule = rebuild_better_option(
            outcomes_0, outcomes_1, block_type, condition, subject_ids
        )
        chosen_option = 1 - choices
        optimal = chosen_option == schedule["better_option"]
        safe = choices * outcomes_0 + (1 - choices) * outcomes_1

        for row, subject_id in enumerate(subject_ids):
            block_rows.append({
                "subject_id": subject_id,
                "condition": condition,
                "task_version": metadata["version"],
                "block_number": int(metadata["block_number"]),
                "block_type": block_type,
                "n_trials": choices.shape[1],
                "n_optimal": int(optimal[row].sum()),
                "n_safe": int(safe[row].sum()),
                "first_better_option": int(schedule["first_better_option"][row]),
                "first_better_option_number": int(schedule["first_better_option"][row]) + 1,
                "better_ball_safe_rate": float(schedule["better_ball_safe_rate"][row]),
                "lowest_segment_safe_rate": float(
                    schedule["lowest_segment_safe_rate"][row]
                ),
            })

    schedule_check = pd.DataFrame(block_rows)
    schedule_check["n_better_ball_safe"] = (
        schedule_check["better_ball_safe_rate"] * schedule_check["n_trials"]
    )
    participants = (
        schedule_check.groupby("subject_id", sort=True)
        .agg(
            task_version=("task_version", lambda v: "+".join(sorted(set(v)))),
            n_blocks=("condition", "size"),
            n_trials=("n_trials", "sum"),
            n_optimal=("n_optimal", "sum"),
            n_safe=("n_safe", "sum"),
            n_better_ball_safe=("n_better_ball_safe", "sum"),
        )
        .reset_index()
    )
    schedule_check = schedule_check.drop(columns="n_better_ball_safe")
    participants["subject_number"] = np.arange(len(participants)) + 1
    participants["proportion_optimal"] = participants["n_optimal"] / participants["n_trials"]
    participants["proportion_safe"] = participants["n_safe"] / participants["n_trials"]
    participants["best_possible_safe"] = (
        participants["n_better_ball_safe"] / participants["n_trials"]
    )
    participants = participants.drop(columns="n_better_ball_safe")
    participants["optimal_binomial_p_value"] = [
        float(binomtest(int(k), int(n), CHANCE_PROPORTION, alternative="greater").pvalue)
        for k, n in zip(participants["n_optimal"], participants["n_trials"])
    ]
    participants["optimal_above_chance"] = participants["optimal_binomial_p_value"] < 0.05
    return participants, schedule_check


def summarise_task_performance(participants: pd.DataFrame) -> pd.DataFrame:
    """Summarise the two task-performance measures across participants.

    For each measure this gives the mean, SD and t-based 95% CI of the
    participants' proportions, and a two-sided one-sample t-test against
    chance (0.5). Safe outcomes also get the best possible rate; optimal
    choices also get how many participants were individually above chance.

    Args:
        participants: Participant table from ``compute_task_performance``.

    Returns:
        Table with one row per measure.

    Raises:
        ValueError: If there are fewer than two participants.
    """
    from scipy import stats

    n = len(participants)
    if n < 2:
        raise ValueError(f"Task performance summary needs at least two participants, got {n}.")

    rows = []
    for measure, column in (
        ("optimal_choices", "proportion_optimal"),
        ("safe_outcomes", "proportion_safe"),
    ):
        values = participants[column].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
        test = stats.ttest_1samp(values, popmean=CHANCE_PROPORTION)
        row = {
            "measure": measure,
            "n_participants": n,
            "mean": mean,
            "sd": sd,
            "ci_lower": mean - half_width,
            "ci_upper": mean + half_width,
            "chance": CHANCE_PROPORTION,
            "t": float(test.statistic),
            "df": n - 1,
            "p_value": float(test.pvalue),
            "test": "two-sided one-sample t-test against 0.5 on participant proportions",
            "best_possible": np.nan,
            "n_above_chance": np.nan,
            "percent_above_chance": np.nan,
            "above_chance_test": "",
        }
        if measure == "safe_outcomes":
            row["best_possible"] = float(participants["best_possible_safe"].mean())
        else:
            n_above = int(participants["optimal_above_chance"].sum())
            row["n_above_chance"] = n_above
            row["percent_above_chance"] = 100.0 * n_above / n
            row["above_chance_test"] = (
                "one-sided exact binomial test against 0.5 per participant, p < .05"
            )
        rows.append(row)
    return pd.DataFrame(rows)
