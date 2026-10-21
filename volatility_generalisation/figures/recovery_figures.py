"""Supplementary figures that show how well the models and parameters are recovered.

Recovery means simulating data from known models and parameter values, fitting
them again, and checking whether the known values come back. This module reads
the tables that the recovery workflows have already saved, checks that nothing
is missing, and draws three figures without fitting anything again:

* S1, model recovery: how often each simulated model is the best-fitting model.
* S2, participant-level parameter recovery of the integrated models.
* S3, group-level effect recovery of the integrated models.

Fits that did not converge are drawn and marked, never left out. A missing
table, replicate, model or parameter stops the run.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..modelling_code.data_generation import load_parameter_bounds
from ..modelling_code.integrated_model_recovery import build_recovery_matrices
from ..modelling_code.integrated_model_selection import get_default_models
from .publication_figures import (
    FIGURE2_MODEL_ORDER,
    LABEL_FONT_SIZE,
    MODEL_DISPLAY_NAMES,
    OUTPUT_FORMATS,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    _despine,
    add_panel_labels,
)

# Bounds of each parameter that a model estimates: {model: {parameter: (min, max)}}.
ParameterDesign = Dict[str, Dict[str, Tuple[float, float]]]

# Parameter names in words, followed by the symbol in brackets, written in
# matplotlib's maths notation. Each label has the form "name ($symbol$)".
PARAMETER_LABELS: Dict[str, str] = {
    "learning_rate": r"Learning rate ($\alpha$)",
    "learning_rate_pos": r"Learning rate, positive ($\alpha^{+}$)",
    "learning_rate_neg": r"Learning rate, negative ($\alpha^{-}$)",
    "temperature": r"Choice temperature ($T$)",
    "kappa": r"Learning-rate scale ($\kappa$)",
    "gamma": r"Associability update rate ($\gamma$)",
    "tau": r"Update rate ($\tau$)",
    "tau_pos": r"Update rate, positive ($\tau^{+}$)",
    "tau_neg": r"Update rate, negative ($\tau^{-}$)",
    "decay": r"Memory decay ($\lambda$)",
}

# Settings that draw the symbols in the DM Sans font where it has the glyph. DM
# Sans has no Greek letters, so those come from STIX Sans, a similar sans-serif
# font that matplotlib ships. DM Sans is registered by ``ensure_dm_sans_font``.
MATH_FONT_SETTINGS: Dict[str, str] = {
    "mathtext.fontset": "custom",
    "mathtext.rm": "DM Sans",
    "mathtext.it": "DM Sans:italic",
    "mathtext.bf": "DM Sans:bold",
    "mathtext.fallback": "stixsans",
}

# The group-level effects that S3 draws, in panel order, in words. The spread
# between participants is saved in the table but is not drawn.
PLOTTED_EFFECT_LABELS: Dict[str, str] = {
    "mu": "Group mean",
    "beta_current_vol": "Current volatility",
    "beta_previous_vol": "Previous volatility",
    "beta_current_previous_interaction": "Current \u00d7 previous volatility",
}

# Colour-blind-safe colours, one per model, in the order of FIGURE2_MODEL_ORDER.
MODEL_COLOURS: Dict[str, str] = dict(
    zip(FIGURE2_MODEL_ORDER, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"))
)

IDENTITY_LINE_COLOUR = "#8C8C8C"

FIGURE_FILE_NAMES: Dict[str, str] = {
    "s1": "figure-s1-model-recovery",
    "s2": "figure-s2-participant-parameter-recovery",
    "s3": "figure-s3-group-effect-recovery",
}


# ---------------------------------------------------------------------------
# Inputs: what to expect, and checks that everything is there
# ---------------------------------------------------------------------------


def load_recovery_models(config_path: str | Path) -> List[str]:
    """Read the models that the recovery runs should cover.

    Args:
        config_path: Path to the pipeline settings file (``scripts/config.toml``).

    Returns:
        The model names listed under ``fitting``.

    Raises:
        ValueError: If the file lists no models.
    """
    models = list(get_default_models(config_path))
    if not models:
        raise ValueError(f"{config_path} must list models under [fitting].")
    return models


def load_parameter_design(
    models: Sequence[str],
    bounds_path: str | Path | None = None,
) -> ParameterDesign:
    """List the parameters that each model estimates, with their bounds.

    Parameters with a fixed value are left out, because there is nothing to
    recover for them.

    Args:
        models: Model names.
        bounds_path: Path to the parameter bounds file. If None, uses the
            default file.

    Returns:
        Dictionary from model name to a dictionary from parameter name to the
        (minimum, maximum) of the parameter.
    """
    design: ParameterDesign = {}
    for model in models:
        bounds = load_parameter_bounds(model, bounds_path)
        design[model] = {
            name: (float(spec["min"]), float(spec["max"]))
            for name, spec in bounds.items()
            if "fixed" not in spec
        }
    return design


def _row_keys(table: pd.DataFrame, columns: Sequence[str]) -> List[Tuple[Any, ...]]:
    """List the key of every row, as a tuple of the values in the given columns.

    Args:
        table: Table to read.
        columns: Columns that make up the key.

    Returns:
        One tuple per row, in row order, of plain Python values.
    """
    return list(zip(*(table[name].tolist() for name in columns)))


def _check_complete(
    found: Iterable[Tuple[Any, ...]],
    expected: set,
    source: str,
    allow_extra: bool,
) -> None:
    """Stop if rows are missing, repeated or (unless allowed) not expected.

    Args:
        found: The key of each row in the table, repeated if a row repeats.
        expected: The keys that the table must have.
        source: Name of the table, used in the error message.
        allow_extra: If True, keys that are not expected are accepted. This is
            for tables that also hold values that the figure does not draw.

    Raises:
        ValueError: If any expected key is missing, any key appears more than
            once, or (unless ``allow_extra``) any key is not expected. The
            message names each such key.
    """
    counts = Counter(found)
    missing = sorted(expected - set(counts))
    repeated = sorted(key for key, n in counts.items() if n > 1)
    extra = [] if allow_extra else sorted(set(counts) - expected)
    if missing or repeated or extra:
        raise ValueError(
            f"{source} is incomplete or repeats rows. Missing: {missing}. "
            f"More than one row for: {repeated}. Not expected: {extra}."
        )


def _keep_expected(
    table: pd.DataFrame, columns: Sequence[str], expected: set
) -> pd.DataFrame:
    """Keep the rows whose key is one of the expected keys.

    Args:
        table: Table to filter.
        columns: Columns that make up the key.
        expected: The keys to keep.

    Returns:
        The rows of ``table`` with an expected key, in their original order.
    """
    keep = [key in expected for key in _row_keys(table, columns)]
    return table[keep]


def _read_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read one saved table and check that it has the columns the figure needs.

    Args:
        path: Path to the CSV file.
        columns: Columns that must be present.

    Returns:
        The table. A ``converged`` column, if required, is checked to hold
        True or False.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing, or ``converged`` is not
            True or False.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Recovery table not found: {path}")
    table = pd.read_csv(path)
    absent = [name for name in columns if name not in table.columns]
    if absent:
        raise ValueError(f"{path} lacks the columns {absent}.")
    if "converged" in columns and table["converged"].dtype != bool:
        raise ValueError(f"The 'converged' column of {path} must be True or False.")
    return table


def _require_finite(table: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    """Stop if a value that the figure needs is missing or not a finite number.

    Args:
        table: Table to check.
        columns: Numeric columns that must hold finite values in every row.
        source: Name of the table, used in the error message.

    Raises:
        ValueError: If any value in these columns is missing or not finite.
    """
    for name in columns:
        n_bad = int((~np.isfinite(table[name].astype(float))).sum())
        if n_bad:
            raise ValueError(f"{source} has {n_bad} non-finite values in '{name}'.")


def load_model_recovery_winners(
    integrated_dir: Path,
    models: Sequence[str],
    replicates: int,
) -> pd.DataFrame:
    """Read the winning model of every simulated data set, and check all are there.

    Args:
        integrated_dir: Folder with the compiled integrated recovery results
            (the one that holds ``tables/``).
        models: Names of the simulated models.
        replicates: Number of replicates that each simulated model must have.

    Returns:
        The winners table, with one row per replicate and simulated model.

    Raises:
        ValueError: If a simulated model does not have exactly the replicates
            0 to ``replicates`` minus 1, or a winner is not one of ``models``.
    """
    path = integrated_dir / "tables" / "model_recovery_winners.csv"
    winners = _read_table(
        path, ["replicate", "simulated_model", "fitted_model", "converged"]
    )
    expected = {(m, r) for m in models for r in range(replicates)}
    found = _row_keys(winners, ["simulated_model", "replicate"])
    _check_complete(found, expected, str(path), allow_extra=False)
    unknown = sorted(set(winners["fitted_model"]) - set(models))
    if unknown:
        raise ValueError(
            f"{path} has winning models that are not configured: {unknown}."
        )
    return winners


def load_participant_recovery(
    integrated_dir: Path,
    design: ParameterDesign,
    replicates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the participant-level recovery tables and check all are there.

    Args:
        integrated_dir: Folder with the compiled integrated recovery results.
        design: Parameters of each model, from ``load_parameter_design``.
        replicates: Number of replicates that every model must have.

    Returns:
        Tuple of the table with one row per participant, replicate and
        parameter, and the table with one row per replicate and parameter
        that holds the correlation of true and recovered values.

    Raises:
        ValueError: If a replicate, model or parameter is missing or repeated,
            a participant is missing, or a value is not finite.
    """
    rows_path = integrated_dir / "tables" / "parameter_recovery.csv"
    summary_path = integrated_dir / "tables" / "parameter_recovery_summary.csv"
    key = ["replicate", "simulated_model", "parameter"]
    rows = _read_table(
        rows_path,
        [*key, "subject_index", "true_value", "recovered_mean", "converged"],
    )
    summary = _read_table(summary_path, [*key, "pearson_r", "n_subjects", "converged"])

    expected = {
        (replicate, model, parameter)
        for model, parameters in design.items()
        for parameter in parameters
        for replicate in range(replicates)
    }
    # Rows for parameters with a fixed value are also in the tables. They are
    # not drawn, so they are allowed here and dropped below.
    _check_complete(_row_keys(summary, key), expected, str(summary_path), True)
    summary = _keep_expected(summary, key, expected)
    _require_finite(summary, ["pearson_r"], str(summary_path))

    rows = _keep_expected(rows, key, expected)
    _require_finite(rows, ["true_value", "recovered_mean"], str(rows_path))
    # Every fit must have all of its participants, as its summary row says.
    n_found = Counter(_row_keys(rows, key))
    for row_key, n_subjects in zip(_row_keys(summary, key), summary["n_subjects"]):
        if n_found[row_key] != int(n_subjects):
            raise ValueError(
                f"{rows_path} has {n_found[row_key]} participants for {row_key}, "
                f"but {summary_path} says {int(n_subjects)}."
            )
    return rows, summary


def load_group_effect_recovery(
    integrated_dir: Path,
    design: ParameterDesign,
    replicates: int,
) -> pd.DataFrame:
    """Read the group-level effect recovery table and check all are there.

    Args:
        integrated_dir: Folder with the compiled integrated recovery results.
        design: Parameters of each model, from ``load_parameter_design``.
        replicates: Number of replicates that every model must have.

    Returns:
        The table with one row per replicate, model, parameter and drawn
        effect.

    Raises:
        ValueError: If a replicate, model, parameter or drawn effect is missing
            or repeated, or a value is not finite.
    """
    path = integrated_dir / "tables" / "group_effect_recovery.csv"
    key = ["replicate", "simulated_model", "parameter", "effect"]
    values = ["true_value", "posterior_mean", "eti95_lower", "eti95_upper"]
    effects = _read_table(path, [*key, *values, "converged"])
    expected = {
        (replicate, model, parameter, effect)
        for model, parameters in design.items()
        for parameter in parameters
        for replicate in range(replicates)
        for effect in PLOTTED_EFFECT_LABELS
    }
    # The spread between participants is also in the table but is not drawn.
    _check_complete(_row_keys(effects, key), expected, str(path), True)
    effects = _keep_expected(effects, key, expected)
    _require_finite(effects, values, str(path))
    return effects


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def use_dm_sans_for_symbols() -> None:
    """Draw maths symbols in DM Sans, with STIX Sans for the Greek letters.

    This changes matplotlib's settings for the rest of the run. Call
    ``ensure_dm_sans_font`` first, so that DM Sans is registered.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(MATH_FONT_SETTINGS)


def _parameter_title(parameter: str) -> str:
    """Give the title of a parameter's panel, on two lines if the name is long.

    The name is wrapped at 22 characters. The symbol stays on the last line, so
    that no line is cut inside the maths notation.

    Args:
        parameter: Parameter name, for example ``"learning_rate_pos"``.

    Returns:
        The label from ``PARAMETER_LABELS``, with a line break where needed.
    """
    name, symbol = PARAMETER_LABELS[parameter].split(" ($", 1)
    lines = textwrap.wrap(name, 22)
    lines[-1] += f" (${symbol}"
    return "\n".join(lines)


def _one_line_name(model: str) -> str:
    """Give the display name of a model without line breaks.

    Args:
        model: Model name.

    Returns:
        The name from ``MODEL_DISPLAY_NAMES`` on one line.
    """
    return MODEL_DISPLAY_NAMES[model].replace("-\n", "-").replace("\n", " ")


def _save_figure(fig: Any, figures_dir: Path, key: str) -> None:
    """Save a figure in every output format and close it.

    Args:
        fig: Matplotlib figure.
        figures_dir: Folder for the figure files.
        key: Which figure it is, ``"s1"`` to ``"s3"``. It sets the file name.
    """
    import matplotlib.pyplot as plt

    for extension in OUTPUT_FORMATS:
        path = figures_dir / f"{FIGURE_FILE_NAMES[key]}{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _scatter_converged(
    ax: Any,
    x: Any,
    y: Any,
    converged: Any,
    colour: str,
) -> None:
    """Draw points as filled circles, or open circles if the fit did not converge.

    Args:
        ax: Matplotlib axes to draw on.
        x: Horizontal positions.
        y: Vertical positions.
        converged: True for each point whose fit converged.
        colour: Colour of the points.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    done = np.asarray(converged, dtype=bool)
    ax.scatter(x[done], y[done], s=5, color=colour, alpha=0.5, linewidths=0, zorder=3)
    ax.scatter(
        x[~done],
        y[~done],
        s=7,
        facecolors="none",
        edgecolors=colour,
        linewidths=0.6,
        zorder=4,
    )


def _draw_heatmap(
    ax: Any,
    values: np.ndarray,
    cell_text: Sequence[Sequence[str]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
) -> Any:
    """Draw a heat map from 0 to 1 with a text label in every cell.

    Args:
        ax: Matplotlib axes to draw on.
        values: Values, with one row per row label. Colours run from 0 to 1.
        cell_text: Text to print in each cell.
        row_labels: Labels of the rows.
        column_labels: Labels of the columns.

    Returns:
        The image, for adding a colour bar.
    """
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(column_labels)), column_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                cell_text[row][column],
                ha="center",
                va="center",
                fontsize=TICK_FONT_SIZE,
                color="white" if values[row, column] > 0.6 else "#262626",
            )
    for side in ax.spines.values():
        side.set_visible(False)
    return image


def make_model_recovery_figure(
    winners: pd.DataFrame,
    models: Sequence[str],
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Draw S1, the confusion and inversion matrices, and save its source table.

    The matrices come from ``build_recovery_matrices``. The confusion matrix
    shows, for each simulated model, the share of its data sets won by each
    model. The inversion matrix shows, for each winning model, the share of its
    wins that came from each simulated model. The source table also gives the
    number of wins in each cell, and how many of those wins came from fits
    that did not converge.

    Args:
        winners: Winners table from ``load_model_recovery_winners``.
        models: Model names, in the order of the rows and columns.
        figures_dir: Folder for the figure files.
        tables_dir: Folder for the source table.
    """
    import matplotlib.pyplot as plt

    # Model recovery asks: when data come from model A, does model A win? Row
    # i of the confusion matrix is the share of model i's data sets won by each
    # model, so a good recovery has its values on the diagonal.
    confusion, inversion = build_recovery_matrices(winners, list(models))
    cell = ["simulated_model", "fitted_model"]
    n_wins = winners.groupby(cell).size()
    n_not_converged = winners[~winners["converged"]].groupby(cell).size()
    source = pd.DataFrame(
        [
            {
                "simulated_model": simulated,
                "recovered_model": recovered,
                "n_winners": int(n_wins.get((simulated, recovered), 0)),
                "n_winners_not_converged": int(
                    n_not_converged.get((simulated, recovered), 0)
                ),
                "confusion_share": float(confusion.loc[simulated, recovered]),
                "inversion_share": float(inversion.loc[recovered, simulated]),
            }
            for simulated in models
            for recovered in models
        ]
    )
    source.to_csv(tables_dir / "figure-s1-source.csv", index=False)

    names = [MODEL_DISPLAY_NAMES[model] for model in models]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    panels = (
        (axes[0], confusion, "Simulated model", "Recovered model"),
        (axes[1], inversion, "Recovered model", "Simulated model"),
    )
    titles = (
        "Share of each simulated model's\ndata sets won by each model",
        "Share of each recovered model's\nwins from each simulated model",
    )
    for (ax, matrix, row_name, column_name), title in zip(panels, titles):
        values = matrix.to_numpy(dtype=float)
        text = [[f"{value:.2f}" for value in row] for row in values]
        _draw_heatmap(ax, values, text, names, names)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel(column_name, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(row_name, fontsize=LABEL_FONT_SIZE)
    add_panel_labels(fig, axes)
    _save_figure(fig, figures_dir, "s1")


def make_participant_recovery_figure(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    design: ParameterDesign,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Draw S2, participant-level parameter recovery, and save its tables.

    Each panel is one parameter of one model. Every point is one simulated
    participant in one replicate: its true value against the posterior mean
    from the fit of the same model. The dashed line is where recovery is
    perfect. Points from fits that did not converge are open circles. The
    correlation printed in each panel is the median across replicates of the
    correlations already saved in the summary table. The medians and their
    ranges (minimum and maximum) are saved next to the points. Panel titles
    give each parameter's name and symbol.

    Args:
        rows: Participant table from ``load_participant_recovery``.
        summary: Correlation table from ``load_participant_recovery``.
        design: Parameters of each model, with their bounds.
        figures_dir: Folder for the figure files.
        tables_dir: Folder for the source tables.
    """
    import matplotlib.pyplot as plt

    use_dm_sans_for_symbols()
    columns = [
        "replicate",
        "simulated_model",
        "parameter",
        "subject_index",
        "true_value",
        "recovered_mean",
        "converged",
    ]
    source = rows[columns].sort_values(columns[:4]).reset_index(drop=True)
    source.to_csv(tables_dir / "figure-s2-source.csv", index=False)
    by_panel = (
        summary.groupby(["simulated_model", "parameter"])["pearson_r"]
        .agg(median_r="median", min_r="min", max_r="max", n_replicates="size")
        .reset_index()
    )
    by_panel.to_csv(tables_dir / "figure-s2-correlations.csv", index=False)
    stats = by_panel.set_index(["simulated_model", "parameter"])

    n_columns = max(len(parameters) for parameters in design.values())
    fig, axes = plt.subplots(
        len(design),
        n_columns,
        figsize=(7.0, 1.6 * len(design) + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (model, parameters) in enumerate(design.items()):
        for column in range(len(parameters), n_columns):
            axes[row, column].axis("off")
        for column, (parameter, (lower, upper)) in enumerate(parameters.items()):
            ax = axes[row, column]
            is_panel = (source["simulated_model"] == model) & (
                source["parameter"] == parameter
            )
            points = source[is_panel]
            # A point on the dashed line is a perfectly recovered participant.
            ax.plot(
                [lower, upper],
                [lower, upper],
                color=IDENTITY_LINE_COLOUR,
                linestyle="--",
                linewidth=0.6,
                zorder=1,
            )
            _scatter_converged(
                ax,
                points["true_value"],
                points["recovered_mean"],
                points["converged"],
                MODEL_COLOURS[model],
            )
            panel = stats.loc[(model, parameter)]
            ax.text(
                0.04,
                0.96,
                f"r = {panel['median_r']:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=TICK_FONT_SIZE,
            )
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.locator_params(nbins=3)
            ax.set_title(_parameter_title(parameter), fontsize=LABEL_FONT_SIZE)
            _despine(ax)
            if column == 0:
                ax.set_ylabel(MODEL_DISPLAY_NAMES[model], fontsize=LABEL_FONT_SIZE)
    fig.supxlabel("True value", fontsize=LABEL_FONT_SIZE)
    fig.supylabel("Posterior mean (recovered value)", fontsize=LABEL_FONT_SIZE)
    _save_figure(fig, figures_dir, "s2")


def make_group_effect_figure(
    effects: pd.DataFrame,
    design: ParameterDesign,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Draw S3, group-level effect recovery, and save its tables.

    There is one panel for each group-level effect: the group mean, and the
    effects of current volatility, previous volatility and their interaction.
    Each point is one parameter of one model in one replicate. It shows the
    true value of the effect against its posterior mean, with the 95%
    equal-tailed interval, on the scale before the parameter is mapped into its
    bounds. Points are coloured by model, and points from fits that did not
    converge are open circles. The correlation in each panel pools all models,
    parameters and replicates, and is saved next to the points.

    Args:
        effects: Table from ``load_group_effect_recovery``.
        design: Parameters of each model, which gives the models to draw.
        figures_dir: Folder for the figure files.
        tables_dir: Folder for the source tables.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = [
        "replicate",
        "simulated_model",
        "parameter",
        "effect",
        "true_value",
        "posterior_mean",
        "eti95_lower",
        "eti95_upper",
        "converged",
    ]
    source = effects[columns].sort_values(columns[:4]).reset_index(drop=True)
    source.to_csv(tables_dir / "figure-s3-source.csv", index=False)

    # The pooled correlation compares the true effects with their posterior
    # means over every model, parameter and replicate in the panel.
    pooled = []
    for effect in PLOTTED_EFFECT_LABELS:
        points = source[source["effect"] == effect]
        r = np.corrcoef(points["true_value"], points["posterior_mean"])[0, 1]
        pooled.append({"effect": effect, "pooled_r": float(r), "n_points": len(points)})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(tables_dir / "figure-s3-correlations.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.2), constrained_layout=True)
    for ax, (effect, label) in zip(axes.reshape(-1), PLOTTED_EFFECT_LABELS.items()):
        points = source[source["effect"] == effect]
        low = min(points["true_value"].min(), points["eti95_lower"].min())
        high = max(points["true_value"].max(), points["eti95_upper"].max())
        pad = 0.05 * (high - low)
        # A point on the dashed line is a perfectly recovered effect.
        ax.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            color=IDENTITY_LINE_COLOUR,
            linestyle="--",
            linewidth=0.6,
            zorder=1,
        )
        for model in design:
            model_points = points[points["simulated_model"] == model]
            ax.vlines(
                model_points["true_value"],
                model_points["eti95_lower"],
                model_points["eti95_upper"],
                color=MODEL_COLOURS[model],
                linewidth=0.5,
                alpha=0.4,
                zorder=2,
            )
            _scatter_converged(
                ax,
                model_points["true_value"],
                model_points["posterior_mean"],
                model_points["converged"],
                MODEL_COLOURS[model],
            )
        r = pooled.set_index("effect").loc[effect, "pooled_r"]
        ax.text(
            0.04,
            0.96,
            f"r = {r:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=TICK_FONT_SIZE,
        )
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_title(label, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("True value", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Posterior mean (95% interval)", fontsize=LABEL_FONT_SIZE)
        _despine(ax)
    add_panel_labels(fig, axes)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4,
            color=MODEL_COLOURS[model],
            label=_one_line_name(model),
        )
        for model in design
    ]
    if not source["converged"].all():
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=4,
                markerfacecolor="none",
                color="#262626",
                label="Fit did not converge",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=TICK_FONT_SIZE,
    )
    _save_figure(fig, figures_dir, "s3")
