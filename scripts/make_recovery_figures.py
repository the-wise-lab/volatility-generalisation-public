"""Recovery Figures Script

Draws the three supplementary recovery figures (S1 to S3) from the tables that
the recovery workflows have already saved. It fits nothing. It first checks that
every replicate, model and parameter is present, and stops at the
first input that is missing. Fits that did not converge are drawn and marked.
Run with ``--help`` for usage.

Usage::

    uv run python scripts/make_recovery_figures.py --replicates 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from volatility_generalisation.figures.publication_figures import ensure_dm_sans_font
from volatility_generalisation.figures.recovery_figures import (
    FIGURE_FILE_NAMES,
    load_group_effect_recovery,
    load_model_recovery_winners,
    load_parameter_design,
    load_participant_recovery,
    load_recovery_models,
    make_group_effect_figure,
    make_model_recovery_figure,
    make_participant_recovery_figure,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Read the command-line options of the recovery figures script.

    Args:
        argv: Command-line arguments to read. If ``None``, the arguments the
            script was started with are used.

    Returns:
        The parsed options ``config``, ``integrated_dir``, ``replicates`` and
        ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Draw the supplementary recovery figures from saved recovery tables.",
    )
    parser.add_argument(
        "--config",
        default="scripts/config.toml",
        help="Settings file that lists the models (default: %(default)s).",
    )
    parser.add_argument(
        "--integrated-dir",
        default="results/integrated_model_recovery/summary",
        help="Compiled integrated recovery results, the folder that holds tables/ "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        required=True,
        help="Number of integrated recovery replicates that must be present "
        "(replicates 0 to N-1).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/recovery_figures",
        help="Folder for figures/ and tables/ (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    """Draw figures S1 to S3 and save the tables behind them.

    All inputs are read and checked before anything is drawn. Each figure is
    saved as PDF, SVG and PNG, and the values it plots are saved as a table.

    Args:
        args: Parsed command-line options from ``parse_args``.

    Raises:
        ValueError: If ``--replicates`` is less than 1, or a table is incomplete.
        FileNotFoundError: If a table is missing.
    """
    if args.replicates < 1:
        raise ValueError(f"--replicates must be at least 1, got {args.replicates}")
    ensure_dm_sans_font()

    models = load_recovery_models(args.config)
    design = load_parameter_design(models)
    integrated_dir = Path(args.integrated_dir)

    # Read and check every input first, so that no figure is drawn from partial
    # results.
    winners = load_model_recovery_winners(integrated_dir, models, args.replicates)
    participant_rows, participant_summary = load_participant_recovery(
        integrated_dir, design, args.replicates
    )
    group_effects = load_group_effect_recovery(integrated_dir, design, args.replicates)

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    make_model_recovery_figure(winners, models, figures_dir, tables_dir)
    make_participant_recovery_figure(
        participant_rows, participant_summary, design, figures_dir, tables_dir
    )
    make_group_effect_figure(group_effects, design, figures_dir, tables_dir)
    for name in FIGURE_FILE_NAMES.values():
        print(f"  [figure] {name}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main(parse_args())
