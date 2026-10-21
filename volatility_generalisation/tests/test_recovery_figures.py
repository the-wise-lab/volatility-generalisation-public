"""Tests for the supplementary recovery figures script.

The inputs are small synthetic recovery tables: two models, two replicates and
six simulated participants per fit. Nothing is fitted.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volatility_generalisation.figures.recovery_figures import (
    PARAMETER_LABELS,
    PLOTTED_EFFECT_LABELS,
    _parameter_title,
    load_parameter_design,
    use_dm_sans_for_symbols,
)
from volatility_generalisation.figures.publication_figures import ensure_dm_sans_font

RW = "rescorla_wagner_single_value"
DUAL = "dual_lr_single_value"
MODELS = [RW, DUAL]
REPLICATES = 2
N_SUBJECTS = 6
ALL_EFFECTS = [*PLOTTED_EFFECT_LABELS, "subject_sigma"]


def _load_script():
    """Import the figures script, which is not part of the package.

    Returns:
        The imported ``make_recovery_figures`` module.
    """
    path = Path(__file__).resolve().parents[2] / "scripts" / "make_recovery_figures.py"
    spec = importlib.util.spec_from_file_location("make_recovery_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write synthetic recovery tables for every figure.

    The fits of the Rescorla-Wagner model in replicate 1 did not converge, and
    in replicate 1 the dual learning-rate model wins the Rescorla-Wagner data
    set. The correlations in
    the summary tables are not the correlations of the rows beside them, so a
    figure that recomputes them would show different values.

    Args:
        tmp_path: Folder to write in.

    Returns:
        Paths of the settings file and the integrated results folder.
    """
    rng = np.random.default_rng(0)
    design = load_parameter_design(MODELS)
    config = tmp_path / "config.toml"
    config.write_text(
        f"[fitting]\nmodels = {json.dumps(MODELS)}\n",
        encoding="utf-8",
    )

    tables = tmp_path / "integrated" / "tables"
    tables.mkdir(parents=True)
    winners, rows, summary, effects = [], [], [], []
    for replicate in range(REPLICATES):
        for model in MODELS:
            done = not (model == RW and replicate == 1)
            winner = DUAL if (model == RW and replicate == 1) else model
            winners.append(
                {
                    "replicate": replicate,
                    "simulated_model": model,
                    "fitted_model": winner,
                    "converged": done,
                }
            )
            for index, (parameter, (low, high)) in enumerate(design[model].items()):
                key = {
                    "replicate": replicate,
                    "simulated_model": model,
                    "parameter": parameter,
                }
                true = rng.uniform(low, high, N_SUBJECTS)
                recovered = true + rng.normal(0, 0.05 * (high - low), N_SUBJECTS)
                for subject in range(N_SUBJECTS):
                    rows.append(
                        {
                            **key,
                            "subject_index": subject,
                            "true_value": true[subject],
                            "recovered_mean": recovered[subject],
                            "converged": done,
                        }
                    )
                summary.append(
                    {
                        **key,
                        "pearson_r": 0.9 - 0.01 * (replicate + index),
                        "n_subjects": N_SUBJECTS,
                        "converged": done,
                    }
                )
                for effect in ALL_EFFECTS:
                    mean = rng.normal()
                    effects.append(
                        {
                            **key,
                            "effect": effect,
                            "true_value": mean + rng.normal(0, 0.1),
                            "posterior_mean": mean,
                            "eti95_lower": mean - 0.3,
                            "eti95_upper": mean + 0.3,
                            "converged": done,
                        }
                    )
    for name, content in [
        ("model_recovery_winners", winners),
        ("parameter_recovery", rows),
        ("parameter_recovery_summary", summary),
        ("group_effect_recovery", effects),
    ]:
        pd.DataFrame(content).to_csv(tables / f"{name}.csv", index=False)

    return {"config": config, "integrated": tmp_path / "integrated"}


def _run(inputs: dict[str, Path], output_dir: Path) -> None:
    """Run the figures script on the synthetic inputs.

    Args:
        inputs: Paths from ``_write_inputs``.
        output_dir: Folder for the figures and tables.
    """
    script = _load_script()
    args = script.parse_args(
        [
            "--config",
            str(inputs["config"]),
            "--integrated-dir",
            str(inputs["integrated"]),
            "--replicates",
            str(REPLICATES),
            "--output-dir",
            str(output_dir),
        ]
    )
    script.main(args)


@pytest.mark.rendering
def test_script_draws_every_figure_from_the_tables_and_keeps_unconverged_fits(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    output_dir = tmp_path / "output"
    _run(inputs, output_dir)

    names = [
        "figure-s1-model-recovery",
        "figure-s2-participant-parameter-recovery",
        "figure-s3-group-effect-recovery",
    ]
    for name in names:
        for extension in ("pdf", "svg", "png"):
            assert (output_dir / "figures" / f"{name}.{extension}").is_file()

    tables = output_dir / "tables"
    saved = inputs["integrated"] / "tables"

    # S1: the winners were Rescorla-Wagner, dual, dual, dual for simulated
    # Rescorla-Wagner replicates 0 and 1 and dual-learning-rate replicates 0 and 1.
    s1 = pd.read_csv(tables / "figure-s1-source.csv").set_index(
        ["simulated_model", "recovered_model"]
    )
    assert s1.loc[(RW, RW), "confusion_share"] == 0.5
    assert s1.loc[(RW, DUAL), "confusion_share"] == 0.5
    assert s1.loc[(DUAL, DUAL), "confusion_share"] == 1.0
    assert s1.loc[(DUAL, DUAL), "inversion_share"] == pytest.approx(2 / 3)
    assert s1.loc[(RW, DUAL), "n_winners_not_converged"] == 1
    assert s1["n_winners_not_converged"].sum() == 1

    # S2: the points are the saved participant rows, and the correlation is the
    # median of the saved summary correlations, not recomputed from the points.
    rows = pd.read_csv(saved / "parameter_recovery.csv")
    key = ["replicate", "simulated_model", "parameter", "subject_index"]
    expected_rows = rows.sort_values(key).reset_index(drop=True)
    s2 = pd.read_csv(tables / "figure-s2-source.csv")
    pd.testing.assert_frame_equal(s2, expected_rows[s2.columns])
    summary = pd.read_csv(saved / "parameter_recovery_summary.csv")
    panel = summary[
        (summary["simulated_model"] == RW) & (summary["parameter"] == "learning_rate")
    ]
    r_stats = pd.read_csv(tables / "figure-s2-correlations.csv").set_index(
        ["simulated_model", "parameter"]
    )
    assert r_stats.loc[(RW, "learning_rate"), "median_r"] == pytest.approx(
        np.median(panel["pearson_r"])
    )
    assert r_stats.loc[(RW, "learning_rate"), "min_r"] == pytest.approx(
        panel["pearson_r"].min()
    )
    assert r_stats.loc[(RW, "learning_rate"), "max_r"] == pytest.approx(
        panel["pearson_r"].max()
    )
    # Fits that did not converge stay in the table, with their flag.
    assert len(s2) == len(rows)
    assert int((~s2["converged"]).sum()) == int((~rows["converged"]).sum()) > 0

    # S3: the four drawn effects, as saved. The spread between participants is
    # not drawn.
    effects = pd.read_csv(saved / "group_effect_recovery.csv")
    drawn = effects[effects["effect"].isin(PLOTTED_EFFECT_LABELS)]
    s3 = pd.read_csv(tables / "figure-s3-source.csv")
    assert len(s3) == len(drawn)
    assert set(s3["effect"]) == set(PLOTTED_EFFECT_LABELS)
    key = ["replicate", "simulated_model", "parameter", "effect"]
    pd.testing.assert_frame_equal(
        s3, drawn.sort_values(key).reset_index(drop=True)[s3.columns]
    )
    assert int((~s3["converged"]).sum()) > 0
    pooled = pd.read_csv(tables / "figure-s3-correlations.csv").set_index("effect")
    mu = drawn[drawn["effect"] == "mu"]
    assert pooled.loc["mu", "pooled_r"] == pytest.approx(
        np.corrcoef(mu["true_value"], mu["posterior_mean"])[0, 1]
    )


def test_script_stops_before_drawing_if_a_model_recovery_winner_is_missing(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    path = inputs["integrated"] / "tables" / "model_recovery_winners.csv"
    winners = pd.read_csv(path)
    drop = (winners["simulated_model"] == DUAL) & (winners["replicate"] == 1)
    winners[~drop].to_csv(path, index=False)

    with pytest.raises(ValueError, match="incomplete"):
        _run(inputs, tmp_path / "output")
    assert not (tmp_path / "output" / "figures").exists()


@pytest.mark.rendering
def test_every_parameter_label_draws_without_missing_glyphs(caplog) -> None:
    import matplotlib.pyplot as plt

    ensure_dm_sans_font()
    with plt.rc_context():
        use_dm_sans_for_symbols()
        fig = plt.figure()
        for index, parameter in enumerate(PARAMETER_LABELS):
            fig.text(0.1, 0.05 + 0.09 * index, _parameter_title(parameter))
        caplog.set_level(logging.WARNING)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.canvas.draw()
        plt.close(fig)

    messages = [str(w.message) for w in caught]
    messages += [record.getMessage() for record in caplog.records]
    assert not [m for m in messages if "glyph" in m.lower()], messages
