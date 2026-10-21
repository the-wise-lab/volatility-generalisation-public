import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from volatility_generalisation.modelling_code.condition_metadata import (
    CONDITION_METADATA,
)

REPO_BOUNDS_FILE = Path("config/model_bounds.toml")
DUAL_LR = "dual_lr_single_value"
RESCORLA_WAGNER = "rescorla_wagner_single_value"


def _load_compile_results_module():
    script_path = Path("scripts/compile_results.py")
    spec = importlib.util.spec_from_file_location("compile_results", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_fit_folder(
    results_dir: Path,
    condition: str,
    model: str = DUAL_LR,
    n_subjects: int = 3,
    converged: bool = True,
    ppc_arrays: dict | None = None,
    **metadata_overrides,
) -> Path:
    """Write a small fit folder with all five result files.

    Args:
        results_dir: Folder that will hold the fit folder.
        condition: Condition name.
        model: Model name.
        n_subjects: Number of participants.
        converged: Whether the fit is recorded as converged.
        ppc_arrays: Arrays for ``ppc_results.npz``, instead of the defaults.
        **metadata_overrides: Entries that replace or add to ``metadata.json``.
            For example ``subject_ids=None`` or ``config_sha256=None``.

    Returns:
        The fit folder.
    """
    fit_dir = results_dir / f"{condition}_{model}"
    fit_dir.mkdir(parents=True)
    metadata = {
        "condition": condition,
        "model": model,
        "converged": converged,
        "accuracy": 0.75,
        "n_subjects": n_subjects,
        "n_trials": 3,
        "config_path": "config/model_bounds.toml",
        "config_sha256": "0" * 64,
        "subject_ids": [f"{condition}_s{i}" for i in range(n_subjects)],
    }
    metadata.update(metadata_overrides)
    summaries = {
        name: {
            "mean": 0.5,
            "std": 0.1,
            "median": 0.5,
            "min": 0.4,
            "max": 0.6,
            "subject_means": [0.4 + 0.05 * i for i in range(n_subjects)],
        }
        for name in ["learning_rate", "temperature"]
    }
    (fit_dir / "metadata.json").write_text(json.dumps(metadata))
    (fit_dir / "diagnostics.json").write_text(
        json.dumps({"converged": converged, "divergences": 0})
    )
    (fit_dir / "parameter_summaries.json").write_text(json.dumps(summaries))
    np.savez_compressed(
        fit_dir / "posterior_samples.npz", learning_rate=np.full((2, n_subjects), 0.5)
    )
    if ppc_arrays is None:
        all_probs = np.full((2, n_subjects, 3), 0.6)
        ppc_arrays = {
            "actual_choices": np.tile([1.0, 0.0, 1.0], (n_subjects, 1)),
            "all_simulated_probs": all_probs,
            "all_simulated_choices": np.ones((2, n_subjects, 3)),
            "simulated_probs": all_probs.mean(axis=0),
            "accuracy": np.array(0.75),
        }
    np.savez_compressed(fit_dir / "ppc_results.npz", **ppc_arrays)
    return fit_dir


def _make_fit_grid(results_dir: Path, models: list[str]) -> None:
    """Write fit folders for every condition and model, using the project bounds.

    Each model gets different participant estimates, so that one model's
    results cannot pass for another's.

    Args:
        results_dir: Folder that will hold the fit folders.
        models: Models to write fit folders for.
    """
    bounds_hash = hashlib.sha256(REPO_BOUNDS_FILE.read_bytes()).hexdigest()
    for model_index, model in enumerate(models):
        for condition_index, condition in enumerate(CONDITION_METADATA):
            fit_dir = _make_fit_folder(
                results_dir,
                condition,
                model=model,
                config_path=str(REPO_BOUNDS_FILE.resolve()),
                config_sha256=bounds_hash,
            )
            summaries_path = fit_dir / "parameter_summaries.json"
            summaries = json.loads(summaries_path.read_text())
            for name, values in summaries.items():
                values["subject_means"] = [
                    0.2 + 0.1 * model_index + 0.02 * condition_index + 0.05 * i
                    for i in range(len(values["subject_means"]))
                ]
            summaries_path.write_text(json.dumps(summaries))


def _write_bounds_file(
    path: Path, extra_comment: str = "", model: str = DUAL_LR
) -> str:
    """Write a small parameter bounds file.

    Args:
        path: File to write. Its folder is created if needed.
        extra_comment: Text added at the top of the file, to change its hash
            without changing the bounds.
        model: Model that gets a section in the file.

    Returns:
        The SHA-256 hash of the file's contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"{extra_comment}\n[common]\ntemperature = {{ min = 0.2, max = 0.6 }}\n"
        f"[{model}]\nlearning_rate_pos = {{ min = 0.0, max = 1.0 }}\n"
        "tau = { min = 0.0, max = 5.0 }\nbias = { fixed = 0.5 }\n"
    )
    path.write_text(text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truncate(path: Path) -> None:
    """Cut a file in half, so that it can no longer be read.

    Args:
        path: File to cut. It is rewritten with its first half only.

    Returns:
        None.
    """
    content = path.read_bytes()
    path.write_bytes(content[: len(content) // 2])


def _run_main(
    module, monkeypatch, tmp_path: Path, results_dir: Path, **options
) -> None:
    """Run ``main`` with the given command-line options.

    Args:
        module: The loaded ``compile_results`` module.
        monkeypatch: Pytest fixture used to replace ``parse_args``.
        tmp_path: Folder for the output and prepared-data folders.
        results_dir: Folder with the fit folders.
        **options: Command-line options that replace the defaults, for example
            ``analysis_model=["none"]`` or ``conditions=[...]``.

    Returns:
        None.
    """
    args = {
        "results_dir": str(results_dir),
        "output_dir": [str(tmp_path / "compiled")],
        "analysis_model": [DUAL_LR],
        "conditions": list(CONDITION_METADATA),
        "cross_parameter_correction": "none",
        "data_dir": str(tmp_path / "prepared"),
        "overwrite": False,
    }
    args.update(options)
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(**args))
    module.main()


def _second_block_subject_table(
    participants: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build a subject-level table for the four second-block conditions.

    Args:
        participants: Number of participants (up to 3) to keep for each
            condition. Conditions not listed keep all 3.

    Returns:
        Table with one row per participant, in the layout that
        ``export_subject_parameters`` produces.
    """
    rows = []
    cell_specs = [
        (
            "StableV4_2",
            "Stable",
            "Stable",
            [0.20, 0.22, 0.24],
            [0.15, 0.18, 0.20],
            [0.30, 0.32, 0.34],
        ),
        (
            "StableV1",
            "Stable",
            "Volatile",
            [0.35, 0.38, 0.40],
            [0.22, 0.24, 0.26],
            [0.36, 0.38, 0.40],
        ),
        (
            "VolatileV2",
            "Volatile",
            "Stable",
            [0.45, 0.48, 0.50],
            [0.30, 0.33, 0.35],
            [0.42, 0.44, 0.46],
        ),
        (
            "VolatileV3_2",
            "Volatile",
            "Volatile",
            [0.55, 0.58, 0.60],
            [0.40, 0.42, 0.45],
            [0.48, 0.50, 0.52],
        ),
    ]
    for (
        condition,
        block_type,
        previous_block_type,
        lr_pos,
        lr_neg,
        temperature,
    ) in cell_specs:
        n_kept = (participants or {}).get(condition, 3)
        for subject_idx, (pos_value, neg_value, temp_value) in enumerate(
            list(zip(lr_pos, lr_neg, temperature))[:n_kept]
        ):
            rows.append(
                {
                    "subject_id": f"{condition}_subj_{subject_idx}",
                    "subject_idx": subject_idx,
                    "condition": condition,
                    "model": DUAL_LR,
                    "block_type": block_type,
                    "version": condition.split("V")[-1],
                    "block_number": 2,
                    "previous_block_type": previous_block_type,
                    "converged": True,
                    "learning_rate_pos": pos_value,
                    "learning_rate_neg": neg_value,
                    "temperature": temp_value,
                }
            )
    return pd.DataFrame(rows)


def test_get_parameter_transform_uses_config_bounds() -> None:
    module = _load_compile_results_module()
    dual_lr_bounds = module.load_parameter_bounds(DUAL_LR, REPO_BOUNDS_FILE)
    leaky_beta_bounds = module.load_parameter_bounds(
        "leaky_beta_single_value", REPO_BOUNDS_FILE
    )

    assert (
        module._get_parameter_transform("learning_rate_pos", dual_lr_bounds) == "logit"
    )
    assert module._get_parameter_transform("temperature", dual_lr_bounds) == "logit"
    assert module._get_parameter_transform("tau", leaky_beta_bounds) == "identity"


def test_find_bounds_file_stops_when_bounds_file_cannot_be_read(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    recorded_file = tmp_path / "bounds.toml"
    recorded_file.write_text("temperature = \n")
    file_hash = hashlib.sha256(recorded_file.read_bytes()).hexdigest()
    _make_fit_folder(
        tmp_path / "results",
        "VolatileV1",
        config_path=str(recorded_file),
        config_sha256=file_hash,
    )
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    with pytest.raises(ValueError, match=f"bounds.toml.*'{DUAL_LR}'"):
        module.find_bounds_file(fits, tmp_path / "project")


def test_find_fit_folders_ignores_other_folders(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    _make_fit_folder(tmp_path, "StableV4_2")
    _make_fit_folder(tmp_path, "VolatileV1", model=RESCORLA_WAGNER)
    (tmp_path / "compiled").mkdir()
    (tmp_path / "VolatileV1_not_a_model").mkdir()

    fits = module.find_fit_folders(tmp_path, list(CONDITION_METADATA))

    assert [(fit.condition, fit.model) for fit in fits] == [
        ("StableV4_2", DUAL_LR),
        ("VolatileV1", RESCORLA_WAGNER),
    ]


def test_find_fit_folders_stops_on_empty_fit_folder(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    _make_fit_folder(tmp_path, "StableV1")
    (tmp_path / f"VolatileV1_{DUAL_LR}").mkdir()

    with pytest.raises(FileNotFoundError, match=f"VolatileV1_{DUAL_LR}.*metadata.json"):
        module.find_fit_folders(tmp_path, list(CONDITION_METADATA))


@pytest.mark.parametrize("file_name", ["diagnostics.json", "posterior_samples.npz"])
def test_find_fit_folders_stops_on_unreadable_file(
    tmp_path: Path, file_name: str
) -> None:
    module = _load_compile_results_module()
    fit_dir = _make_fit_folder(tmp_path, "VolatileV1")
    _truncate(fit_dir / file_name)

    with pytest.raises(ValueError, match=f"{file_name}.*VolatileV1_{DUAL_LR}"):
        module.find_fit_folders(tmp_path, list(CONDITION_METADATA))


def test_find_fit_folders_leaves_out_unlisted_conditions(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    _make_fit_folder(tmp_path, "VolatileV1")
    (tmp_path / f"StableV1_{DUAL_LR}").mkdir()  # empty, but not asked for

    fits = module.find_fit_folders(tmp_path, ["VolatileV1"])

    assert [fit.condition for fit in fits] == ["VolatileV1"]


def test_apply_cross_parameter_correction_bonferroni() -> None:
    module = _load_compile_results_module()

    overview = pd.DataFrame(
        {
            "parameter": ["a", "b", "c"],
            "current_block_p_raw": [0.01, 0.04, 0.20],
            "previous_block_p_raw": [0.03, 0.50, 0.80],
            "interaction_p_raw": [0.02, 0.10, 0.30],
        }
    )

    corrected = module._apply_cross_parameter_correction(overview, "bonferroni")

    assert corrected["omnibus_correction_method"].eq("bonferroni").all()
    assert corrected["current_block_p_corrected"].tolist() == pytest.approx([0.03, 0.12, 0.60])
    assert corrected["previous_block_p_corrected"].tolist() == pytest.approx([0.09, 1.0, 1.0])
    assert corrected["interaction_p_corrected"].tolist() == pytest.approx([0.06, 0.30, 0.90])


@pytest.mark.rendering
def test_run_parameter_analyses_writes_overview_and_parameter_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_compile_results_module()

    subject_df = _second_block_subject_table()
    overview = module.run_parameter_analyses(
        subject_df,
        tmp_path,
        DUAL_LR,
        module.load_parameter_bounds(DUAL_LR, REPO_BOUNDS_FILE),
        cross_parameter_correction="bonferroni",
    )

    assert overview is not None
    output = capsys.readouterr().out
    assert "M =" in output
    assert "SD =" in output
    assert "Volatile->Stable" in output
    assert "Stable->Stable" in output
    assert set(overview["parameter"]) == {"learning_rate_pos", "learning_rate_neg", "temperature"}
    assert overview["status"].eq("ok").all()
    assert "skip_reason" not in overview.columns
    assert overview["omnibus_correction_method"].eq("bonferroni").all()

    assert (tmp_path / "parameter_overview.csv").exists()
    for parameter_name in ["learning_rate_pos", "learning_rate_neg", "temperature"]:
        parameter_dir = tmp_path / "parameters" / parameter_name
        assert (parameter_dir / "anova_results.csv").exists()
        assert (parameter_dir / "posthoc_comparisons.csv").exists()
        assert (parameter_dir / "cell_summary.csv").exists()
        assert (parameter_dir / "shapiro_checks.csv").exists()
        assert (parameter_dir / "levene_check.csv").exists()
        assert (parameter_dir / "interaction_plot.png").exists()


def test_run_parameter_analyses_stops_on_small_cell(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    subject_df = _second_block_subject_table({"VolatileV2": 2})

    with pytest.raises(
        ValueError, match="learning_rate_pos.*fewer than 3.*Volatile/Stable"
    ):
        module.run_parameter_analyses(
            subject_df,
            tmp_path,
            DUAL_LR,
            module.load_parameter_bounds(DUAL_LR, REPO_BOUNDS_FILE),
        )


def test_run_parameter_analyses_stops_when_bayes_factor_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_compile_results_module()
    original_error = ValueError("no Bayes factor")

    def failing_bayes_factor(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(module.pg, "bayesfactor_ttest", failing_bayes_factor)

    with pytest.raises(RuntimeError, match="learning_rate_pos") as exc_info:
        module.run_parameter_analyses(
            _second_block_subject_table(),
            tmp_path,
            DUAL_LR,
            module.load_parameter_bounds(DUAL_LR, REPO_BOUNDS_FILE),
        )
    assert exc_info.value.__cause__ is original_error


@pytest.mark.parametrize("n_missing", [1, 12])
def test_run_parameter_analyses_stops_on_missing_participant_value(
    tmp_path: Path, n_missing: int
) -> None:
    module = _load_compile_results_module()
    subject_df = _second_block_subject_table()
    subject_df.loc[subject_df.index[:n_missing], "learning_rate_neg"] = np.nan

    with pytest.raises(
        ValueError,
        match=rf"'learning_rate_neg'.*no finite value for {n_missing} participant\(s\)",
    ):
        module.run_parameter_analyses(
            subject_df,
            tmp_path,
            DUAL_LR,
            module.load_parameter_bounds(DUAL_LR, REPO_BOUNDS_FILE),
        )
    assert not (tmp_path / "parameter_overview.csv").exists()


def test_run_parameter_analyses_stops_when_table_holds_other_models(
    tmp_path: Path,
) -> None:
    module = _load_compile_results_module()
    subject_df = _second_block_subject_table()
    subject_df.loc[subject_df.index[0], "model"] = RESCORLA_WAGNER

    with pytest.raises(ValueError, match=f"also holds other models: {RESCORLA_WAGNER}"):
        module.run_parameter_analyses(
            subject_df,
            tmp_path,
            DUAL_LR,
            module.load_parameter_bounds(DUAL_LR, REPO_BOUNDS_FILE),
        )


def test_existing_output_entries_reports_nonempty_directory(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    (tmp_path / "summary_table.csv").write_text("done\n", encoding="utf-8")
    (tmp_path / "parameters").mkdir()

    assert module._existing_output_entries(tmp_path) == ["parameters", "summary_table.csv"]


def test_export_combined_ppc_writes_tables_and_requests_figures(
    tmp_path: Path, monkeypatch
) -> None:
    """The CSV tables are written, and both figures are asked for with the right tables.

    The two plotting functions are replaced by stand-ins that only record what
    they are asked to draw, so that this check does not draw and save images.
    The real PNG writers are covered by the multi-model compilation test.
    """
    module = _load_compile_results_module()
    trialwise_requests = []
    scatter_requests = []
    monkeypatch.setattr(
        module,
        "_plot_combined_ppc_trialwise",
        lambda table, path, model_name: trialwise_requests.append(
            (table.copy(), path, model_name)
        ),
    )
    monkeypatch.setattr(
        module,
        "_plot_combined_ppc_scatter",
        lambda table, path, model_name: scatter_requests.append(
            (table.copy(), path, model_name)
        ),
    )
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "compiled"
    results_dir.mkdir()
    output_dir.mkdir()

    def write_ppc(condition: str, actual_choices: np.ndarray, all_probs: np.ndarray, all_choices: np.ndarray) -> None:
        _make_fit_folder(
            results_dir,
            condition,
            n_subjects=actual_choices.shape[0],
            ppc_arrays=dict(
                actual_choices=actual_choices,
                all_simulated_probs=all_probs,
                all_simulated_choices=all_choices,
                simulated_probs=all_probs.mean(axis=0),
                simulated_choices=(all_probs.mean(axis=0) >= 0.5).astype(float),
                accuracy=np.array(0.75),
                trial_accuracy=np.array([0.5, 1.0, 0.5]),
            ),
        )

    write_ppc(
        "VolatileV1",
        actual_choices=np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        all_probs=np.array(
            [
                [[0.8, 0.3, 0.7], [0.9, 0.6, 0.4]],
                [[0.7, 0.4, 0.8], [0.8, 0.5, 0.3]],
            ]
        ),
        all_choices=np.array(
            [
                [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            ]
        ),
    )
    write_ppc(
        "StableV1",
        actual_choices=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        all_probs=np.array(
            [
                [[0.2, 0.7, 0.4], [0.3, 0.8, 0.6]],
                [[0.1, 0.6, 0.5], [0.2, 0.7, 0.7]],
            ]
        ),
        all_choices=np.array(
            [
                [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
            ]
        ),
    )

    fits = module.find_fit_folders(results_dir, ["VolatileV1", "StableV1"])
    result = module.export_combined_ppc(fits, output_dir)

    assert result is not None
    trialwise_df, summary_df = result
    assert len(trialwise_df) == 6
    assert len(summary_df) == 2

    volatile_trial_0 = trialwise_df[
        (trialwise_df["condition"] == "VolatileV1")
        & (trialwise_df["trial"] == 0)
    ].iloc[0]
    assert volatile_trial_0["block_number"] == 1
    assert volatile_trial_0["observed_choice_prob"] == pytest.approx(1.0)
    assert volatile_trial_0["predicted_choice_prob"] == pytest.approx(0.8)

    stable_summary = summary_df[summary_df["condition"] == "StableV1"].iloc[0]
    assert stable_summary["block_number"] == 2
    assert stable_summary["previous_block_type"] == "Volatile"
    assert stable_summary["predicted_choice_prob_mean"] == pytest.approx(0.4833333333)

    assert (output_dir / "posterior_predictive_trialwise.csv").exists()
    assert (output_dir / "posterior_predictive_summary.csv").exists()

    # One figure of each kind was asked for, for the one model, at the expected
    # path. Nothing is drawn here, so no image files should exist.
    assert len(trialwise_requests) == 1
    assert len(scatter_requests) == 1
    trialwise_table, trialwise_path, trialwise_model = trialwise_requests[0]
    scatter_table, scatter_path, scatter_model = scatter_requests[0]
    assert trialwise_path == output_dir / "posterior_predictive_trialwise.png"
    assert scatter_path == output_dir / "posterior_predictive_scatter.png"
    assert trialwise_model == scatter_model == DUAL_LR
    pd.testing.assert_frame_equal(trialwise_table, scatter_table)
    assert not list(output_dir.glob("*.png"))

    # The table handed to the plots holds the per-trial values worked out by
    # hand: observed = mean choice over participants; predicted = mean of the
    # simulated choice probabilities over draws and participants. Rows are
    # ordered by block, so VolatileV1 (block 1) comes before StableV1 (block 2).
    assert trialwise_table["condition"].tolist() == ["VolatileV1"] * 3 + ["StableV1"] * 3
    assert trialwise_table["trial"].tolist() == [0, 1, 2, 0, 1, 2]
    assert trialwise_table["observed_choice_prob"].tolist() == pytest.approx(
        [1.0, 0.5, 0.5, 0.0, 1.0, 0.5]
    )
    assert trialwise_table["predicted_choice_prob"].tolist() == pytest.approx(
        [0.80, 0.45, 0.55, 0.20, 0.70, 0.55]
    )


def test_main_requires_overwrite_for_nonempty_output_dir(tmp_path: Path, monkeypatch) -> None:
    module = _load_compile_results_module()
    output_dir = tmp_path / "compiled"
    output_dir.mkdir()
    (output_dir / "summary_table.csv").write_text("done\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "results_dir": str(tmp_path / "results"),
                "output_dir": [str(output_dir)],
                "analysis_model": ["rescorla_wagner_single_value"],
                "conditions": list(CONDITION_METADATA),
                "cross_parameter_correction": "none",
                "data_dir": str(tmp_path / "prepared"),
                "overwrite": False,
            },
        )(),
    )

    with pytest.raises(RuntimeError, match="Existing compiled outputs found"):
        module.main()


def test_main_stops_when_analysis_model_is_missing_in_a_condition(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_compile_results_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for condition in CONDITION_METADATA:
        if condition != "VolatileV3_2":
            _make_fit_folder(results_dir, condition)

    with pytest.raises(
        ValueError, match="no fit folder in these conditions: VolatileV3_2"
    ):
        _run_main(module, monkeypatch, tmp_path, results_dir)


def test_main_stops_when_conditions_leave_out_second_block(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_compile_results_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    with pytest.raises(ValueError, match="second-block.*StableV1"):
        _run_main(
            module,
            monkeypatch,
            tmp_path,
            results_dir,
            conditions=["VolatileV1", "StableV2"],
        )


def test_main_stops_when_analysis_model_has_no_fits(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_compile_results_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_fit_folder(results_dir, "VolatileV1", model=RESCORLA_WAGNER)

    with pytest.raises(
        ValueError, match=f"'{DUAL_LR}' has no fit folders.*{RESCORLA_WAGNER}"
    ):
        _run_main(module, monkeypatch, tmp_path, results_dir)


@pytest.mark.rendering
def test_main_with_no_analysis_model_compiles_every_model_including_unconverged(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_compile_results_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for model in [DUAL_LR, RESCORLA_WAGNER]:
        for condition in ["VolatileV1", "StableV1"]:
            _make_fit_folder(
                results_dir,
                condition,
                model=model,
                converged=not (model == DUAL_LR and condition == "StableV1"),
            )

    _run_main(
        module,
        monkeypatch,
        tmp_path,
        results_dir,
        analysis_model=["none"],
        conditions=["VolatileV1", "StableV1"],
    )

    output_dir = tmp_path / "compiled"
    assert "Second-block analyses not run" in capsys.readouterr().out
    assert not (output_dir / "parameter_overview.csv").exists()
    summary = pd.read_csv(output_dir / "summary_table.csv")
    assert len(summary) == 4
    assert summary.loc[
        ~summary["converged"], ["model", "condition"]
    ].values.tolist() == [[DUAL_LR, "StableV1"]]
    subject_table = pd.read_csv(output_dir / "subject_parameters.csv")
    assert set(subject_table["model"]) == {DUAL_LR, RESCORLA_WAGNER}
    assert len(subject_table) == 12
    assert not subject_table.loc[
        (subject_table["model"] == DUAL_LR)
        & (subject_table["condition"] == "StableV1"),
        "converged",
    ].any()
    assert len(pd.read_csv(output_dir / "posterior_predictive_summary.csv")) == 4
    for model in [DUAL_LR, RESCORLA_WAGNER]:
        assert (output_dir / f"posterior_predictive_trialwise_{model}.png").exists()
        assert (output_dir / f"posterior_predictive_scatter_{model}.png").exists()


def test_main_compiles_each_analysis_model_into_its_own_folder(
    tmp_path: Path, monkeypatch
) -> None:
    """One run with two models writes the same tables as one run per model.

    Args:
        tmp_path: Temporary folder from pytest.
        monkeypatch: Pytest fixture used to replace ``parse_args``.
    """
    module = _load_compile_results_module()
    results_dir = tmp_path / "results"
    _make_fit_grid(results_dir, [RESCORLA_WAGNER, DUAL_LR])
    together = [tmp_path / "together_rw", tmp_path / "together_dual"]
    alone = [tmp_path / "alone_rw", tmp_path / "alone_dual"]

    _run_main(
        module,
        monkeypatch,
        tmp_path,
        results_dir,
        analysis_model=[RESCORLA_WAGNER, DUAL_LR],
        output_dir=[str(folder) for folder in together],
    )
    for model, folder in zip([RESCORLA_WAGNER, DUAL_LR], alone):
        _run_main(
            module,
            monkeypatch,
            tmp_path,
            results_dir,
            analysis_model=[model],
            output_dir=[str(folder)],
        )

    for model, together_dir, alone_dir in zip(
        [RESCORLA_WAGNER, DUAL_LR], together, alone
    ):
        subject_table = pd.read_csv(together_dir / "subject_parameters.csv")
        assert set(subject_table["model"]) == {model}
        assert (together_dir / "parameter_overview.csv").exists()
        tables = sorted(
            path.relative_to(alone_dir) for path in alone_dir.rglob("*.csv")
        )
        assert tables == sorted(
            path.relative_to(together_dir) for path in together_dir.rglob("*.csv")
        )
        for table in tables:
            assert (together_dir / table).read_bytes() == (
                alone_dir / table
            ).read_bytes()


@pytest.mark.parametrize(
    "analysis_model, output_dir, message",
    [
        ([DUAL_LR], ["compiled", "compiled_dual_lr"], "one --output-dir for each"),
        ([RESCORLA_WAGNER, DUAL_LR], ["compiled", "./compiled"], "folders repeat"),
    ],
)
def test_main_stops_when_models_and_folders_do_not_pair_up(
    tmp_path: Path,
    monkeypatch,
    analysis_model: list[str],
    output_dir: list[str],
    message: str,
) -> None:
    """Each analysis model needs exactly one folder of its own.

    Args:
        tmp_path: Temporary folder from pytest.
        monkeypatch: Pytest fixture used to replace ``parse_args``.
        analysis_model: Models given to ``--analysis-model``.
        output_dir: Folders given to ``--output-dir``.
        message: Text the error must contain.
    """
    module = _load_compile_results_module()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=message):
        _run_main(
            module,
            monkeypatch,
            tmp_path,
            tmp_path / "results",
            analysis_model=analysis_model,
            output_dir=output_dir,
        )
    assert not (tmp_path / "compiled").exists()


def test_find_bounds_file_uses_recorded_path_with_matching_hash(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    recorded_file = tmp_path / "elsewhere" / "bounds.toml"
    file_hash = _write_bounds_file(recorded_file)
    _make_fit_folder(
        tmp_path / "results",
        "VolatileV1",
        config_path=str(recorded_file),
        config_sha256=file_hash,
    )
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    bounds = module.find_bounds_file(fits, project_root=tmp_path / "project")

    assert bounds == module.load_parameter_bounds(DUAL_LR, recorded_file)
    assert module._get_parameter_transform("learning_rate_pos", bounds) == "logit"
    assert module._get_parameter_transform("tau", bounds) == "identity"
    assert module._get_parameter_transform("bias", bounds) == "identity"


def test_find_bounds_file_uses_project_file_when_recorded_path_is_missing(
    tmp_path: Path,
) -> None:
    module = _load_compile_results_module()
    project_root = tmp_path / "project"
    file_hash = _write_bounds_file(project_root / "config" / "model_bounds.toml")
    _make_fit_folder(
        tmp_path / "results",
        "VolatileV1",
        config_path="/hpc/project/config/model_bounds.toml",
        config_sha256=file_hash,
    )
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    bounds = module.find_bounds_file(fits, project_root)

    assert bounds == module.load_parameter_bounds(
        DUAL_LR, project_root / "config" / "model_bounds.toml"
    )


def test_find_bounds_file_stops_when_no_file_matches_the_hash(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    project_root = tmp_path / "project"
    _write_bounds_file(
        project_root / "config" / "model_bounds.toml", extra_comment="# edited later"
    )
    recorded_hash = "a" * 64
    _make_fit_folder(
        tmp_path / "results",
        "VolatileV1",
        config_path="/hpc/project/config/model_bounds.toml",
        config_sha256=recorded_hash,
    )
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    with pytest.raises(
        ValueError,
        match=f"VolatileV1_{DUAL_LR}.*{recorded_hash}.*changed after the fit",
    ):
        module.find_bounds_file(fits, project_root)


def test_find_bounds_file_stops_when_fit_without_hash_has_missing_path(
    tmp_path: Path,
) -> None:
    module = _load_compile_results_module()
    project_root = tmp_path / "project"
    _write_bounds_file(project_root / "config" / "model_bounds.toml")
    _make_fit_folder(
        tmp_path / "results",
        "VolatileV1",
        config_path="/hpc/project/config/model_bounds.toml",
        config_sha256=None,
    )
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    with pytest.raises(
        FileNotFoundError, match=f"VolatileV1_{DUAL_LR}.*/hpc/project/config"
    ):
        module.find_bounds_file(fits, project_root)


def test_find_bounds_file_stops_when_model_has_no_section(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    recorded_file = tmp_path / "bounds.toml"
    file_hash = _write_bounds_file(recorded_file, model=RESCORLA_WAGNER)
    _make_fit_folder(
        tmp_path / "results",
        "VolatileV1",
        config_path=str(recorded_file),
        config_sha256=file_hash,
    )
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    with pytest.raises(
        ValueError, match=f"bounds.toml.*no usable bounds for model '{DUAL_LR}'"
    ) as exc_info:
        module.find_bounds_file(fits, tmp_path / "project")
    assert exc_info.value.__cause__ is not None


def test_find_bounds_file_stops_when_only_some_fits_have_a_hash(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    recorded_file = tmp_path / "bounds.toml"
    file_hash = _write_bounds_file(recorded_file)
    _make_fit_folder(
        tmp_path / "results",
        "VolatileV1",
        config_path=str(recorded_file),
        config_sha256=file_hash,
    )
    _make_fit_folder(
        tmp_path / "results",
        "StableV1",
        config_path=str(recorded_file),
        config_sha256=None,
    )
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1", "StableV1"])

    with pytest.raises(
        ValueError, match="do not all record the same parameter bounds file hash"
    ):
        module.find_bounds_file(fits, tmp_path / "project")


def test_subject_parameter_table_stops_when_prepared_file_is_missing(
    tmp_path: Path,
) -> None:
    module = _load_compile_results_module()
    _make_fit_folder(tmp_path / "results", "VolatileV1", subject_ids=None)
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    with pytest.raises(FileNotFoundError, match="VolatileV1.npz"):
        module.build_subject_parameter_table(fits, tmp_path / "prepared")


def test_subject_parameter_table_stops_on_zero_participants(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    _make_fit_folder(tmp_path / "results", "VolatileV1", n_subjects=0)
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    with pytest.raises(ValueError, match=f"VolatileV1_{DUAL_LR}.*0 participants"):
        module.build_subject_parameter_table(fits, tmp_path / "prepared")


@pytest.mark.parametrize("bad_value", [None, float("nan")])
def test_subject_parameter_table_stops_on_missing_participant_mean(
    tmp_path: Path, bad_value: float | None
) -> None:
    module = _load_compile_results_module()
    fit_dir = _make_fit_folder(tmp_path / "results", "VolatileV1")
    summaries_path = fit_dir / "parameter_summaries.json"
    summaries = json.loads(summaries_path.read_text())
    summaries["temperature"]["subject_means"][1] = bad_value
    summaries_path.write_text(json.dumps(summaries))
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])

    with pytest.raises(
        ValueError,
        match=r"VolatileV1_.*'temperature'.*VolatileV1_s1 \(subject_idx 1\)",
    ):
        module.build_subject_parameter_table(fits, tmp_path / "prepared")


def test_main_writes_nothing_when_a_participant_mean_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad fit of the second analysis model stops the run before any writing.

    Args:
        tmp_path: Temporary folder from pytest.
        monkeypatch: Pytest fixture used to replace ``parse_args``.
    """
    module = _load_compile_results_module()
    results_dir = tmp_path / "results"
    _make_fit_grid(results_dir, [RESCORLA_WAGNER, DUAL_LR])
    summaries_path = results_dir / f"VolatileV1_{DUAL_LR}" / "parameter_summaries.json"
    summaries = json.loads(summaries_path.read_text())
    summaries["temperature"]["subject_means"][0] = None
    summaries_path.write_text(json.dumps(summaries))
    output_dirs = [tmp_path / "compiled", tmp_path / "compiled_dual_lr"]

    with pytest.raises(ValueError, match="no finite posterior mean"):
        _run_main(
            module,
            monkeypatch,
            tmp_path,
            results_dir,
            analysis_model=[RESCORLA_WAGNER, DUAL_LR],
            output_dir=[str(folder) for folder in output_dirs],
        )
    for folder in output_dirs:
        assert not folder.exists()


def test_export_combined_ppc_stops_on_unreadable_file(tmp_path: Path) -> None:
    module = _load_compile_results_module()
    fit_dir = _make_fit_folder(tmp_path / "results", "VolatileV1")
    fits = module.find_fit_folders(tmp_path / "results", ["VolatileV1"])
    _truncate(fit_dir / "ppc_results.npz")

    with pytest.raises(ValueError, match=f"ppc_results.npz.*VolatileV1_{DUAL_LR}"):
        module.export_combined_ppc(fits, tmp_path)
