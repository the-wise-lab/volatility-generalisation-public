"""Targeted tests for scripts/regression_analysis.py.

Covers the error cases (unknown parameter, missing input file, missing
column, zero model rows), the questionnaire-archive member check, the check
that one parameter's run leaves another parameter's output folder untouched,
and the list of figures that a run writes. It also covers the reported
factor-count rules (eigenvalues, parallel analysis, CNG), matching of factors
by Tucker congruence and the stop on an impossible factor solution, on
simulated data with a known structure; the write-then-compare rule of the
shared checks folder; the files a real run writes into that folder; the
deletion of a stale summary_df.csv; and the loadings plot for one factor. A
small invented questionnaire archive covers the scoring key, the
attention-check matching and the repeated-answer stop. Small invented
learning-rate frames cover the change score (fits that did not converge are
kept and counted; a missing or repeated block stops the run) and the
Benjamini-Hochberg correction over every factor and version together.
"""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.multitest import multipletests


def _load_module():
    """Import scripts/regression_analysis.py as a fresh module object.

    Returns:
        The freshly executed module, following the same dynamic-import
        pattern used by the other script-testing modules in this test
        suite (see test_outcome_coding.py, test_compile_results.py).
    """
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "regression_analysis.py"
    spec = importlib.util.spec_from_file_location("regression_analysis", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_lr_csv(path: Path, rows: list[dict]) -> Path:
    """Write a minimal subject-parameters CSV for a `load_lr_data` test.

    Args:
        path: Destination CSV path.
        rows: One dict per row; keys become columns.

    Returns:
        `path`, for convenient chaining at the call site.
    """
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_zip(path: Path, members: dict[str, str]) -> Path:
    """Write a zip archive with the given text members, for an archive test.

    Args:
        path: Destination zip path.
        members: Mapping of archive member name to its text content.

    Returns:
        `path`, for convenient chaining at the call site.
    """
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


# ── Parameter mapping and error exits ────────────────────────────────────────


def test_parameter_configs_match_contract():
    module = _load_module()

    assert module.PARAMETER_CONFIGS["learning_rate"] == {
        "lr_csv": Path("results/compiled/subject_parameters.csv"),
        "model": "rescorla_wagner_single_value",
        "lr_column": "learning_rate",
    }
    dual_csv = Path("results/compiled_dual_lr/subject_parameters.csv")
    assert module.PARAMETER_CONFIGS["learning_rate_neg"] == {
        "lr_csv": dual_csv,
        "model": "dual_lr_single_value",
        "lr_column": "learning_rate_neg",
    }
    assert module.PARAMETER_CONFIGS["learning_rate_pos"] == {
        "lr_csv": dual_csv,
        "model": "dual_lr_single_value",
        "lr_column": "learning_rate_pos",
    }


def test_output_dir_defaults_to_results_regression():
    module = _load_module()
    args = module.parse_args(["--parameter", "learning_rate"])
    assert args.output_dir == "results/regression"
    assert args.factor_checks_dir == "results/factor_analysis_checks"


def test_unknown_parameter_exits_non_zero():
    module = _load_module()
    with pytest.raises(SystemExit) as exc_info:
        module.parse_args(["--parameter", "not_a_real_parameter"])
    assert exc_info.value.code != 0


def test_missing_lr_csv_raises_and_names_file(tmp_path: Path):
    module = _load_module()
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        module.load_lr_data(missing, "rescorla_wagner_single_value", "learning_rate")


def test_missing_lr_column_raises_and_names_column(tmp_path: Path):
    module = _load_module()
    csv_path = _write_lr_csv(
        tmp_path / "params.csv",
        [{"subject_id": "s1", "model": "rescorla_wagner_single_value"}],
    )
    with pytest.raises(ValueError, match="learning_rate"):
        module.load_lr_data(csv_path, "rescorla_wagner_single_value", "learning_rate")


def test_missing_model_column_raises_and_names_it(tmp_path: Path):
    module = _load_module()
    csv_path = _write_lr_csv(
        tmp_path / "params.csv",
        [{"subject_id": "s1", "learning_rate": 0.5}],
    )
    with pytest.raises(ValueError, match="'model'"):
        module.load_lr_data(csv_path, "rescorla_wagner_single_value", "learning_rate")


def test_zero_rows_for_model_raises_and_names_model(tmp_path: Path):
    module = _load_module()
    csv_path = _write_lr_csv(
        tmp_path / "params.csv",
        [{"subject_id": "s1", "model": "some_other_model", "learning_rate": 0.5}],
    )
    with pytest.raises(ValueError, match="rescorla_wagner_single_value"):
        module.load_lr_data(csv_path, "rescorla_wagner_single_value", "learning_rate")


def test_valid_lr_csv_loads_successfully(tmp_path: Path):
    module = _load_module()
    csv_path = _write_lr_csv(
        tmp_path / "params.csv",
        [
            {
                "subject_id": "s1",
                "model": "rescorla_wagner_single_value",
                "converged": True,
                "learning_rate": 0.5,
            }
        ],
    )
    lr_data = module.load_lr_data(csv_path, "rescorla_wagner_single_value", "learning_rate")
    assert list(lr_data["subject_id"]) == ["s1"]


# ── Learning-rate change and the multiple-comparison correction ─────────────


def _lr_blocks(rows: list[tuple[str, int, float, bool]]) -> pd.DataFrame:
    """Build a small per-block logit learning-rate frame for `build_lr_pivot`.

    Args:
        rows: One ``(subject_id, block_number, lr_logit, converged)`` tuple
            per block. Every participant does version V1.

    Returns:
        A frame with ``subject_id``, ``version``, ``block_number``,
        ``lr_logit`` and a True/False ``converged`` column.
    """
    frame = pd.DataFrame(
        rows, columns=["subject_id", "block_number", "lr_logit", "converged"]
    )
    frame.insert(1, "version", "V1")
    return frame


def test_lr_pivot_keeps_and_flags_nonconverged_fits_and_stops_on_bad_blocks():
    """Non-converged fits are kept and flagged; a missing or repeated block stops.

    No other test builds the change score. This catches a participant whose
    fit did not converge being dropped or given the wrong flag, and a missing
    or repeated block being skipped or resolved by picking one row.
    """
    module = _load_module()
    blocks = [
        ("p1", 1, -1.0, True),
        ("p1", 2, 0.5, True),
        ("p2", 1, 0.0, True),
        ("p2", 2, -0.5, False),
    ]

    pivot = module.build_lr_pivot(_lr_blocks(blocks))

    # p2's block 2 did not converge, but p2 is still in the frame, flagged.
    assert list(pivot["subject_id"]) == ["p1", "p2"]
    assert list(pivot["both_blocks_converged"]) == [True, False]
    assert list(pivot["lr_change"]) == [1.5, -0.5]
    counts = module.count_participants_per_version(pivot)
    assert counts["n_nonconverged_V1"] == 1
    assert counts["n_regression_participants_V1"] == 2

    with pytest.raises(ValueError, match="'p3'"):
        module.build_lr_pivot(_lr_blocks(blocks + [("p3", 1, 0.2, True)]))
    with pytest.raises(ValueError, match="'p1'"):
        module.build_lr_pivot(_lr_blocks(blocks + [("p1", 2, 0.7, True)]))


def test_fdr_correction_spans_every_factor_and_version_together():
    """The q-values treat all factors x versions in a table as one family.

    No other test touches the correction. This catches a correction grouped
    by factor or by version, which would look valid but control too few
    tests. The data are drawn so that both wrong groupings give different
    q-values from the right one, which the test checks first. It also checks
    the baseline-adjusted table, whose family is its factor rows only: if the
    ``lr_block1`` rows joined the family, every q-value there would change.
    """
    module = _load_module()
    rng = np.random.default_rng(3)
    n_per_version = 30
    factor_cols = ["ML1", "ML2", "ML3"]
    rows = []
    for version in module.VERSIONS:
        for i in range(n_per_version):
            rows.append({"subject_id": f"{version}_{i}", "version": version})
    lr_pivot = pd.DataFrame(rows)
    scores = pd.DataFrame(
        rng.normal(size=(len(lr_pivot), len(factor_cols))), columns=factor_cols
    )
    scores.insert(0, "assignedSubjectId", lr_pivot["subject_id"])
    # ML1 has a real effect, so the p-values spread out across factors.
    lr_pivot["lr_change"] = 0.6 * scores["ML1"] + rng.normal(size=len(lr_pivot))
    lr_pivot["lr_block1"] = rng.normal(size=len(lr_pivot))
    lr_pivot["lr_block2"] = lr_pivot["lr_block1"] + lr_pivot["lr_change"]

    merged, results = module.build_updated_factor_results_df(
        lr_pivot, scores, factor_cols
    )

    assert len(results) == 12

    def bh(p_values: pd.Series) -> np.ndarray:
        return multipletests(p_values, method="fdr_bh")[1]

    by_factor = results.groupby("Factor")["p"].transform(bh)
    by_version = results.groupby("Version")["p"].transform(bh)
    whole_table = bh(results["p"])
    assert not np.allclose(by_factor, whole_table)
    assert not np.allclose(by_version, whole_table)

    np.testing.assert_allclose(results["p_fdr"], whole_table)
    assert list(results["Sig_fdr"]) == [module.sig_stars(q) for q in whole_table]
    assert "p_bonf" not in results.columns

    adjusted = module.build_baseline_adjusted_results_df(merged, factor_cols)
    baseline_rows = adjusted["Predictor"] == "lr_block1"
    assert baseline_rows.sum() == 4
    factor_family = bh(adjusted.loc[~baseline_rows, "p"])
    assert not np.allclose(factor_family, bh(adjusted["p"])[~baseline_rows])
    np.testing.assert_allclose(adjusted.loc[~baseline_rows, "p_fdr"], factor_family)
    assert adjusted.loc[baseline_rows, "p_fdr"].isna().all()
    assert (adjusted.loc[baseline_rows, "Sig_fdr"] == "").all()


# ── Questionnaire archive member check ───────────────────────────────────────


def test_missing_questionnaire_zip_raises_and_names_file(tmp_path: Path):
    module = _load_module()
    missing = tmp_path / "does_not_exist.zip"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        module.load_questionnaire_data(missing)


def test_zip_with_zero_csv_members_raises(tmp_path: Path):
    module = _load_module()
    zip_path = _make_zip(tmp_path / "q.zip", {"readme.txt": "not a csv"})
    with pytest.raises(ValueError, match="exactly one CSV"):
        module.load_questionnaire_data(zip_path)


def test_zip_with_two_csv_members_raises(tmp_path: Path):
    module = _load_module()
    zip_path = _make_zip(
        tmp_path / "q.zip",
        {"a.csv": "assignedSubjectId\ns1\n", "b.csv": "assignedSubjectId\ns2\n"},
    )
    with pytest.raises(ValueError, match="exactly one CSV"):
        module.load_questionnaire_data(zip_path)


def test_zip_ignores_macosx_entries(tmp_path: Path):
    module = _load_module()
    zip_path = _make_zip(
        tmp_path / "q.zip",
        {
            "__MACOSX/._data.csv": "junk",
            "data.csv": "assignedSubjectId\ns1\n",
        },
    )
    data_q = module.load_questionnaire_data(zip_path)
    assert list(data_q["assignedSubjectId"]) == ["s1"]


def test_questionnaire_data_loads_fresh_each_call(tmp_path: Path):
    module = _load_module()
    zip_path = _make_zip(tmp_path / "q.zip", {"data.csv": "assignedSubjectId\ns1\n"})
    first = module.load_questionnaire_data(zip_path)
    second = module.load_questionnaire_data(zip_path)
    assert first is not second
    pd.testing.assert_frame_equal(first, second)


# ── A single real run, used by both checks ───────────────────────────────────


@pytest.fixture(scope="module")
def real_learning_rate_run(tmp_path_factory):
    """Run the real `learning_rate` pipeline once, for the output and figure checks.

    Uses the real, committed inputs and the real `ensure_dm_sans_font()`
    with the bundled font file, so that the figure-list check exercises the
    real path.
    """
    module = _load_module()
    output_dir = tmp_path_factory.mktemp("regression_output")
    # The shared checks folder lies outside `output_dir`, as it does in a real run.
    factor_checks_dir = tmp_path_factory.mktemp("factor_checks")

    # Seed another parameter's directory with a marker file, to prove the
    # `learning_rate` run below does not touch it (output-folder check).
    marker_dir = output_dir / "learning_rate_neg" / "tables"
    marker_dir.mkdir(parents=True)
    marker_path = marker_dir / "marker.csv"
    marker_path.write_text("untouched\n")
    marker_before = marker_path.read_text()
    marker_mtime_before = marker_path.stat().st_mtime_ns

    # Leave a stale summary_df.csv in this parameter's folder, to prove the
    # run deletes it.
    own_tables_dir = output_dir / "learning_rate" / "tables"
    own_tables_dir.mkdir(parents=True)
    (own_tables_dir / "summary_df.csv").write_text("old,results\n")

    config = module.PARAMETER_CONFIGS["learning_rate"]
    result = module.run_regression_analysis(
        parameter="learning_rate",
        lr_csv=config["lr_csv"],
        model_name=config["model"],
        lr_column=config["lr_column"],
        questionnaire_zip=module.QUESTIONNAIRE_ZIP,
        output_dir=output_dir,
        factor_checks_dir=factor_checks_dir,
    )

    return {
        "module": module,
        "output_dir": output_dir,
        "factor_checks_dir": factor_checks_dir,
        "result": result,
        # The font every figure of the run was drawn with.
        "font_family": list(module.plt.rcParams["font.family"]),
        "marker_path": marker_path,
        "marker_before": marker_before,
        "marker_mtime_before": marker_mtime_before,
    }


@pytest.mark.real_data
def test_c04_other_parameters_directory_untouched(real_learning_rate_run):
    marker_path = real_learning_rate_run["marker_path"]
    assert marker_path.read_text() == real_learning_rate_run["marker_before"]
    assert marker_path.stat().st_mtime_ns == real_learning_rate_run["marker_mtime_before"]

    other_dir = real_learning_rate_run["output_dir"] / "learning_rate_neg"
    assert {p for p in other_dir.rglob("*")} == {other_dir / "tables", marker_path}


@pytest.mark.real_data
def test_c04_run_writes_only_under_its_own_parameter_dir(real_learning_rate_run):
    output_dir = real_learning_rate_run["output_dir"]
    top_level = {p.name for p in output_dir.iterdir()}
    assert top_level == {"learning_rate", "learning_rate_neg"}


@pytest.mark.real_data
def test_c05_figure_list_matches_expected(real_learning_rate_run):
    figures_dir = real_learning_rate_run["output_dir"] / "learning_rate" / "figures"
    actual = {p.name for p in figures_dir.glob("*.svg")}

    # The 19 figures every run must write, spelled out here (not read from the
    # script), so that a figure dropped or renamed in the script is caught.
    fixed = {
        "factor_loadings_all_questionnaires.svg",
        "questionnaire_correlation_matrix.svg",
        "lr_change_vs_STICSA.svg",
        "lr_change_vs_IUS_12.svg",
        "lr_change_vs_PHQ_8.svg",
        "lr_change_vs_GAD_7.svg",
        "lr_change_vs_LOT_R.svg",
        "lr_change_vs_BIS.svg",
        "lr_change_vs_AES.svg",
        "lr_change_vs_EAT.svg",
        "lr_change_vs_SRDS.svg",
        "lr_change_vs_STAI.svg",
        "lr_change_vs_OCI.svg",
        "lr_change_vs_LSAS.svg",
        "factor_scores_vs_lr_change.svg",
        "scree_plot.svg",
        "selected_factor_score_correlations.svg",
        "selected_factor_loadings.svg",
        "efa_factor_loadings.svg",
    }
    # One more figure is allowed, depending on the data.
    optional = {"high_correlations.svg"}

    assert len(fixed) == 19
    assert fixed.issubset(actual)
    assert actual.issubset(fixed | optional)
    assert actual == set(real_learning_rate_run["result"]["figures_written"])
    # The figures use the publication font. The seaborn theme resets the font,
    # so it must be set again after the theme.
    module = real_learning_rate_run["module"]
    assert real_learning_rate_run["font_family"] == [module.DM_SANS_FONT_NAME]


@pytest.mark.real_data
def test_c05_high_correlations_figure_matches_pair_count(real_learning_rate_run):
    """`high_correlations.svg` must be present exactly when pairs were found.

    `test_c05_figure_list_matches_expected` only checks that the actual
    figure set is a subset of fixed-plus-optional names, which would pass
    silently even if `high_correlations.svg` were always missing. This
    checks the actual data-dependent condition instead.
    """
    result = real_learning_rate_run["result"]
    figures_dir = real_learning_rate_run["output_dir"] / "learning_rate" / "figures"
    high_correlations_path = figures_dir / "high_correlations.svg"

    assert high_correlations_path.is_file() == (result["high_correlation_pair_count"] > 0)


@pytest.mark.real_data
def test_tables_written_with_fixed_names(real_learning_rate_run):
    tables_dir = real_learning_rate_run["output_dir"] / "learning_rate" / "tables"
    actual = {p.name for p in tables_dir.glob("*.csv")}
    expected = {
        "counts.csv",
        "initial_eigenvalues.csv",
        "initial_factor_loadings.csv",
        "indiv_df.csv",
        "factor_results_df.csv",
        "eigenvalues.csv",
        "factor_loadings_selected_solution.csv",
        "factor_scores_selected_solution.csv",
        "lr_change_regression_results.csv",
        "lr_change_baseline_adjusted_results.csv",
    }
    assert expected.issubset(actual)
    # summary_df.csv is not among the tables; the copy planted by the fixture
    # is deleted.
    assert "summary_df.csv" not in actual


@pytest.mark.real_data
def test_factor_checks_folder_holds_exactly_the_check_files(real_learning_rate_run):
    """The checks folder has the seven check files, and they agree with counts.csv.

    The factor count in `counts.csv` and `factor_count.csv` must be the
    chosen count, `SELECTED_N_FACTORS`, not one of the reported rule counts.
    """
    checks_dir = real_learning_rate_run["factor_checks_dir"]
    assert {p.name for p in checks_dir.iterdir()} == {
        "factor_count.csv",
        "factor_count_by_position.csv",
        "suitability.csv",
        "kmo_by_item.csv",
        "split_half_congruence.csv",
        "skew_filtered_items.csv",
        "questionnaire_reliability.csv",
    }

    # One reliability row for each of the 12 questionnaires, with the columns
    # the plan names. Negative item-rest correlations are reported, not tested.
    reliability = pd.read_csv(checks_dir / "questionnaire_reliability.csv")
    module = real_learning_rate_run["module"]
    assert list(reliability["questionnaire"]) == module.QUESTIONNAIRE_KEYS
    assert list(reliability.columns) == [
        "questionnaire",
        "n_items",
        "n_participants",
        "cronbach_alpha",
        "min_item_rest_correlation",
        "item_with_min_item_rest_correlation",
    ]

    factor_count = pd.read_csv(checks_dir / "factor_count.csv")
    assert len(factor_count) == 1
    selected = int(factor_count["selected_n_factors"].iloc[0])
    assert selected == real_learning_rate_run["module"].SELECTED_N_FACTORS

    tables_dir = real_learning_rate_run["output_dir"] / "learning_rate" / "tables"
    counts = pd.read_csv(tables_dir / "counts.csv")
    assert int(counts["updated_selected_n_factors"].iloc[0]) == selected


@pytest.mark.real_data
def test_counts_csv_contains_c03_scalars(real_learning_rate_run):
    """counts.csv has one column per metric, with integer counts kept as integers.

    A single shared "value" column (long format) would upcast every count
    to float64 as soon as any metric (e.g. `mean_age`) is a float, printing
    integer counts like "313.0". One column per metric keeps each column's
    own dtype.
    """
    tables_dir = real_learning_rate_run["output_dir"] / "learning_rate" / "tables"
    counts = pd.read_csv(tables_dir / "counts.csv")
    assert len(counts) == 1

    integer_metrics = [
        "num_ppts_LR_cleaned",
        "num_ppts_Q_cleaned",
        "num_ppts_with_any_questionnaire",
        "num_ppts_completed_all_questionnaires",
        "num_ppts_failed_more_than_one_check",
        "num_male",
        "num_female",
        "n_factors",
        "updated_selected_n_factors",
    ]
    for version in ["V1", "V2", "V3", "V4"]:
        integer_metrics += [
            f"n_nonconverged_{version}",
            f"n_regression_participants_{version}",
        ]
    for metric in integer_metrics + ["mean_age", "std_age"]:
        assert metric in counts.columns

    for metric in integer_metrics:
        assert pd.api.types.is_integer_dtype(counts[metric]), (
            f"{metric} should stay an integer dtype, got {counts[metric].dtype}"
        )


@pytest.mark.real_data
def test_factor_analysis_items_hold_no_filler_or_attention_check(
    real_learning_rate_run,
):
    """No filler or attention-check item reaches the item-level factor analysis.

    The key marks the 4 LOT-R fillers and the 8 attention checks (the IUS-12
    check included). If any of them were scored, it would enter the totals and
    the factor solution without belonging to any scale.
    """
    module = real_learning_rate_run["module"]
    scoring_key = module.load_scoring_key(module.SCORING_KEY_CSV)
    is_left_out = ~scoring_key["scoring"].isin(module.SCORED_ROLES)
    left_out = set(scoring_key.loc[is_left_out, "question_id"])
    assert len(left_out) == 12

    loadings = real_learning_rate_run["result"]["updated_selected_loadings"]
    assert left_out.isdisjoint(set(loadings["item"]))


# ── Scoring key and attention checks on an invented archive ──────────────────

# One row of the invented archive is (questionnaire key, item number, text,
# role, data marks it as an attention check). As in the archive, the
# text is a short label for scale items and the full wording for the checks.
# The wording of the two checks differs from the real ones only in the final
# full stop, as in the real data.
_SYNTHETIC_ITEMS = [
    ("STICSA", 1, "fast heartbeat", "forward", False),
    ("STICSA", 2, "My hands have seven fingers.", "attention_check", False),
    ("STAI", 1, "feeling composed", "reverse", False),
    ("STAI", 2, "sense of failure", "forward", False),
    (
        "STAI",
        3,
        "I feel like I have always been a human being",
        "attention_check",
        True,
    ),
    ("STAI", 4, "likes tea", "not_scored_filler", False),
    ("EAT", 1, "fear of weight gain", "reverse", False),
]
_SYNTHETIC_TITLES = {
    "STICSA": "STICSA Assessment",
    "STAI": "STAI Assessment",
    "EAT": "EAT Assessment",
}
_FOUR_LABELS = ["not at all", "a little", "moderately", "very much so"]
_SYNTHETIC_LABELS = {
    "STICSA": _FOUR_LABELS,
    "STAI": _FOUR_LABELS,
    "EAT": ["always", "usually", "often", "sometimes", "rarely", "never"],
}


def _synthetic_archive(raw_answers: dict[str, list[int]]):
    """Build an invented long-format questionnaire archive and its scoring key.

    Args:
        raw_answers: For each participant, the raw answer to each of the 7
            items of ``_SYNTHETIC_ITEMS``, in that order.

    Returns:
        A tuple ``(data_q, scoring_key)``. STICSA and STAI answers range from 0
        to 3 and EAT answers from 0 to 5.
    """
    rows = []
    for subject, answers in raw_answers.items():
        for (key, number, text, _role, marked), raw in zip(_SYNTHETIC_ITEMS, answers):
            rows.append(
                {
                    "assignedSubjectId": subject,
                    "questionnaireKey": key,
                    "questionnaireTitle": _SYNTHETIC_TITLES[key],
                    "questionId": f"{key}__{number}",
                    "questionText": text,
                    "isAttentionCheck": marked,
                    "rawResponse": raw,
                    "responseLabel": _SYNTHETIC_LABELS[key][raw],
                    "scaleMin": 0,
                    "scaleMax": 5 if key == "EAT" else 3,
                }
            )
    scoring_key = pd.DataFrame(
        [
            {
                "questionnaire_key": key,
                "question_id": f"{key}__{number}",
                "item_label": text,
                "scoring": role,
            }
            for key, number, text, role, _marked in _SYNTHETIC_ITEMS
        ]
    )
    return pd.DataFrame(rows), scoring_key


def test_scoring_key_attention_checks_and_repeated_answers_on_invented_archive():
    """Scoring, check matching and the stops, on a 6-participant archive.

    No other test covers scoring or the matching of attention checks, so a
    reversed item left unreversed, a check missed because its wording ends in
    a full stop, or a quietly accepted key error would pass unnoticed. The
    answers per participant are heartbeat, seven fingers, composed, failure,
    human being, tea, weight gain. p2 answers "a little" to seven fingers, the
    second most correct answer, so passes. p3 fails only the STICSA check, p4
    fails both checks, and p5 fails only the STAI check, so only p4 is
    excluded.
    """
    module = _load_module()
    # Only the two checks that exist in the invented archive.
    module.ATTENTION_CHECKS = [
        check
        for check in module.ATTENTION_CHECKS
        if check["questionnaire"] in ("STICSA Assessment", "STAI Assessment")
    ]
    keys = ["STICSA", "STAI", "EAT"]
    data_q, scoring_key = _synthetic_archive(
        {
            "p1": [1, 0, 3, 3, 3, 2, 0],
            "p2": [2, 1, 1, 0, 2, 1, 4],
            "p3": [1, 3, 2, 1, 3, 2, 2],
            "p4": [3, 3, 2, 1, 0, 0, 5],
            "p5": [0, 0, 2, 2, 0, 3, 1],
            "p6": [0, 0, 0, 0, 3, 0, 0],
        }
    )
    lr_data = pd.DataFrame({"subject_id": ["p1", "p2", "p3", "p4", "p5", "p7"]})

    cleaned = module.clean_and_match_data(lr_data, data_q, keys, scoring_key)

    # The STICSA check ends in a full stop in the data but not in the checks
    # list, and the STAI check is data-marked while the STICSA one is not.
    # Both are found, so p4 fails 2 checks and is excluded; if the STICSA
    # check were missed, p4 would have only 1 failure and stay in. p3 and p5
    # (one failure each) stay in, as does p2, whose "a little" passes. p6 has
    # no fit.
    assert cleaned["num_ppts_failed_more_than_one_check"] == 1
    assert sorted(cleaned["q_all"]["assignedSubjectId"]) == ["p1", "p2", "p3", "p5"]

    # "Very much so" (3) on the reverse item "composed" scores 0 and on the
    # forward item "failure" scores 3. EAT "always" (0) scores 5 on its own
    # 0-5 range.
    scores = cleaned["item_scores"].set_index(["assignedSubjectId", "questionId"])[
        "itemScore"
    ]
    assert scores[("p1", "STAI__1")] == 0
    assert scores[("p1", "STAI__2")] == 3
    assert scores[("p1", "EAT__1")] == 5
    assert scores[("p2", "EAT__1")] == 1

    # Fillers and both checks reach neither the item scores nor the totals.
    left_out = {"STICSA__2", "STAI__3", "STAI__4"}
    assert left_out.isdisjoint(set(cleaned["item_scores"]["questionId"]))
    q_all = cleaned["q_all"].set_index("assignedSubjectId")
    assert q_all.loc["p1", "STAI"] == 0 + 3
    assert q_all.loc["p1", "STICSA"] == 1
    item_table, _ = module.prepare_item_level_data(cleaned["item_scores"])
    assert left_out.isdisjoint(set(item_table.columns))

    def fails(match, data=data_q, key=scoring_key):
        """Assert that cleaning and scoring stop with a matching error.

        Args:
            match: Text the error message must contain (a regular expression).
            data: The questionnaire rows to clean, by default the valid ones.
            key: The scoring key to use, by default the valid one.
        """
        with pytest.raises(ValueError, match=match):
            module.clean_and_match_data(lr_data, data, keys, key)

    # A key entry is missing, an item's label differs from the data's text
    # (the archive was made from another version of the key), and a check
    # item is relabelled "forward".
    fails("different items", key=scoring_key[scoring_key["question_id"] != "STAI__2"])
    other_label = scoring_key.copy()
    other_label.loc[other_label["question_id"] == "STAI__2", "item_label"] = (
        "sense of success"
    )
    fails("differs from their item_label", key=other_label)
    relabelled = scoring_key.copy()
    relabelled.loc[relabelled["question_id"] == "STICSA__2", "scoring"] = "forward"
    fails("must be 'attention_check'", key=relabelled)

    # A check whose question is not in the data stops the run.
    module.ATTENTION_CHECKS = [
        *module.ATTENTION_CHECKS,
        {
            "questionnaire": "STAI Assessment",
            "question": "i have no such question",
            "valid_responses": ["not at all"],
        },
    ]
    fails("matched 0 questions")
    module.ATTENTION_CHECKS = module.ATTENTION_CHECKS[:-1]

    # A misspelt passing answer stops the run instead of failing everyone who
    # gave that answer.
    checks_before = module.ATTENTION_CHECKS
    module.ATTENTION_CHECKS = [
        {**check, "valid_responses": [*check["valid_responses"], "not at al"]}
        for check in checks_before
    ]
    fails("nobody gave")
    module.ATTENTION_CHECKS = checks_before

    # A participant who started a questionnaire but skipped its check.
    is_skipped = (data_q["assignedSubjectId"] == "p2") & (
        data_q["questionId"] == "STAI__3"
    )
    skipped = data_q[~is_skipped]
    fails("not answered", data=skipped)

    # Two answers to one item for a participant with a fit.
    first_row_of_p1 = data_q[data_q["assignedSubjectId"] == "p1"].iloc[[0]]
    repeated = pd.concat([data_q, first_row_of_p1])
    fails("more than one answer", data=repeated)


# ── Factor count and factor checks on simulated data ─────────────────────────


def _simulate_three_factor_items(n_participants: int = 600) -> pd.DataFrame:
    """Simulate items with a known 3-factor structure.

    There are three uncorrelated factors with 12 items each. Every item of the
    first, second and third factor loads 0.8, 0.65 and 0.5 on its factor, and
    the rest of its variance is independent noise.

    Args:
        n_participants: Number of simulated participants.

    Returns:
        A participants-by-items table with 36 columns named ``F<k>__<i>``.
    """
    rng = np.random.default_rng(2026)
    columns = {}
    for factor_number, loading in enumerate([0.8, 0.65, 0.5], start=1):
        factor_scores = rng.standard_normal(n_participants)
        for item_number in range(1, 13):
            noise = rng.standard_normal(n_participants)
            columns[f"F{factor_number}__{item_number}"] = (
                loading * factor_scores + np.sqrt(1 - loading**2) * noise
            )
    return pd.DataFrame(columns)


def test_factor_count_and_factor_matching_on_known_three_factor_data():
    """Eigenvalues, CNG, parallel analysis and factor matching find 3 factors.

    A wrong factor count would quietly change which factors enter every
    learning-rate regression, and the existing real-data run cannot tell a
    wrong count from a right one. Here the answer is known: 3 factors.
    """
    module = _load_module()
    items = _simulate_three_factor_items()

    # Eigenvalues come straight from the correlation matrix, and equal the
    # ones factor_analyzer reports.
    eigenvalues = module.calculate_item_eigenvalues(items)
    reference_model = FactorAnalyzer(n_factors=3, rotation=None, method="ml").fit(items)
    factor_analyzer_eigenvalues = reference_model.get_eigenvalues()[0]
    np.testing.assert_allclose(
        eigenvalues, factor_analyzer_eigenvalues, rtol=0, atol=1e-10
    )

    # CNG matches the worked example of nFactors::nCng. The slope changes
    # 1.9, 1.4 and 0.45 have their maximum at position 1, so the count is 3.
    # An off-by-one error would give 2 or 4.
    worked_example = np.array([6, 4, 2, 1, 0.9, 0.8, 0.7, 0.6])
    np.testing.assert_allclose(
        module.calculate_cng_slope_changes(worked_example), [1.9, 1.4, 0.45]
    )
    assert module.calculate_cng_count(worked_example) == 3

    # Parallel analysis finds the 3 real factors. CNG on its own may place the
    # elbow one step later, so only a lower bound is checked for it.
    thresholds = module.calculate_parallel_analysis_thresholds(*items.shape)
    counts = module.calculate_factor_count_rules(eigenvalues, thresholds)
    assert counts["pa_count"] == 3
    assert counts["cng_count"] >= 3

    # Matching recovers a known column permutation and sign flip. Column 0 of
    # the scrambled loadings is factor 2, column 1 is factor 0 with its sign
    # flipped, and column 2 is factor 1.
    full_model = module.fit_factor_model_with_checks(
        items, 3, "simulated items, 3 factors, full sample"
    )
    loadings = full_model.loadings_
    scrambled = loadings[:, [2, 0, 1]] * np.array([1.0, -1.0, 1.0])
    matched_columns, signs, congruences = module.match_factors_to_reference(
        loadings, scrambled
    )
    assert list(matched_columns) == [1, 2, 0]
    assert list(signs) == [-1.0, 1.0, 1.0]
    np.testing.assert_allclose(congruences, 1.0)

    # In each half of the participants, every factor is matched to a
    # full-sample factor with high congruence.
    full_loadings = pd.DataFrame(
        loadings, index=items.columns, columns=["ML1", "ML2", "ML3"]
    )
    split_half = module.calculate_split_half_congruence(items, full_loadings)
    assert len(split_half) == 6
    assert (split_half["congruence"] > 0.9).all()


def test_item_with_unique_variance_on_the_lower_bound_stops_the_run():
    """An impossible (Heywood) item stops the run with an error naming the fit.

    One item is the factor plus a tiny amount of noise. Its true unique
    variance (about 0.000001) is below the 0.005 lower bound that
    factor_analyzer allows, so the fitted communality is at least 0.995. The
    other five items are built so that their correlations fit a 1-factor model
    exactly (all of them 0.36, and 0.6 with the near-copy of the factor), which
    removes any dependence on random sampling.
    """
    module = _load_module()
    n_participants = 200
    rng = np.random.default_rng(7)
    random_values = rng.standard_normal((n_participants, 7))
    random_values -= random_values.mean(axis=0)
    # QR gives uncorrelated, mean-zero columns of equal length.
    uncorrelated, _ = np.linalg.qr(random_values)
    factor = uncorrelated[:, 0]
    items = {"near_copy_of_factor": factor + 0.001 * uncorrelated[:, 1]}
    for item_number in range(1, 6):
        items[f"item_{item_number}"] = (
            0.6 * factor + 0.8 * uncorrelated[:, item_number + 1]
        )

    with pytest.raises(ValueError, match="simulated Heywood check.*Heywood case"):
        module.fit_factor_model_with_checks(
            pd.DataFrame(items), 1, "simulated Heywood check, 1 factor"
        )


# ── Shared folder of factor-analysis checks ──────────────────────────────────


def test_checks_folder_is_written_once_then_compared(tmp_path: Path):
    """A second run must reproduce the stored checks; a changed value stops it.

    The three learning-rate parameters write into one folder. Without a
    comparison that really fires, the runs could silently use different
    factor solutions.
    """
    module = _load_module()
    folder = tmp_path / "factor_checks"
    tables = {
        "factor_count.csv": pd.DataFrame({"selected_n_factors": [3]}),
        "suitability.csv": pd.DataFrame({"kmo_overall": [0.8123456789]}),
    }

    module.write_or_compare_factor_checks(tables, folder)
    assert {p.name for p in folder.iterdir()} == set(tables)

    # Identical tables pass.
    module.write_or_compare_factor_checks(tables, folder)

    # A changed value raises, and the stored file is left as it was.
    changed = {**tables, "suitability.csv": pd.DataFrame({"kmo_overall": [0.85]})}
    stored_before = (folder / "suitability.csv").read_text()
    with pytest.raises(ValueError, match="different data or code"):
        module.write_or_compare_factor_checks(changed, folder)
    assert (folder / "suitability.csv").read_text() == stored_before

    # A folder with only some of the files raises.
    (folder / "factor_count.csv").unlink()
    with pytest.raises(ValueError, match="only some"):
        module.write_or_compare_factor_checks(tables, folder)


def test_item_loadings_plot_works_for_one_factor(tmp_path: Path):
    """The loadings bar plot is drawn for a one-factor solution.

    The plotting helper fails on a single panel when it makes the figure
    itself; the real run uses 4 factors, so it cannot show this case.
    """
    module = _load_module()
    loadings_csv = tmp_path / "loadings.csv"
    pd.DataFrame(
        {
            "item": ["AES__1", "AES__2", "BIS__1"],
            "ML1": [0.7, 0.5, -0.2],
            "measure": ["AES", "AES", "BIS"],
            "itemNumber": [1, 2, 3],
        }
    ).to_csv(loadings_csv, index=False)
    figure_path = tmp_path / "loadings.svg"

    module.plot_item_loadings_by_measure(loadings_csv, ["One factor"], figure_path)

    assert figure_path.is_file()


def test_factor_names_stop_the_run_when_factors_move():
    """A factor name is refused when its factor's order or sign has changed.

    New data can give the same factors in a different ML order or with the
    sign reversed. The real run only shows the case where the names fit.
    """
    module = _load_module()
    # One item per marker questionnaire, each loading on its own factor with
    # the expected sign (IUS-12 negatively on ML4).
    factor_cols = ["ML1", "ML2", "ML3", "ML4"]
    loadings = pd.DataFrame(
        np.array(
            [
                [0.7, 0.0, 0.1, 0.0],
                [0.0, 0.8, 0.0, 0.1],
                [0.1, 0.0, 0.6, 0.2],
                [0.2, 0.1, 0.0, -0.6],
            ]
        ),
        columns=factor_cols,
    )
    loadings["measure"] = ["STICSA", "LSAS", "AES", "IUS_12"]
    assert module.name_selected_factors(loadings, factor_cols) == list(
        module.FACTOR_NAMES.values()
    )

    # ML1 and ML2 swap places.
    swapped = loadings.rename(columns={"ML1": "ML2", "ML2": "ML1"})
    with pytest.raises(ValueError, match="ML1 does not fit"):
        module.name_selected_factors(swapped, factor_cols)

    # ML4 has its sign reversed.
    reversed_sign = loadings.assign(ML4=-loadings["ML4"])
    with pytest.raises(ValueError, match="ML4 does not fit"):
        module.name_selected_factors(reversed_sign, factor_cols)
