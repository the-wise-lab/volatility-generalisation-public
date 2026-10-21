#!/usr/bin/env python3
"""Individual-differences regression analysis for learning-rate parameters.

This script tests whether each participant's learning-rate parameter relates
to their questionnaire scores. It covers ``learning_rate`` from
``rescorla_wagner_single_value``, and ``learning_rate_neg`` and
``learning_rate_pos`` from ``dual_lr_single_value``. It runs after
``compile_results.py`` has written the subject-level parameter tables. Each
``--parameter`` run reads its own fixed inputs fresh and writes its own output
folder under ``--output-dir``, so the three runs do not overwrite each other's
results.

Scientific choices made by this analysis:

- the battery of 6 attention checks (STICSA, PHQ-8, GAD-7, STAI, AES and OCI),
  excluding anyone who fails more than 1. Each check passes on its two most
  correct answers. Each check is found by its wording, ignoring spaces,
  capital letters and one final full stop (five of the questions end in a
  full stop in the data). The run stops if a check
  matches no question, or if anyone who started that questionnaire did not
  answer it. The passing answers are listed in ``ATTENTION_CHECKS``. The IUS-12
  and LOT-R checks are not in the battery. The IUS-12
  check is failed by 32-61% of respondents, depending on which answers pass
  (one choice of passing answers makes "entirely characteristic of me" a
  fail). Failing these items may reflect symptoms such as blunted emotions or
  anhedonia rather than inattention. Their answers are still left out of the
  IUS-12 and LOT-R totals;
- one scoring key for all 136 items (``config/questionnaire_scoring_key.csv``).
  Each item is ``forward``, ``reverse``, ``not_scored_filler`` (the four LOT-R
  filler items) or ``attention_check`` (the eight check items, including the
  IUS-12 and LOT-R checks that are not used for exclusion). Reverse items score ``scaleMin + scaleMax - rawResponse``. The
  data file's own stored score, reverse-coding and attention-check columns are
  not used for scoring. Fillers and checks reach neither the totals nor the
  item-level factor analysis. Several scales are shortened versions (for
  example 9 of 20 SRDS items and 7 of 18 AES items), so the totals are not
  comparable with published norms;
- requiring completion of all 12 questionnaires;
- item filtering for the updated exploratory factor analysis (EFA): zero
  variance, ``|skew| >= 2``, and pairwise correlation above 0.7;
- mean imputation of missing questionnaire items before the updated EFA;
- oblimin rotation with maximum-likelihood extraction for every factor
  analysis;
- factor count for the updated EFA = 4 (``SELECTED_N_FACTORS``), chosen from
  the elbow of the scree plot. The parallel analysis count
  and the Cattell-Nelson-Gorsuch (CNG) count are reported in
  ``factor_count.csv`` but do not decide anything;
- the logit transform of learning rates;
- participants whose fits did not converge are included, and counted for
  each version in ``counts.csv`` (``n_nonconverged_V1`` to ``_V4``, beside
  ``n_regression_participants_V1`` to ``_V4``). A participant without exactly
  one block 1 and one block 2, a non-finite logit learning rate, or a version
  with too few participants for its model stops the run;
- Benjamini-Hochberg false discovery rate correction (``p_fdr``,
  ``Sig_fdr``), applied once over every test in a table: all item-level
  factors x 4 versions in ``lr_change_regression_results.csv``, the factor
  rows x 4 versions in ``lr_change_baseline_adjusted_results.csv``, 12
  questionnaires x 4 versions in ``indiv_df.csv``, and all totals factors x 4
  versions in ``factor_results_df.csv``;
- the main result is the regression of learning-rate change (block 2 minus
  block 1) on all item-level factors together. A baseline-adjusted model,
  ``lr_block2 ~ lr_block1 + ML1 + ... + MLk`` on the same participants, is
  reported beside it as a sensitivity check. The two can differ, because
  adjusting for a noisy block 1 estimate biases the factor effects when a trait
  relates to block 1 (Lord's paradox);
- the single-questionnaire regressions (``indiv_df.csv``) and the
  totals-factor regressions (``factor_results_df.csv``) are exploratory.

Checks on the factor analyses (they only use questionnaire answers, so every
parameter should give the same result):

- every factor model must converge and must have no communality at or above
  0.995 (a Heywood case, where an item's unique variance has hit the lower
  bound of 0.005 allowed by ``factor_analyzer``). Either problem stops the run
  with an error that names the failed fit;
- the Kaiser-Meyer-Olkin measure and Bartlett's test of sphericity are reported
  but never stop the run;
- one split-half check: the factors from each half of the participants are
  matched to the full-sample factors by Tucker congruence;
- the items removed by the skew filter are listed;
- a reliability table for each questionnaire (Cronbach's alpha and the lowest
  item-rest correlation), a data-based check that the scoring key is right. A
  negative item-rest correlation points to a key error. It is reported and
  never stops the run.

These tables are written once to ``--factor-checks-dir`` (default
``results/factor_analysis_checks``). A later run compares its tables with the
files already there and stops if they differ.
"""

from __future__ import annotations

import argparse
import io
import warnings
import zipfile
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from factor_analyzer import (  # noqa: E402
    FactorAnalyzer,
    calculate_bartlett_sphericity,
    calculate_kmo,
)
from numpy.linalg import eigh  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import skew  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.formula.api import ols  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from stats_utils.efa.plotting import plot_factor_loadings  # noqa: E402

from volatility_generalisation.figures.publication_figures import (  # noqa: E402
    DM_SANS_FONT_NAME,
    ensure_dm_sans_font,
)

# ── Fixed per-parameter input mapping ──────────────────────────────────────

QUESTIONNAIRE_ZIP = Path("data/questionnaire_data.csv.zip")
# The only place where each item's role in scoring is defined.
SCORING_KEY_CSV = Path("config/questionnaire_scoring_key.csv")

PARAMETER_CONFIGS: dict[str, dict[str, Any]] = {
    "learning_rate": {
        "lr_csv": Path("results/compiled/subject_parameters.csv"),
        "model": "rescorla_wagner_single_value",
        "lr_column": "learning_rate",
    },
    "learning_rate_neg": {
        "lr_csv": Path("results/compiled_dual_lr/subject_parameters.csv"),
        "model": "dual_lr_single_value",
        "lr_column": "learning_rate_neg",
    },
    "learning_rate_pos": {
        "lr_csv": Path("results/compiled_dual_lr/subject_parameters.csv"),
        "model": "dual_lr_single_value",
        "lr_column": "learning_rate_pos",
    },
}

# ── Questionnaire constants (shared across parameters) ─────────────────────

QUESTIONNAIRE_KEYS = [
    "STICSA",
    "IUS_12",
    "PHQ_8",
    "GAD_7",
    "LOT_R",
    "BIS",
    "AES",
    "EAT",
    "SRDS",
    "STAI",
    "OCI",
    "LSAS",
]

Q_LABELS = {
    "STICSA": "STICSA (Anxiety)",
    "IUS_12": "IUS-12 (Intolerance of Uncertainty)",
    "PHQ_8": "PHQ-8 (Depression)",
    "GAD_7": "GAD-7 (Generalised Anxiety)",
    "LOT_R": "LOT-R (Optimism)",
    "BIS": "BIS (Impulsivity)",
    "AES": "AES (Apathy)",
    "EAT": "EAT (Eating Attitudes)",
    "SRDS": "SRDS (Depression)",
    "STAI": "STAI (State-Trait Anxiety)",
    "OCI": "OCI (Obsessive-Compulsive)",
    "LSAS": "LSAS (Social Anxiety)",
}

# Bar colour for each questionnaire in the item-loadings figure
# (efa_factor_loadings.svg). The keys are the item-name prefixes in the
# loadings table, so EAT is "EAT_52" here. The items are drawn in this
# (alphabetical) order, so each questionnaire's block of bars sits next to the
# one listed after it. The colours were checked as a set: neighbouring blocks
# stay distinct with normal colour vision and in simulated red-green colour
# blindness, and every colour has at least 3:1 contrast against white.
QUESTIONNAIRE_COLOURS = {
    "AES": "#e34948",
    "BIS": "#2a78d6",
    "EAT_52": "#008300",
    "GAD_7": "#d55181",
    "IUS_12": "#3949ab",
    "LOT_R": "#b5651d",
    "LSAS": "#00897b",
    "OCI": "#eb6834",
    "PHQ_8": "#c2185b",
    "SRDS": "#c98500",
    "STAI": "#6250d6",
    "STICSA": "#199e70",
}

VERSIONS = ["V1", "V2", "V3", "V4"]

VERSION_LABELS = {
    "V1": "Version 1 (Vol→Stab)",
    "V2": "Version 2 (Stab→Vol)",
    "V3": "Version 3 (Vol→Vol)",
    "V4": "Version 4 (Stab→Stab)",
}

# The 6 attention checks used to exclude participants. Question wording is
# written here in the form produced by ``normalise_wording`` (lower case, no
# final full stop), and answers in the form of the lower-cased ``responseLabel``.
# Each check passes on its two most correct answers: for a question that should
# be answered "not at all", "not at all" and the next answer up both pass.
# The IUS-12 check ("I feel surprised when surprising things happen to me") and
# the LOT-R check ("I enjoy it when good things happen to me") are deliberately
# not listed: failing them may reflect symptoms such as blunted emotions or
# anhedonia rather than inattention, so they are left out of the exclusion rule.
# The scoring key still marks them as attention checks, so their answers are
# never scored.
ATTENTION_CHECKS = [
    {
        "questionnaire": "STICSA Assessment",
        "question": "my hands have seven fingers",
        "valid_responses": ["not at all", "a little"],
    },
    {
        "questionnaire": "PHQ_8 Assessment",
        "question": "difficulty standing on one leg",
        "valid_responses": ["not at all", "several days"],
    },
    {
        "questionnaire": "GAD_7 Assessment",
        "question": "being scared of my own hair",
        "valid_responses": ["not at all", "several days"],
    },
    {
        "questionnaire": "STAI Assessment",
        "question": "i feel like i have always been a human being",
        "valid_responses": ["very much so", "moderately"],
    },
    {
        "questionnaire": "AES Assessment",
        "question": "i have no feet",
        "valid_responses": ["not at all", "slightly"],
    },
    {
        "questionnaire": "OCI Assessment",
        "question": "i frequently find myself cleaning my boat before going to bed",
        "valid_responses": ["not at all", "a little"],
    },
]

# The four roles an item can have in the scoring key. Only the first two are
# scored; fillers and attention checks are left out of every score.
SCORING_ROLES = ("forward", "reverse", "not_scored_filler", "attention_check")
SCORED_ROLES = ("forward", "reverse")
SCORING_KEY_COLUMNS = ["questionnaire_key", "question_id", "item_label", "scoring"]

FIXED_FIGURE_NAMES: list[str] = [
    "factor_loadings_all_questionnaires.svg",
    "questionnaire_correlation_matrix.svg",
    *(f"lr_change_vs_{q}.svg" for q in QUESTIONNAIRE_KEYS),
    "factor_scores_vs_lr_change.svg",
    "scree_plot.svg",
    "selected_factor_score_correlations.svg",
    "selected_factor_loadings.svg",
    "efa_factor_loadings.svg",
]
OPTIONAL_FIGURE_NAMES: list[str] = ["high_correlations.svg"]

# ── Factor-count rule and factor-analysis checks ───────────────────────────

# Parallel analysis compares the real eigenvalues with eigenvalues from this
# many random datasets, using the chosen percentile of each ordered eigenvalue.
PA_N_RANDOM_DATASETS = 1000
PA_PERCENTILE = 95
# Fixed seeds, so every run (and every learning-rate parameter) draws the same
# random datasets and the same split of participants.
PARALLEL_ANALYSIS_SEED = 20261006
SPLIT_HALF_SEED = 20261007
# factor_analyzer will not let an item's unique variance fall below 0.005. An
# item whose unique variance sits on that bound has a communality of 0.995 or
# more: an impossible, "Heywood", solution. The small margin below absorbs
# rounding in the optimiser.
HEYWOOD_COMMUNALITY = 0.995
HEYWOOD_ROUNDING_MARGIN = 1e-6
# Number of factors in the item-level EFA, chosen by looking at the scree plot
# and the 3-, 4- and 5-factor loadings, and confirmed after the questionnaire
# scoring and attention-check fixes (302 participants, 90 items). After the
# large first eigenvalue (29.9), the eigenvalues fall steeply
# through factor 4 (6.0, 5.3, 3.8) and then flatten from factor 5 (2.2, 2.1,
# 1.8, ...), so the elbow is after 4 factors. Both random halves of the
# participants reproduce all 4 factors (Tucker congruence 0.96 or more, except
# 0.94 for tolerance of uncertainty in one half). The halves reproduce 3 factors
# less well (lowest 0.84) and do not reproduce a 5-factor solution (lowest
# 0.56). The
# two automatic rules were not used:
# - CNG gave 3, its smallest possible answer. Its largest slope change sits at
#   the very first window only because of the big drop after the first
#   eigenvalue, not because of an elbow at 3.
# - Parallel analysis gave 6. With many items it tends to keep too many
#   factors.
# The factors are named after what a high score means (factor signs are
# arbitrary, so each name is chosen to match the direction of its scores):
# - ML1, negative affect: STICSA, OCI, BIS, STAI, GAD-7, PHQ-8 and SRDS items;
# - ML2, social anxiety: almost only the LSAS items;
# - ML3, apathy and amotivation: AES, SRDS and STAI items, with LOT-R loading
#   negatively;
# - ML4, tolerance of uncertainty: IUS-12 items loading negatively, plus some
#   AES and BIS items.
# The names fit the current order and signs of ML1-ML4. Factor order and sign
# can change when the data change, so `name_selected_factors` checks them
# before they are used.
SELECTED_N_FACTORS = 4
FACTOR_NAMES = {
    "ML1": "Negative affect",
    "ML2": "Social anxiety",
    "ML3": "Apathy and amotivation",
    "ML4": "Tolerance of uncertainty",
}
# The questionnaire that marks each named factor, and the sign its items load
# with. Summed over its items, this questionnaire loads more strongly on its
# own factor than on any other (on the current data: STICSA 7.0 on ML1, LSAS
# 11.0 on ML2, AES 4.0 on ML3 and IUS-12 -3.7 on ML4; none of their sums on
# the other factors is larger than 2.1 in absolute value).
FACTOR_MARKER_QUESTIONNAIRES = {
    "ML1": ("STICSA", 1),
    "ML2": ("LSAS", 1),
    "ML3": ("AES", 1),
    "ML4": ("IUS_12", -1),
}


def sig_stars(p_value: float) -> str:
    """Map a p-value to its conventional significance-star string.

    Args:
        p_value: A p-value (uncorrected or corrected).

    Returns:
        ``"***"`` for ``p < 0.001``, ``"**"`` for ``p < 0.01``, ``"*"`` for
        ``p < 0.05``, and ``""`` otherwise.
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


# ── Input loading ───────────────────────────────────────────────────────────


def load_lr_data(lr_csv: Path, model_name: str, lr_column: str) -> pd.DataFrame:
    """Load and validate the subject-level learning-rate parameters table.

    Args:
        lr_csv: Path to the compiled subject-parameters CSV.
        model_name: Model identifier expected in the ``model`` column
            (matched case-insensitively as a substring).
        lr_column: Name of the learning-rate column required by the
            requested parameter.

    Returns:
        The raw learning-rate parameters table, freshly read from disk.

    Raises:
        FileNotFoundError: If ``lr_csv`` does not exist.
        ValueError: If ``lr_column`` or the ``model`` column is missing from
            the table, if no rows match ``model_name``, or if the
            ``converged`` column is missing or does not hold only
            ``True``/``False`` values.
    """
    if not lr_csv.is_file():
        raise FileNotFoundError(f"Learning-rate parameters file not found: {lr_csv}")

    lr_data = pd.read_csv(lr_csv)

    if "model" not in lr_data.columns:
        raise ValueError(f"Column 'model' not found in {lr_csv}.")
    if lr_column not in lr_data.columns:
        raise ValueError(f"Column {lr_column!r} not found in {lr_csv}.")

    matched = lr_data[
        lr_data["model"].astype(str).str.contains(model_name, case=False, na=False)
    ]
    if matched.empty:
        raise ValueError(f"No rows found for model {model_name!r} in {lr_csv}.")

    # Each block's fit records whether its sampler converged. Participants
    # whose fits did not converge stay in the analysis; the flag is only used
    # to count them. A missing or blank flag would make that count wrong
    # without any sign, so it stops the run instead.
    if "converged" not in lr_data.columns:
        raise ValueError(f"Column 'converged' not found in {lr_csv}.")
    if not pd.api.types.is_bool_dtype(lr_data["converged"]):
        raise ValueError(
            f"Column 'converged' in {lr_csv} must hold only True/False values, "
            f"but it was read as {lr_data['converged'].dtype}."
        )

    return lr_data


def load_questionnaire_data(zip_path: Path) -> pd.DataFrame:
    """Load the questionnaire response data directly from its zip archive.

    Args:
        zip_path: Path to the ``questionnaire_data.csv.zip`` archive.

    Returns:
        The long-format questionnaire response data, freshly read from the
        archive (no unzipped copy is used or cached).

    Raises:
        FileNotFoundError: If ``zip_path`` does not exist.
        ValueError: If the archive does not contain exactly one CSV member,
            ignoring ``__MACOSX/`` entries.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Questionnaire archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.startswith("__MACOSX/") and name.lower().endswith(".csv")
        ]
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one CSV member in {zip_path}, found "
                f"{len(members)}: {members}"
            )
        with archive.open(members[0]) as handle:
            return pd.read_csv(handle)


# ── Scoring key and attention-check matching ────────────────────────────────


def load_scoring_key(key_csv: Path) -> pd.DataFrame:
    """Load the scoring key and check that it is well formed.

    The key says, for every questionnaire item, how the item is used: scored
    as it stands (``forward``), scored upside down (``reverse``), left out
    because it is not part of the scale (``not_scored_filler``), or left out
    because it is an attention check (``attention_check``).

    Args:
        key_csv: Path to ``config/questionnaire_scoring_key.csv``.

    Returns:
        The key, with the columns ``questionnaire_key``, ``question_id``,
        ``item_label`` and ``scoring``. ``item_label`` is a short label for a
        scale item and the full wording for an attention check. Every value
        is read as text.

    Raises:
        FileNotFoundError: If ``key_csv`` does not exist.
        ValueError: If the columns are not the expected four, if a ``scoring``
            value is not one of the four roles, or if an item appears twice.
    """
    if not key_csv.is_file():
        raise FileNotFoundError(f"Scoring key not found: {key_csv}")

    # Read everything as text, so that words such as "None" in an item's
    # label are never turned into missing values.
    scoring_key = pd.read_csv(key_csv, dtype=str, keep_default_na=False)

    if list(scoring_key.columns) != SCORING_KEY_COLUMNS:
        raise ValueError(
            f"{key_csv} must have the columns {SCORING_KEY_COLUMNS}, "
            f"found {list(scoring_key.columns)}."
        )
    unknown_roles = sorted(set(scoring_key["scoring"]) - set(SCORING_ROLES))
    if unknown_roles:
        raise ValueError(
            f"{key_csv} has scoring values {unknown_roles}; "
            f"each must be one of {list(SCORING_ROLES)}."
        )
    repeated = scoring_key[
        scoring_key.duplicated(["questionnaire_key", "question_id"], keep=False)
    ]
    if not repeated.empty:
        raise ValueError(
            f"{key_csv} lists these items more than once: "
            f"{sorted(set(repeated['question_id']))}."
        )
    return scoring_key


def normalise_wording(text: str) -> str:
    """Put a question's wording in the form used to compare wordings.

    The same question can be written with different spaces, capital letters
    or a final full stop in the data and in this script (five of the eight
    attention-check questions end in a full stop in the data, but the list of
    checks leaves it out). Comparing the normalised wording means these
    harmless differences cannot hide a question, while any real difference in
    wording is still found.

    Args:
        text: The wording of a question.

    Returns:
        The wording without spaces at either end, in lower case, and without
        one final full stop.
    """
    text = text.strip().lower()
    return text[:-1] if text.endswith(".") else text


def find_attention_check_rows(
    data_q: pd.DataFrame, check: dict[str, Any]
) -> pd.DataFrame:
    """Find every answer to one attention-check question.

    Args:
        data_q: Long-format questionnaire response data.
        check: One entry of ``ATTENTION_CHECKS``.

    Returns:
        The rows of ``data_q`` that belong to the check's questionnaire and
        whose normalised wording equals the check's normalised wording.
    """
    in_questionnaire = data_q["questionnaireTitle"].str.contains(
        check["questionnaire"], case=False, na=False, regex=False
    )
    wording = data_q["questionText"].astype(str).map(normalise_wording)
    return data_q[in_questionnaire & (wording == normalise_wording(check["question"]))]


def check_scoring_key_against_data(
    data_q: pd.DataFrame, scoring_key: pd.DataFrame, questionnaire_keys: list[str]
) -> None:
    """Stop the run unless the scoring key and the attention checks fit the data.

    These checks use every row of the 12 questionnaires, before anyone is
    excluded. A wrong key entry or an unfound attention check would quietly
    change every score or who is analysed, so each problem stops the run.

    Args:
        data_q: Raw long-format questionnaire response data.
        scoring_key: The key from :func:`load_scoring_key`.
        questionnaire_keys: The 12 questionnaire keys covered by the key.

    Raises:
        ValueError: If the items in the key and in the data differ, if an
            item's ``questionText`` differs from its ``item_label``, if an
            attention check matches other than
            one item, is not marked ``attention_check`` in the key, or was not
            answered by everyone who started its questionnaire, if a row the
            data marks as an attention check is not marked so in the key, or
            if a questionnaire's answer range varies or an answer is not a
            whole number within the range.
    """
    archive = data_q[data_q["questionnaireKey"].isin(questionnaire_keys)]

    # 1. The key and the data hold exactly the same items.
    items = archive[["questionnaireKey", "questionId"]].drop_duplicates()
    archive_items = set(zip(items["questionnaireKey"], items["questionId"]))
    key_items = set(zip(scoring_key["questionnaire_key"], scoring_key["question_id"]))
    if archive_items != key_items:
        raise ValueError(
            "The scoring key and the questionnaire data list different items. "
            f"In the data but not in the key: {sorted(archive_items - key_items)}. "
            f"In the key but not in the data: {sorted(key_items - archive_items)}."
        )

    # 2. Each item's text in the data is its label in the key. The archive's
    # questionText holds the key's labels (and the full wording for attention
    # checks). This step confirms that the archive was made from this key, so
    # no item is scored with a role from a different version of the key.
    data_wording = archive[
        ["questionnaireKey", "questionId", "questionText"]
    ].drop_duplicates()
    data_wording["data_wording"] = (
        data_wording["questionText"].astype(str).map(normalise_wording)
    )
    key_wording = scoring_key.assign(
        key_wording=scoring_key["item_label"].map(normalise_wording)
    )
    compared = data_wording.merge(
        key_wording,
        left_on=["questionnaireKey", "questionId"],
        right_on=["questionnaire_key", "question_id"],
    )
    different = compared[compared["data_wording"] != compared["key_wording"]]
    if not different.empty:
        raise ValueError(
            "The questionText of these items in the questionnaire data differs "
            "from their item_label in the scoring key: "
            f"{sorted(set(different['questionId']))}."
        )

    key_item_list = zip(scoring_key["questionnaire_key"], scoring_key["question_id"])
    roles = dict(zip(key_item_list, scoring_key["scoring"]))

    # 3. Every attention check used for exclusion is found, once, is marked as
    # an attention check in the key, and was answered by everyone who started
    # its questionnaire (a missing answer is never counted as a pass).
    for check in ATTENTION_CHECKS:
        check_rows = find_attention_check_rows(data_q, check)
        matched = check_rows[["questionnaireKey", "questionId"]].drop_duplicates()
        if len(matched) != 1:
            raise ValueError(
                f"Attention check {check['question']!r} ({check['questionnaire']}) "
                f"matched {len(matched)} questions; exactly 1 is required."
            )
        item = (matched["questionnaireKey"].iloc[0], matched["questionId"].iloc[0])
        if roles[item] != "attention_check":
            raise ValueError(
                f"Attention check {item[1]} is marked {roles[item]!r} in the scoring "
                "key; it must be 'attention_check'."
            )
        started = set(
            data_q.loc[
                data_q["questionnaireTitle"].str.contains(
                    check["questionnaire"], case=False, na=False, regex=False
                ),
                "assignedSubjectId",
            ]
        )
        unanswered = started - set(check_rows["assignedSubjectId"])
        if unanswered:
            raise ValueError(
                f"Attention check {item[1]} was not answered by {len(unanswered)} "
                f"participant(s) who started {check['questionnaire']}."
            )
        # Every passing answer must be one of the answers people gave to this
        # question. A mistyped passing answer would otherwise never match, so
        # everyone who chose that answer would quietly count as failing.
        answers_given = set(
            check_rows["responseLabel"].astype(str).str.strip().str.lower()
        )
        unknown_answers = sorted(set(check["valid_responses"]) - answers_given)
        if unknown_answers:
            raise ValueError(
                f"Attention check {item[1]} lists passing answers {unknown_answers} "
                "that nobody gave to this question; check their spelling."
            )

    # 4. Every row the data itself marks as an attention check is also one in
    # the key. (The data's own marking is not used for scoring, but a
    # disagreement would mean the key is wrong.)
    marked = archive[archive["isAttentionCheck"] == True]  # noqa: E712
    marked_items = marked[["questionnaireKey", "questionId"]].drop_duplicates()
    wrongly_keyed = sorted(
        question_id
        for questionnaire_key, question_id in zip(
            marked_items["questionnaireKey"], marked_items["questionId"]
        )
        if roles[(questionnaire_key, question_id)] != "attention_check"
    )
    if wrongly_keyed:
        raise ValueError(
            "The data marks these items as attention checks, but the scoring key "
            f"does not: {wrongly_keyed}."
        )

    # 5. The answer range is fixed for each questionnaire, and every answer is
    # a whole number within it. Reversing an item (min + max - raw) is only
    # correct under these conditions.
    ranges = archive.groupby("questionnaireKey")[["scaleMin", "scaleMax"]].nunique()
    varying = sorted(ranges.index[(ranges > 1).any(axis=1)])
    if varying:
        raise ValueError(
            f"scaleMin or scaleMax varies within questionnaires {varying}."
        )
    answers = pd.to_numeric(archive["rawResponse"], errors="coerce")
    valid = (
        (answers == np.floor(answers))
        & (answers >= archive["scaleMin"])
        & (answers <= archive["scaleMax"])
    )
    if not valid.all():
        raise ValueError(
            "rawResponse is not a whole number within scaleMin..scaleMax for items "
            f"{sorted(set(archive.loc[~valid, 'questionId']))}."
        )


def score_items(data_q: pd.DataFrame, scoring_key: pd.DataFrame) -> pd.DataFrame:
    """Score every answer to a ``forward`` or ``reverse`` item.

    This is the only scoring step. The questionnaire totals and the item-level
    factor analysis are both built from its output, so they always agree.

    A ``forward`` item scores its raw answer. A ``reverse`` item is worded the
    opposite way round (for example, the STAI item about feeling composed, on
    an anxiety scale), so its answer is flipped within the questionnaire's range:
    ``scaleMin + scaleMax - rawResponse``. On a 0-3 scale, the answer 3
    ("Very much so") then scores 0. The data file's own stored score and
    reverse-coding columns are never read. Items marked
    ``not_scored_filler`` or ``attention_check`` have no row in the output.

    Args:
        data_q: Long-format questionnaire response data.
        scoring_key: The key from :func:`load_scoring_key`.

    Returns:
        One row per answer, with the columns ``assignedSubjectId``,
        ``questionnaireKey``, ``questionId`` and ``itemScore``.
    """
    scored_key = scoring_key[scoring_key["scoring"].isin(SCORED_ROLES)]
    # An inner join keeps only the items the key says to score.
    answers = data_q.merge(
        scored_key,
        left_on=["questionnaireKey", "questionId"],
        right_on=["questionnaire_key", "question_id"],
        how="inner",
    )
    raw = answers["rawResponse"].astype(float)
    flipped = answers["scaleMin"] + answers["scaleMax"] - raw
    answers["itemScore"] = np.where(answers["scoring"] == "reverse", flipped, raw)
    return answers[["assignedSubjectId", "questionnaireKey", "questionId", "itemScore"]]


# ── Participant exclusion pipeline ──────────────────────────────────────────


def clean_and_match_data(
    lr_data: pd.DataFrame,
    data_q: pd.DataFrame,
    questionnaire_keys: list[str],
    scoring_key: pd.DataFrame,
) -> dict[str, Any]:
    """Match participants across fits and questionnaires, then apply exclusions.

    First, before anyone is excluded, the scoring key and the attention checks
    are checked against every row of the 12 questionnaires
    (:func:`check_scoring_key_against_data`). The steps then run in this
    order: matching learning-rate subjects to questionnaire subjects,
    identifier and answer normalisation, a stop if any matched participant has
    two answers to one item, the 6-check attention-check exclusion (failing
    more than 1 check), scoring of the items, and exclusion of participants
    who did not complete all 12 questionnaires. Each check finds its question
    with the wording rule in :func:`normalise_wording`. This is always a single
    fresh pass over the inputs passed in; no state is carried over from any
    other parameter's run.

    Args:
        lr_data: Raw subject-level learning-rate parameters.
        data_q: Raw long-format questionnaire response data.
        questionnaire_keys: The 12 questionnaire keys required for
            completion.
        scoring_key: The key from :func:`load_scoring_key`.

    Returns:
        A dict with the cleaned ``lr_data_new`` and ``data_q`` frames, the
        per-subject questionnaire-totals frame ``q_all`` (subjects who
        completed all of ``questionnaire_keys``), the per-answer item scores
        ``item_scores`` for the same subjects (from :func:`score_items`, the
        input to both ``q_all`` and the item-level factor analysis), and the
        scalar counts
        ``num_ppts_with_any_questionnaire`` (a row for any of the 12
        questionnaires; this is not the number who completed all 12),
        ``num_ppts_completed_all_questionnaires`` (the number who completed
        all 12; the same value as ``num_ppts_Q_cleaned`` below),
        ``num_ppts_LR_cleaned``, ``num_ppts_Q_cleaned`` and
        ``num_ppts_failed_more_than_one_check`` (excluded by the 6-check
        rule, counted among participants matched to the fits).

    Raises:
        ValueError: If the scoring key or an attention check does not fit the
            data (see :func:`check_scoring_key_against_data`), or if a matched
            participant has more than one row for the same item.
    """
    # Checked on the whole archive, so that a problem is found even in items
    # that only excluded participants answered.
    check_scoring_key_against_data(data_q, scoring_key, questionnaire_keys)

    data_q = data_q.copy()
    lr_data_new = lr_data.copy()

    # Match LR data to questionnaire data.
    matching_subject_ids = data_q["assignedSubjectId"].unique()
    lr_data_new = lr_data_new[lr_data_new["subject_id"].isin(matching_subject_ids)]

    # Normalise responseLabel and identifiers in both datasets. (Question
    # wording is normalised where it is compared, by `normalise_wording`.)
    data_q["responseLabel"] = data_q["responseLabel"].str.strip().str.lower()
    data_q["assignedSubjectId"] = data_q["assignedSubjectId"].str.strip().str.lower()
    lr_data_new["subject_id"] = lr_data_new["subject_id"].str.strip().str.lower()

    # Keep only IDs present in both datasets.
    matching_ids = set(data_q["assignedSubjectId"]).intersection(
        set(lr_data_new["subject_id"])
    )
    data_q = data_q[data_q["assignedSubjectId"].isin(matching_ids)]
    lr_data_new = lr_data_new[lr_data_new["subject_id"].isin(matching_ids)]

    # Stop if a participant has two answers to one item. Two answers to an
    # attention check could pass and fail at once, and two answers to a scored
    # item would be counted twice. The rows are never summed or averaged.
    repeated = data_q[
        data_q.duplicated(
            ["assignedSubjectId", "questionnaireKey", "questionId"], keep=False
        )
    ]
    if not repeated.empty:
        first = repeated.iloc[0]
        raise ValueError(
            f"{repeated['assignedSubjectId'].nunique()} participant(s) have more "
            "than one answer to the same item, for example participant "
            f"{first['assignedSubjectId']!r}, item {first['questionId']!r}."
        )

    # Exclude participants who fail more than 1 of the 6 attention checks.
    # `check_scoring_key_against_data` has already shown that every check is
    # found and was answered by everyone who started its questionnaire, so a
    # missing answer cannot be mistaken for a pass here.
    failed_checks: dict[str, list[dict[str, str]]] = {}
    for check in ATTENTION_CHECKS:
        check_data = find_attention_check_rows(data_q, check)
        failed_subjects = check_data[
            ~check_data["responseLabel"].isin(check["valid_responses"])
        ]
        for _, row in failed_subjects.iterrows():
            failed_checks.setdefault(row["assignedSubjectId"], []).append(
                {"question": check["question"], "response": row["responseLabel"]}
            )

    failed_checks_df = pd.DataFrame(
        [
            {
                "assignedSubjectId": subject,
                "failed_checks": checks,
                "num_failed": len(checks),
            }
            for subject, checks in failed_checks.items()
        ],
        columns=["assignedSubjectId", "failed_checks", "num_failed"],
    )
    multiple_failures = failed_checks_df[failed_checks_df["num_failed"] > 1]
    failed_more_than_one_ids = set(multiple_failures["assignedSubjectId"])

    participants_to_remove = sorted(failed_more_than_one_ids)
    lr_data_new = lr_data_new[~lr_data_new["subject_id"].isin(participants_to_remove)]
    data_q = data_q[~data_q["assignedSubjectId"].isin(participants_to_remove)]

    # Participants with a row for *any* of the 12 questionnaires (not
    # participants who completed all 12 -- see num_ppts_completed_all_
    # questionnaires below for that), measured on the data state right
    # after the attention-check exclusions and before the incomplete-
    # questionnaire exclusion below. It must not be reported as the number
    # who completed all 12 questionnaires.
    num_ppts_with_any_questionnaire = int(
        data_q[data_q["questionnaireKey"].isin(questionnaire_keys)][
            "assignedSubjectId"
        ].nunique()
    )

    # Score every answer once, from the scoring key. Reverse items are flipped,
    # and fillers and attention checks drop out, so the totals below and the
    # item-level factor analysis later use exactly the same item scores.
    item_scores = score_items(data_q, scoring_key)

    # Per-subject total score for each of the 12 questionnaires.
    q_scores = {}
    for q in questionnaire_keys:
        sub = item_scores[item_scores["questionnaireKey"] == q]
        totals = sub.groupby("assignedSubjectId")["itemScore"].sum().reset_index()
        totals = totals.rename(columns={"itemScore": q})
        q_scores[q] = totals

    q_all = q_scores[questionnaire_keys[0]]
    for q in questionnaire_keys[1:]:
        q_all = pd.merge(q_all, q_scores[q], on="assignedSubjectId", how="inner")

    # Exclude participants who did not complete all 12 questionnaires.
    incomplete_ppts = data_q[
        ~data_q["assignedSubjectId"].isin(q_all["assignedSubjectId"])
    ]["assignedSubjectId"].unique()
    participants_to_remove = incomplete_ppts.tolist()
    lr_data_new = lr_data_new[~lr_data_new["subject_id"].isin(participants_to_remove)]
    data_q = data_q[~data_q["assignedSubjectId"].isin(participants_to_remove)]
    item_scores = item_scores[
        ~item_scores["assignedSubjectId"].isin(participants_to_remove)
    ]

    # Participants who completed all 12 questionnaires: the count after the
    # incomplete-questionnaire exclusion above has removed everyone who did
    # not. Numerically the same value as num_ppts_Q_cleaned below (both are
    # `data_q`'s final subject count); exposed under this explicit name so
    # it is reported next to num_ppts_with_any_questionnaire without either
    # being mistaken for the other.
    num_ppts_completed_all_questionnaires = int(data_q["assignedSubjectId"].nunique())

    return {
        "lr_data_new": lr_data_new,
        "data_q": data_q,
        "q_all": q_all,
        "item_scores": item_scores,
        "num_ppts_with_any_questionnaire": num_ppts_with_any_questionnaire,
        "num_ppts_completed_all_questionnaires": num_ppts_completed_all_questionnaires,
        "num_ppts_LR_cleaned": int(lr_data_new["subject_id"].nunique()),
        "num_ppts_Q_cleaned": int(data_q["assignedSubjectId"].nunique()),
        "num_ppts_failed_more_than_one_check": len(failed_more_than_one_ids),
    }


# ── Factor models with validity checks ──────────────────────────────────────


def fit_factor_model_with_checks(
    data: Any, n_factors: int, fit_name: str
) -> FactorAnalyzer:
    """Fit one oblimin/ML factor model and stop if the solution is not valid.

    Every factor model in this script is fitted through this function. A
    factor model can "finish" without being usable, and ``factor_analyzer``
    does not stop in either case below, so both are checked here:

    - The optimiser can fail to converge. ``factor_analyzer`` then only
      issues a "Failed to converge" warning, and the loadings it returns
      are not the maximum-likelihood solution.
    - An item can get a communality (the share of its variance that the
      factors explain) of 0.995 or more. That means its unique variance sits
      on the lower bound of 0.005, which is an impossible (Heywood) solution.

    Args:
        data: Participants-by-items table or array for the model.
        n_factors: Number of factors to extract.
        fit_name: Plain-English name of this fit, used in the error message,
            for example ``"item-level EFA, 4 factors, split half B"``.

    Returns:
        The fitted ``FactorAnalyzer`` (oblimin rotation, maximum likelihood).

    Raises:
        ValueError: If the optimiser did not converge, or if any
            communality is 0.995 or more.
    """
    model = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="ml")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(data)

    convergence_failures = []
    for caught in caught_warnings:
        if str(caught.message).startswith("Failed to converge"):
            convergence_failures.append(str(caught.message))
        else:
            # Other warnings (such as "no rotation with 1 factor") are not
            # failures, so they are passed on unchanged.
            warnings.warn_explicit(
                caught.message, caught.category, caught.filename, caught.lineno
            )
    if convergence_failures:
        raise ValueError(
            f"{fit_name}: the factor model did not converge "
            f"({convergence_failures[0]}). The loadings cannot be used."
        )

    communalities = pd.Series(
        calculate_oblique_communalities(model), index=_item_names(data)
    )
    # A missing (NaN) loading would slip past the Heywood comparison below,
    # because any comparison with NaN is false, so check for it first.
    if not np.all(np.isfinite(model.loadings_)):
        raise ValueError(
            f"{fit_name}: the factor model returned missing or infinite "
            "loadings. The loadings cannot be used."
        )
    heywood_items = communalities[
        communalities >= HEYWOOD_COMMUNALITY - HEYWOOD_ROUNDING_MARGIN
    ]
    if len(heywood_items) > 0:
        raise ValueError(
            f"{fit_name}: Heywood case. {len(heywood_items)} item(s) have a "
            f"communality of {HEYWOOD_COMMUNALITY} or more, so their unique "
            "variance has hit the lower bound of 0.005: "
            f"{heywood_items.round(4).to_dict()}"
        )
    return model


def _item_names(data: Any) -> list[Any]:
    """Name the columns of a table, or number the columns of an array.

    Args:
        data: Participants-by-items table or array.

    Returns:
        The column names for a table, or ``0, 1, 2, ...`` for an array.
    """
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    return list(range(np.asarray(data).shape[1]))


def calculate_oblique_communalities(model: FactorAnalyzer) -> np.ndarray:
    """Calculate each item's communality from a fitted oblimin solution.

    The communality is the share of an item's variance explained by all
    factors together. With correlated (oblique) factors it is the diagonal of
    ``L Phi L^T``, where ``L`` holds the loadings and ``Phi`` the factor
    correlations. ``FactorAnalyzer.get_communalities()`` ignores ``Phi``, so
    it gives the wrong answer here and is not used.

    ``factor_analyzer`` sorts the factors after rotation but leaves ``phi_`` in
    the old order, so ``phi_`` cannot be combined with ``loadings_`` directly.
    ``structure_`` (= ``L Phi``) is sorted together with the loadings, and the
    communality is the sum over factors of loading times structure.

    Args:
        model: A fitted ``FactorAnalyzer``.

    Returns:
        One communality per item.
    """
    loadings = model.loadings_
    if model.structure_ is None:
        # One factor, or uncorrelated factors: Phi is the identity matrix.
        return np.sum(loadings**2, axis=1)
    return np.sum(loadings * model.structure_, axis=1)


# ── Factor analysis of questionnaire totals ─────────────────────────────────


def fit_kaiser_factor_analysis(
    q_all: pd.DataFrame, questionnaire_keys: list[str]
) -> tuple[np.ndarray, int, pd.DataFrame, np.ndarray, list[str]]:
    """Fit the initial factor analysis on standardised questionnaire totals.

    Args:
        q_all: Per-subject totals for each of ``questionnaire_keys``.
        questionnaire_keys: The 12 questionnaire keys, in column order.

    Returns:
        A tuple ``(eigenvalues, n_factors, loadings, factor_scores,
        factor_cols)``: the sorted descending eigenvalues of the
        standardised totals' correlation matrix; the Kaiser-criterion factor
        count (eigenvalues > 1); the oblimin/ML factor loadings indexed by
        questionnaire key; the per-subject factor scores; and the
        ``Factor_i`` column names.

    Raises:
        ValueError: If the factor model does not converge or has a Heywood
            case (see :func:`fit_factor_model_with_checks`).
    """
    scaler = StandardScaler()
    q_scaled = scaler.fit_transform(q_all[questionnaire_keys])

    corr_matrix = np.corrcoef(q_scaled, rowvar=False)
    eigenvalues, _ = eigh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    n_factors = int(np.sum(eigenvalues > 1))

    fa = fit_factor_model_with_checks(
        q_scaled, n_factors, f"questionnaire-totals EFA, {n_factors} factors"
    )

    factor_cols = [f"Factor_{i + 1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=questionnaire_keys, columns=factor_cols)
    factor_scores = fa.transform(q_scaled)

    return eigenvalues, n_factors, loadings, factor_scores, factor_cols


def build_logit_lr_frame(
    lr_data_new: pd.DataFrame, model_name: str, lr_column: str
) -> pd.DataFrame:
    """Select the modelled rows and add the logit-transformed learning rate.

    Args:
        lr_data_new: Cleaned subject-level learning-rate parameters.
        model_name: Model identifier to match in the ``model`` column.
        lr_column: Learning-rate column to logit-transform.

    Returns:
        The subset of rows for ``model_name``, with an added ``lr_logit``
        column.
    """
    lr_rw = lr_data_new[
        lr_data_new["model"].astype(str).str.contains(model_name, case=False, na=False)
    ].copy()
    lr_rw["lr_logit"] = np.log(lr_rw[lr_column] / (1 - lr_rw[lr_column]))
    return lr_rw


def build_lr_pivot(lr_rw: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-subject logit LR to one row per (subject, version) change.

    Every participant must have exactly one row for block 1 and one for
    block 2 of their task version, each with a finite logit learning rate.
    Nobody is dropped: a missing, repeated or non-finite block stops the run.
    Participants whose fits did not converge are kept and flagged.

    Args:
        lr_rw: Modelled LR rows with ``lr_logit`` and ``converged`` columns
            (one row per block).

    Returns:
        One row per subject/version with ``lr_block1``, ``lr_block2``,
        ``lr_change`` (block 2 minus block 1) and ``both_blocks_converged``
        (``True`` only if the fits of both blocks converged), sorted by
        subject and version.

    Raises:
        ValueError: If a row has a block number other than 1 or 2, if a
            participant/version has more than one row for a block, if it lacks
            block 1 or block 2, or if a logit learning rate is not finite. The
            message names the first participant affected.
    """
    pair = ["subject_id", "version"]

    unexpected = lr_rw[~lr_rw["block_number"].isin([1, 2])]
    if not unexpected.empty:
        first = unexpected.iloc[0]
        raise ValueError(
            f"Participant {first['subject_id']!r} ({first['version']}) has block "
            f"number {first['block_number']!r}; only blocks 1 and 2 are expected."
        )

    # A repeated block would make the change score depend on which row came
    # first, so it stops the run instead of one row being picked.
    repeated = lr_rw[lr_rw.duplicated(pair + ["block_number"], keep=False)]
    if not repeated.empty:
        first = repeated.iloc[0]
        raise ValueError(
            f"{repeated[pair].drop_duplicates().shape[0]} participant(s) have more "
            f"than one row for the same block, for example participant "
            f"{first['subject_id']!r} ({first['version']}), block "
            f"{first['block_number']}."
        )

    # Without both blocks there is no change score, and quietly leaving the
    # participant out would shrink the sample without any record.
    blocks_present = lr_rw.groupby(pair)["block_number"].nunique()
    incomplete = blocks_present[blocks_present != 2]
    if not incomplete.empty:
        subject_id, version = incomplete.index[0]
        raise ValueError(
            f"{len(incomplete)} participant(s) do not have both block 1 and "
            f"block 2, for example participant {subject_id!r} ({version})."
        )

    # A learning rate of exactly 0 or 1, or a missing one, has no finite
    # logit and would break every regression that uses it.
    not_finite = lr_rw[~np.isfinite(lr_rw["lr_logit"].astype(float))]
    if not not_finite.empty:
        first = not_finite.iloc[0]
        raise ValueError(
            f"{len(not_finite)} block(s) have a logit learning rate that is not "
            f"finite, for example participant {first['subject_id']!r} "
            f"({first['version']}), block {first['block_number']}."
        )

    block1 = lr_rw[lr_rw["block_number"] == 1].set_index(pair)
    block2 = lr_rw[lr_rw["block_number"] == 2].set_index(pair).reindex(block1.index)
    lr_pivot = pd.DataFrame(
        {
            "lr_block1": block1["lr_logit"],
            "lr_block2": block2["lr_logit"],
        }
    )
    lr_pivot["lr_change"] = lr_pivot["lr_block2"] - lr_pivot["lr_block1"]
    # Kept in the analysis either way; the flag is only used for counting.
    lr_pivot["both_blocks_converged"] = block1["converged"] & block2["converged"]
    return lr_pivot.reset_index().sort_values(pair, ignore_index=True)


def check_version_sample_size(
    data_v: pd.DataFrame, version: str, n_predictors: int, regression_name: str
) -> None:
    """Stop the run if a task version has too few participants for its model.

    An OLS model with ``n_predictors`` predictors and an intercept needs at
    least ``n_predictors + 2`` participants to leave one degree of freedom
    for the residual error: 3 for a single predictor, and k + 2 for k
    factors. A version that is too small is an error, never a skipped row.

    Args:
        data_v: The participants of one task version.
        version: The task version, for the error message.
        n_predictors: Number of predictors in the model, not counting the
            intercept.
        regression_name: Which regression is being fitted, for the error
            message.

    Raises:
        ValueError: If ``data_v`` has fewer than ``n_predictors + 2`` rows.
    """
    minimum = n_predictors + 2
    if len(data_v) < minimum:
        raise ValueError(
            f"{regression_name}: version {version} has {len(data_v)} "
            f"participant(s), but its model needs at least {minimum}."
        )


def add_fdr_correction(
    results: pd.DataFrame, corrected_rows: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values over one table's whole family of tests.

    Benjamini-Hochberg (BH) controls the false discovery rate: among the
    tests called significant, the expected share of false positives is at
    most 5% at ``p_fdr < 0.05``. This holds when the tests are independent
    or positively related, as is usual for regressions that share one
    outcome. The family is every corrected row of the
    table together, so in a factor table it is every factor in every task
    version, not each factor or each version on its own.

    Args:
        results: A regression results table with a ``p`` column.
        corrected_rows: Boolean mask of the rows that make up the family.
            ``None`` means every row. Rows outside the family get an empty
            ``p_fdr`` and ``Sig_fdr``.

    Returns:
        A copy of ``results`` with ``p_fdr`` (the BH-adjusted p-value, or
        q-value) and ``Sig_fdr`` (:func:`sig_stars` of ``p_fdr``) added.

    Raises:
        ValueError: If the family is empty or holds a p-value that is not
            finite.
    """
    results = results.copy()
    if corrected_rows is None:
        corrected_rows = pd.Series(True, index=results.index)
    p_values = results.loc[corrected_rows, "p"].astype(float)
    if p_values.empty:
        raise ValueError("No p-values to correct: the family of tests is empty.")
    if not np.isfinite(p_values).all():
        raise ValueError(
            f"{int((~np.isfinite(p_values)).sum())} p-value(s) in the family are "
            "not finite, so the correction cannot be applied."
        )

    results["p_fdr"] = np.nan
    results.loc[corrected_rows, "p_fdr"] = multipletests(p_values, method="fdr_bh")[1]
    results["Sig_fdr"] = ""
    results.loc[corrected_rows, "Sig_fdr"] = results.loc[corrected_rows, "p_fdr"].apply(
        sig_stars
    )
    return results


def build_indiv_df(
    lr_pivot: pd.DataFrame, q_all: pd.DataFrame, questionnaire_keys: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit per-questionnaire, per-version univariate regressions on LR change.

    Args:
        lr_pivot: Per-subject/version LR change, from :func:`build_lr_pivot`.
        q_all: Per-subject questionnaire totals.
        questionnaire_keys: The 12 questionnaire keys.

    Returns:
        A tuple ``(indiv_df, lr_q)``: the per-(questionnaire, version)
        regression results, with Benjamini-Hochberg q-values over all 12
        questionnaires x 4 versions, and the merged LR-change/
        questionnaire-totals frame used to fit them (also needed for
        demographics and plotting).

    Raises:
        ValueError: If a version has fewer than 3 participants.
    """
    lr_q = pd.merge(
        lr_pivot,
        q_all[["assignedSubjectId"] + questionnaire_keys],
        left_on="subject_id",
        right_on="assignedSubjectId",
        how="inner",
    )

    rows = []
    for q in questionnaire_keys:
        for v in VERSIONS:
            data_v = lr_q[lr_q["version"].astype(str) == v]
            check_version_sample_size(data_v, v, 1, f"Questionnaire {q}")
            X = sm.add_constant(data_v[q])
            y = data_v["lr_change"]
            m = sm.OLS(y, X).fit()
            rows.append(
                {
                    "Questionnaire": Q_LABELS.get(q, q),
                    "q_key": q,
                    "Version": v,
                    "N": len(data_v),
                    "β": m.params.iloc[1],
                    "SE": m.bse.iloc[1],
                    "t": m.tvalues.iloc[1],
                    "p": m.pvalues.iloc[1],
                    "R²": m.rsquared,
                }
            )

    indiv_df = add_fdr_correction(pd.DataFrame(rows))

    return indiv_df, lr_q


def compute_demographics(lr_q: pd.DataFrame, data_q: pd.DataFrame) -> dict[str, float]:
    """Compute demographics for subjects present in the LR/questionnaire merge.

    Args:
        lr_q: Merged LR-change/questionnaire-totals frame, from
            :func:`build_indiv_df`.
        data_q: Cleaned questionnaire response data.

    Returns:
        A dict with ``num_male``, ``num_female`` (counts), and ``mean_age``,
        ``std_age`` (years).
    """
    unique_ids = lr_q["subject_id"].unique()
    unique_info = data_q.drop_duplicates(subset="assignedSubjectId")
    unique_info = unique_info[unique_info["assignedSubjectId"].isin(unique_ids)]

    return {
        "num_male": int((unique_info["sex"].str.lower() == "male").sum()),
        "num_female": int((unique_info["sex"].str.lower() == "female").sum()),
        "mean_age": float(unique_info["age"].mean()),
        "std_age": float(unique_info["age"].std()),
    }


def build_factor_results_df(
    lr_pivot: pd.DataFrame, q_all: pd.DataFrame, factor_cols: list[str]
) -> pd.DataFrame:
    """Fit per-factor, per-version univariate regressions on LR change.

    Args:
        lr_pivot: Per-subject/version LR change, from :func:`build_lr_pivot`.
        q_all: Per-subject questionnaire totals, including the Kaiser
            factor-score columns.
        factor_cols: The Kaiser ``Factor_i`` column names.

    Returns:
        Per-(factor, version) regression results, with Benjamini-Hochberg
        q-values over all factors x 4 versions.

    Raises:
        ValueError: If a version has fewer than 3 participants.
    """
    lr_factors = pd.merge(
        lr_pivot,
        q_all[["assignedSubjectId"] + factor_cols],
        left_on="subject_id",
        right_on="assignedSubjectId",
        how="inner",
    )

    rows = []
    for version in VERSIONS:
        data_v = lr_factors[lr_factors["version"].astype(str) == version]
        check_version_sample_size(data_v, version, 1, "Totals factor regression")
        for factor in factor_cols:
            X = sm.add_constant(data_v[factor])
            y = data_v["lr_change"]
            model = sm.OLS(y, X).fit()
            rows.append(
                {
                    "Version": version,
                    "Factor": factor,
                    "N": len(data_v),
                    "β": model.params[factor],
                    "SE": model.bse[factor],
                    "t": model.tvalues[factor],
                    "p": model.pvalues[factor],
                    "R²": model.rsquared,
                    "Sig": sig_stars(model.pvalues[factor]),
                }
            )

    factor_results_df = add_fdr_correction(pd.DataFrame(rows))

    return factor_results_df


# ── Updated exploratory factor analysis (item level) ────────────────────────


def prepare_item_level_data(
    item_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Pivot the item scores to a participants-by-items table.

    The item scores come from :func:`score_items`, the same scores that make
    up the questionnaire totals, so reverse items are already flipped and
    fillers and attention checks are already absent. Zero-variance items are
    then dropped.

    Args:
        item_scores: Item scores of the cleaned participants, with the columns
            ``assignedSubjectId``, ``questionId`` and ``itemScore``.

    Returns:
        A tuple ``(updated_q_data, updated_subject_ids)``: subjects-by-items
        wide table, and the subject IDs in row order (needed later to
        re-attach ``assignedSubjectId`` to the factor scores).

    Raises:
        ValueError: If a participant has two scores for one item (pandas
            refuses to pivot repeated entries).
    """
    updated_q_data = item_scores.pivot(
        index="assignedSubjectId", columns="questionId", values="itemScore"
    )

    updated_subject_ids = updated_q_data.index.to_series().reset_index(drop=True)

    zero_variance_items = updated_q_data.columns[
        updated_q_data.nunique(dropna=True) <= 1
    ].tolist()
    updated_q_data = updated_q_data.drop(columns=zero_variance_items)

    return updated_q_data, updated_subject_ids


def filter_skewed_items(
    updated_q_data: pd.DataFrame,
    scoring_key: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop items with ``|skewness| >= 2`` and list the items that were dropped.

    Strongly skewed items are removed before the factor analysis. Clinical
    questionnaires such as the OCI and EAT are often skewed in non-clinical
    samples, so the removed items are listed to show which questionnaires lose
    items.

    Args:
        updated_q_data: Subjects-by-items wide table.
        scoring_key: The questionnaire scoring key, used to look up which
            questionnaire each removed item belongs to.

    Returns:
        A tuple ``(kept_data, removed_items)``: the table restricted to items
        with ``|skewness| < 2``, and a table with one row per removed item and
        the columns ``item``, ``questionnaire`` (from the scoring key) and
        ``skew``.
    """
    skewness = updated_q_data.apply(lambda col: skew(col, nan_policy="omit"))
    is_kept = abs(skewness) < 2
    removed_names = skewness.index[~is_kept]
    questionnaire_of_item = scoring_key.set_index("question_id")["questionnaire_key"]
    removed_items = pd.DataFrame(
        {
            "item": removed_names,
            # Look the questionnaire up in the key rather than cutting the item
            # name, because not every item name has the same form (the EAT
            # items are "EAT_51" and "EAT_52", with a single underscore).
            "questionnaire": [questionnaire_of_item[name] for name in removed_names],
            "skew": skewness[removed_names].to_numpy(),
        }
    )
    return updated_q_data.loc[:, is_kept], removed_items


def remove_high_correlation_items(
    updated_q_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Iteratively remove items involved in a pairwise correlation above 0.7.

    At each step, the item involved in the most remaining high-correlation
    pairs is removed, until no pair with |correlation| > 0.7 remains.

    Args:
        updated_q_data: Subjects-by-items wide table (post skew filtering).

    Returns:
        A tuple ``(cleaned_data, high_corr_pairs_df, removed_items)``: the
        table with the offending items dropped, the table of all
        high-correlation item pairs found before removal, and the list of
        removed item names in removal order.
    """
    cormat = updated_q_data[sorted(updated_q_data.columns)].corr().round(2)
    np.fill_diagonal(cormat.values, 0)
    abs_cormat = abs(cormat)

    high_corr_pairs: list[tuple[str, str, float]] = []
    for i, item_a in enumerate(abs_cormat.columns):
        for item_b in abs_cormat.columns[i + 1 :]:
            corr_value = abs_cormat.loc[item_a, item_b]
            if corr_value > 0.7:
                high_corr_pairs.append((item_a, item_b, corr_value))

    high_corr_pairs_df = pd.DataFrame(
        high_corr_pairs, columns=["item_a", "item_b", "abs_correlation"]
    ).sort_values("abs_correlation", ascending=False)

    removed_items: list[str] = []
    remaining_pairs = list(high_corr_pairs)
    while remaining_pairs:
        pair_counts = pd.Series(
            [item for pair in remaining_pairs for item in pair[:2]]
        ).value_counts()
        item_to_remove = pair_counts.index[0]
        removed_items.append(item_to_remove)
        remaining_pairs = [
            pair for pair in remaining_pairs if item_to_remove not in pair[:2]
        ]

    cleaned_data = updated_q_data.drop(columns=removed_items)
    return cleaned_data, high_corr_pairs_df, removed_items


def impute_missing_items(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Replace each missing questionnaire answer with that item's mean.

    Args:
        cleaned_data: Subjects-by-items wide table (post correlation
            filtering), numeric with possible missing values.

    Returns:
        The same table with every missing value replaced by the mean of its
        item (column).
    """
    cleaned_data = cleaned_data.apply(pd.to_numeric, errors="coerce")
    return cleaned_data.fillna(cleaned_data.mean())


def calculate_item_eigenvalues(item_data: pd.DataFrame) -> np.ndarray:
    """Calculate the eigenvalues of the items' correlation matrix.

    An eigenvalue says how much of the total variance one principal component
    of the items carries. A matrix of ``p`` standardised items has total
    variance ``p``, so a large first eigenvalue means a few strong components
    run through the items. These are the values plotted in the scree plot and
    used by parallel analysis and CNG. They come straight from the correlation
    matrix and not from a factor model.

    Args:
        item_data: Mean-imputed subjects-by-items table.

    Returns:
        The eigenvalues, one per item, in descending order.
    """
    correlation_matrix = np.corrcoef(item_data.to_numpy(dtype=float), rowvar=False)
    return np.sort(np.linalg.eigvalsh(correlation_matrix))[::-1]


def calculate_parallel_analysis_thresholds(
    n_participants: int, n_items: int
) -> np.ndarray:
    """Find the eigenvalues that pure noise would produce (parallel analysis).

    Even items that share nothing have a first eigenvalue above 1 in a finite
    sample, because chance correlations make some components larger than
    others. Parallel analysis measures that chance level: it makes many
    random datasets of the same size (same number of participants and items,
    but independent standard-normal values), takes the correlation-matrix
    eigenvalues of each, and keeps the 95th percentile of the first, second,
    third... eigenvalue. A real eigenvalue is only worth keeping if it beats
    this level.

    Args:
        n_participants: Number of participants (rows) in the real data.
        n_items: Number of items (columns) in the real data.

    Returns:
        The 95th percentile of each ordered eigenvalue over
        ``PA_N_RANDOM_DATASETS`` random datasets, in descending order.
    """
    # A fixed seed means every run draws the same random datasets.
    rng = np.random.default_rng(PARALLEL_ANALYSIS_SEED)
    random_eigenvalues = np.empty((PA_N_RANDOM_DATASETS, n_items))
    for dataset in range(PA_N_RANDOM_DATASETS):
        random_data = rng.standard_normal((n_participants, n_items))
        correlation_matrix = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[dataset] = np.linalg.eigvalsh(correlation_matrix)[::-1]
    return np.percentile(random_eigenvalues, PA_PERCENTILE, axis=0)


def calculate_cng_slope_changes(eigenvalues: np.ndarray) -> np.ndarray:
    """Measure how sharply the scree plot flattens (Cattell-Nelson-Gorsuch).

    A scree plot drops steeply over the real factors and then flattens into a
    gentle slope over the noise. The CNG test looks for the place where the
    slope changes most. It follows ``nCng`` in the R package ``nFactors``.
    With eigenvalues ``x[1..n]`` and positions 1..n, for each ``i`` from 1 to
    ``n - 5`` it takes the least-squares slope of the three eigenvalues
    ``x[i+3..i+5]`` and subtracts the least-squares slope of the three
    eigenvalues ``x[i..i+2]``. Slopes are negative, so a large positive
    difference means a steep stretch followed by a flat one.

    Args:
        eigenvalues: Eigenvalues in descending order, at least 6 of them.

    Returns:
        An array of ``n - 5`` slope changes. Entry ``i - 1`` belongs to
        position ``i``.

    Raises:
        ValueError: If there are fewer than 6 eigenvalues.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if len(eigenvalues) < 6:
        raise ValueError(
            f"CNG needs at least 6 eigenvalues, but only {len(eigenvalues)} "
            "were given."
        )
    # Least-squares slope of three points at positions 1, 2, 3 (any three
    # consecutive positions give the same weights once centred).
    centred_positions = np.array([-1.0, 0.0, 1.0])
    window_slopes = np.array(
        [
            centred_positions @ eigenvalues[start : start + 3]
            for start in range(len(eigenvalues) - 2)
        ]
    )
    window_slopes = window_slopes / (centred_positions @ centred_positions)
    return window_slopes[3:] - window_slopes[:-3]


def calculate_cng_count(eigenvalues: np.ndarray) -> int:
    """Count the factors the CNG test supports.

    The count is ``i + 2``, where ``i`` is the (1-based) position of the
    largest slope change (see :func:`calculate_cng_slope_changes`): the last
    eigenvalue of the steep stretch before the scree plot flattens. For the
    eigenvalues 6, 4, 2, 1, 0.9, 0.8, 0.7, 0.6 the slope changes are 1.9, 1.4
    and 0.45, the largest is at ``i = 1``, and the count is 3.

    Args:
        eigenvalues: Eigenvalues in descending order, at least 6 of them.

    Returns:
        The CNG factor count.

    Raises:
        ValueError: If there are fewer than 6 eigenvalues, or if the largest
            slope change is shared by more than one position (``nCng`` would
            then return several counts).
    """
    slope_changes = calculate_cng_slope_changes(eigenvalues)
    is_largest = np.isclose(slope_changes, slope_changes.max(), rtol=0, atol=1e-12)
    if is_largest.sum() > 1:
        raise ValueError(
            "CNG slope changes have more than one largest value, so the "
            f"factor count is not unique: positions {np.flatnonzero(is_largest) + 1}."
        )
    return int(np.argmax(slope_changes)) + 3


def calculate_factor_count_rules(
    eigenvalues: np.ndarray, parallel_analysis_thresholds: np.ndarray
) -> dict[str, int]:
    """Work out the factor counts that two common rules would give.

    Parallel analysis keeps the leading eigenvalues that beat what noise
    produces. CNG finds the scree plot's elbow from changes in slope. Neither
    decides the count used in the analysis, which is fixed at
    ``SELECTED_N_FACTORS``; they are reported so a reader can see how that
    choice compares.

    Args:
        eigenvalues: Observed eigenvalues in descending order.
        parallel_analysis_thresholds: 95th percentile of the noise
            eigenvalues, in descending order, one per observed eigenvalue.

    Returns:
        A dict with ``pa_count`` and ``cng_count``.
    """
    # Count eigenvalues above their noise level, stopping at the first one
    # that is not: a later large eigenvalue after a failure does not count.
    beats_noise = eigenvalues > parallel_analysis_thresholds
    pa_count = len(beats_noise) if beats_noise.all() else int(np.argmin(beats_noise))
    return {
        "pa_count": pa_count,
        "cng_count": calculate_cng_count(eigenvalues),
    }


def calculate_suitability(
    item_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report whether the items are suited to factor analysis (KMO, Bartlett).

    The Kaiser-Meyer-Olkin (KMO) measure compares the items' correlations
    with their partial correlations (correlations left after removing the
    other items); values near 1 mean the items share common variance. Bartlett's
    test asks whether the correlation matrix differs from the identity matrix
    (no shared variance). The values are reported only; no cut-off stops the
    run.

    Args:
        item_data: Mean-imputed subjects-by-items table.

    Returns:
        A tuple ``(suitability, kmo_by_item)``: a one-row table with
        ``kmo_overall``, ``bartlett_chi_square``,
        ``bartlett_degrees_of_freedom`` and ``bartlett_p_value``, and a table
        with the KMO of each ``item``.

    Raises:
        ValueError: If any statistic is not a finite number.
    """
    values = item_data.to_numpy(dtype=float)
    kmo_per_item, kmo_overall = calculate_kmo(values)
    chi_square, p_value = calculate_bartlett_sphericity(values)
    n_items = values.shape[1]
    # factor_analyzer does not return the degrees of freedom, but it tests
    # the p * (p - 1) / 2 distinct correlations.
    degrees_of_freedom = n_items * (n_items - 1) / 2

    statistics = np.array([kmo_overall, chi_square, p_value, *kmo_per_item])
    if not np.isfinite(statistics).all():
        raise ValueError(
            "The suitability statistics (KMO, Bartlett) are not all finite "
            "numbers; the correlation matrix may be singular."
        )
    suitability = pd.DataFrame(
        {
            "kmo_overall": [kmo_overall],
            "bartlett_chi_square": [chi_square],
            "bartlett_degrees_of_freedom": [degrees_of_freedom],
            "bartlett_p_value": [p_value],
        }
    )
    kmo_by_item = pd.DataFrame({"item": item_data.columns, "kmo": kmo_per_item})
    return suitability, kmo_by_item


def calculate_tucker_congruence(
    reference_loadings: np.ndarray, other_loadings: np.ndarray
) -> np.ndarray:
    """Calculate Tucker's congruence between every pair of factors.

    Tucker's congruence measures how alike two loading columns are: the sum of
    products of the loadings divided by the square root of the product of the
    sums of squares. It is 1 for identical patterns, 0 for unrelated ones and
    -1 for the same pattern with flipped sign. Values of about 0.95 or more
    are usually read as the same factor.

    Args:
        reference_loadings: Items-by-factors loadings from the full sample.
        other_loadings: Items-by-factors loadings in the same item order.

    Returns:
        A factors-by-factors matrix. Entry ``[r, c]`` is the congruence of
        reference factor ``r`` with factor ``c`` of ``other_loadings``.
    """
    cross_products = reference_loadings.T @ other_loadings
    lengths = np.outer(
        np.sqrt(np.sum(reference_loadings**2, axis=0)),
        np.sqrt(np.sum(other_loadings**2, axis=0)),
    )
    return cross_products / lengths


def match_factors_to_reference(
    reference_loadings: np.ndarray, other_loadings: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match each factor of one solution to a factor of a reference solution.

    Factor order and sign are arbitrary in a separate factor analysis, so
    factors from a half sample cannot be compared column by column. This
    pairs the factors so that the total absolute Tucker congruence is the
    highest (each factor used once), then flips signs so that every matched
    congruence is positive.

    Args:
        reference_loadings: Items-by-factors loadings from the full sample.
        other_loadings: Items-by-factors loadings in the same item order.

    Returns:
        A tuple ``(matched_columns, signs, congruences)``. For reference
        factor ``r``, ``matched_columns[r]`` is the column of
        ``other_loadings`` that matches it, ``signs[r]`` is the sign (+1 or
        -1) that makes the match positive, and ``congruences[r]`` is the
        resulting positive congruence.
    """
    congruence = calculate_tucker_congruence(reference_loadings, other_loadings)
    reference_columns, matched_columns = linear_sum_assignment(-np.abs(congruence))
    matched_values = congruence[reference_columns, matched_columns]
    signs = np.where(matched_values < 0, -1.0, 1.0)
    return matched_columns, signs, matched_values * signs


def calculate_split_half_congruence(
    item_data: pd.DataFrame, full_loadings: pd.DataFrame
) -> pd.DataFrame:
    """Check that the factors reappear when the participants are split in two.

    Factors that change a lot when only half the participants are used are
    unlikely to replicate. The participants are shuffled once with a fixed
    seed. Half A is the first half of the shuffled rows and half B is the
    rest. The same number of factors is fitted in each half and matched to
    the full-sample factors by Tucker congruence.

    Args:
        item_data: Mean-imputed subjects-by-items table.
        full_loadings: Full-sample loadings, indexed by item name, with one
            column per factor.

    Returns:
        One row per half and full-sample factor, with the columns ``half``,
        ``full_sample_factor``, ``half_factor`` (the matched factor of that
        half) and ``congruence`` (positive after the sign flip).
    """
    n_factors = full_loadings.shape[1]
    reference = full_loadings.loc[item_data.columns].to_numpy()

    rng = np.random.default_rng(SPLIT_HALF_SEED)
    shuffled_rows = rng.permutation(len(item_data))
    n_in_first_half = len(item_data) // 2
    halves = {
        "A": shuffled_rows[:n_in_first_half],
        "B": shuffled_rows[n_in_first_half:],
    }

    rows = []
    for half_name, half_rows in halves.items():
        half_model = fit_factor_model_with_checks(
            item_data.iloc[half_rows],
            n_factors,
            f"item-level EFA, {n_factors} factors, split half {half_name}",
        )
        matched_columns, _, congruences = match_factors_to_reference(
            reference, half_model.loadings_
        )
        for factor_number in range(n_factors):
            rows.append(
                {
                    "half": half_name,
                    "full_sample_factor": f"ML{factor_number + 1}",
                    "half_factor": f"ML{matched_columns[factor_number] + 1}",
                    "congruence": congruences[factor_number],
                }
            )
    return pd.DataFrame(rows)


def build_factor_count_tables(
    n_participants: int,
    eigenvalues: np.ndarray,
    parallel_analysis_thresholds: np.ndarray,
    factor_counts: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two tables that document the factor-count decision.

    Args:
        n_participants: Number of participants in the item-level analysis.
        eigenvalues: Observed eigenvalues in descending order.
        parallel_analysis_thresholds: 95th percentile of the noise
            eigenvalues, one per observed eigenvalue.
        factor_counts: The counts returned by
            :func:`calculate_factor_count_rules`.

    Returns:
        A tuple ``(factor_count, by_position)``: a one-row table of the counts
        and settings, and a table with, for each eigenvalue position, the
        observed eigenvalue, the parallel-analysis threshold and the CNG
        slope change (empty where CNG does not define one).
    """
    factor_count = pd.DataFrame(
        [
            {
                "n_participants": n_participants,
                "n_items": len(eigenvalues),
                "pa_count": factor_counts["pa_count"],
                "cng_count": factor_counts["cng_count"],
                "selected_n_factors": SELECTED_N_FACTORS,
                "pa_n_random_datasets": PA_N_RANDOM_DATASETS,
                "pa_seed": PARALLEL_ANALYSIS_SEED,
                "split_half_seed": SPLIT_HALF_SEED,
            }
        ]
    )

    slope_changes = np.full(len(eigenvalues), np.nan)
    cng_values = calculate_cng_slope_changes(eigenvalues)
    slope_changes[: len(cng_values)] = cng_values
    by_position = pd.DataFrame(
        {
            "position": np.arange(1, len(eigenvalues) + 1),
            "observed_eigenvalue": eigenvalues,
            "parallel_analysis_95th_percentile": parallel_analysis_thresholds,
            "cng_slope_change": slope_changes,
        }
    )
    return factor_count, by_position


def calculate_questionnaire_reliability(
    item_scores: pd.DataFrame, scoring_key: pd.DataFrame, questionnaire_keys: list[str]
) -> pd.DataFrame:
    """Calculate Cronbach's alpha and item-rest correlations for each questionnaire.

    This is the data-based check that the scoring key is right. Items of one
    scale should rise and fall together once reverse items are flipped, so
    every item should correlate positively with the sum of the other items
    (its "item-rest" correlation). A negative item-rest correlation points to
    an item with the wrong role in the key. It is reported and never stops
    the run, because no cut-off is part of the method. Cronbach's alpha is
    ``k / (k - 1) * (1 - sum of item variances / variance of the total)`` for
    ``k`` items, using sample variances.

    Only participants who answered every scored item of a questionnaire are
    used for that questionnaire.

    Args:
        item_scores: Item scores of the cleaned participants, from
            :func:`score_items`.
        scoring_key: The key from :func:`load_scoring_key`; it lists the scored
            items of each questionnaire.
        questionnaire_keys: The questionnaires to report, in output order.

    Returns:
        One row per questionnaire with the columns ``questionnaire``,
        ``n_items``, ``n_participants``, ``cronbach_alpha``,
        ``min_item_rest_correlation`` and
        ``item_with_min_item_rest_correlation``.

    Raises:
        ValueError: If alpha or the item-rest correlations cannot be
            calculated for a questionnaire (for example, no participant
            answered all of its scored items).
    """
    wide = item_scores.pivot(
        index="assignedSubjectId", columns="questionId", values="itemScore"
    )
    rows = []
    for q in questionnaire_keys:
        is_scored_item = (scoring_key["questionnaire_key"] == q) & scoring_key[
            "scoring"
        ].isin(SCORED_ROLES)
        items = scoring_key.loc[is_scored_item, "question_id"].tolist()
        answers = wide.reindex(columns=items).dropna()
        n_participants, n_items = answers.shape

        total = answers.sum(axis=1)
        alpha = (
            n_items
            / (n_items - 1)
            * (1 - answers.var(ddof=1).sum() / total.var(ddof=1))
        )
        # Correlate each item with the sum of all the other items.
        item_rest = pd.Series(
            {item: answers[item].corr(total - answers[item]) for item in items}
        )
        if not np.isfinite(alpha) or item_rest.isna().all():
            raise ValueError(
                f"Reliability of {q} could not be calculated from "
                f"{n_participants} participant(s) who answered all {n_items} "
                "scored items."
            )
        rows.append(
            {
                "questionnaire": q,
                "n_items": n_items,
                "n_participants": n_participants,
                "cronbach_alpha": alpha,
                "min_item_rest_correlation": item_rest.min(),
                "item_with_min_item_rest_correlation": item_rest.idxmin(),
            }
        )
    return pd.DataFrame(rows)


def write_or_compare_factor_checks(
    check_tables: dict[str, pd.DataFrame], factor_checks_dir: Path
) -> None:
    """Write the factor-analysis checks, or prove they match an earlier run.

    The checks use only questionnaire answers, so each learning-rate
    parameter's run should produce identical tables. The first run writes
    them. Later runs compare their tables with the stored files, which shows
    that every run used the same factor solution. Nothing is overwritten.

    Args:
        check_tables: Mapping of file name to table. All files are handled
            together.
        factor_checks_dir: The shared folder for these files.

    Raises:
        ValueError: If only some of the files already exist, or if a stored
            file differs from the new table (different data or code).
    """
    existing = [name for name in check_tables if (factor_checks_dir / name).is_file()]
    if not existing:
        for name, table in check_tables.items():
            _write_table(table, factor_checks_dir / name)
        return

    if len(existing) < len(check_tables):
        missing = [name for name in check_tables if name not in existing]
        raise ValueError(
            f"{factor_checks_dir} holds only some of the factor-analysis "
            f"checks (missing: {missing}). Delete the folder and run again."
        )

    for name, table in check_tables.items():
        stored = pd.read_csv(factor_checks_dir / name)
        # Read the new table back through CSV text, so both tables have been
        # through the same rounding and column types.
        new = pd.read_csv(io.StringIO(table.to_csv(index=False)))
        try:
            pd.testing.assert_frame_equal(stored, new, check_exact=False, rtol=1e-8)
        except AssertionError as error:
            raise ValueError(
                f"{factor_checks_dir / name} differs from this run's table. "
                "The folder holds checks from different data or code; delete "
                "it before the checks can be replaced."
            ) from error


def fit_selected_factor_solution(
    cleaned_data: pd.DataFrame,
    selected_n_factors: int,
    updated_subject_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fit and label the selected-solution factor model.

    Args:
        cleaned_data: Mean-imputed subjects-by-items wide table.
        selected_n_factors: Number of factors to extract
            (``SELECTED_N_FACTORS``).
        updated_subject_ids: Subject IDs in ``cleaned_data``'s row order.

    Returns:
        A tuple ``(selected_loadings, factor_scores, factor_cols)``: the
        factor loadings labelled with ``measure``/``itemNumber``/``item`` and
        sorted by measure then item number; the per-subject factor scores
        with ``assignedSubjectId``; and the ``ML{i}`` column names.
    """
    model = fit_factor_model_with_checks(
        cleaned_data,
        selected_n_factors,
        f"item-level EFA, {selected_n_factors} factors, full sample",
    )

    ml_cols = [f"ML{i}" for i in range(1, selected_n_factors + 1)]
    selected_loadings = pd.DataFrame(
        model.loadings_, index=cleaned_data.columns, columns=ml_cols
    )

    split_parts = selected_loadings.index.to_series().str.split("__", expand=True)
    selected_loadings["measure"] = split_parts[0]
    selected_loadings["itemNumber"] = range(1, len(selected_loadings) + 1)
    selected_loadings["item"] = selected_loadings.index

    selected_loadings["sort_measure"] = split_parts[0]
    selected_loadings["sort_number"] = pd.to_numeric(
        split_parts[1], errors="coerce"
    ).astype("Int64")
    selected_loadings = selected_loadings.sort_values(
        ["sort_measure", "sort_number"]
    ).drop(["sort_measure", "sort_number"], axis=1)

    factor_scores = pd.DataFrame(
        model.transform(cleaned_data), columns=ml_cols
    )
    factor_scores["assignedSubjectId"] = updated_subject_ids.values
    factor_scores = factor_scores[["assignedSubjectId"] + ml_cols]

    return selected_loadings, factor_scores, ml_cols


def name_selected_factors(
    selected_loadings: pd.DataFrame, factor_cols: list[str]
) -> list[str]:
    """Give each selected factor its name, after checking the name still fits.

    The names in ``FACTOR_NAMES`` were chosen by reading the loadings of one
    fit. A factor analysis does not fix the order or the sign of its factors,
    so new data can give the same factors as different ``ML`` columns, or
    with all signs reversed. A name used without a check could then label the
    wrong factor in a figure.

    For each factor, this function sums the loadings of its marker
    questionnaire's items (``FACTOR_MARKER_QUESTIONNAIRES``) on every factor.
    The name fits if that questionnaire loads most strongly (largest absolute
    sum) on the named factor, with the expected sign.

    Args:
        selected_loadings: Selected-solution loadings, one row per item, with
            a ``measure`` column (the questionnaire) and one column per
            factor.
        factor_cols: The factor columns, ``ML1`` ... ``ML<n>``.

    Returns:
        The factor names, in the order of ``factor_cols``.

    Raises:
        ValueError: If the factors are not exactly the named ones, or if a
            marker questionnaire loads most strongly on another factor or
            with the other sign. The names must then be checked again by
            reading the loadings.
    """
    if list(factor_cols) != list(FACTOR_NAMES):
        raise ValueError(
            f"Factor names are set for {list(FACTOR_NAMES)}, but the selected "
            f"solution has {list(factor_cols)}. Choose the names again from "
            "the loadings."
        )

    # Rows: questionnaires; columns: factors. Each value is the sum of that
    # questionnaire's item loadings on that factor.
    summed_loadings = selected_loadings.groupby("measure")[factor_cols].sum()
    for factor, (questionnaire, expected_sign) in FACTOR_MARKER_QUESTIONNAIRES.items():
        sums = summed_loadings.loc[questionnaire]
        strongest_factor = sums.abs().idxmax()
        if (
            strongest_factor != factor
            or np.sign(sums[strongest_factor]) != expected_sign
        ):
            raise ValueError(
                f"The name '{FACTOR_NAMES[factor]}' for {factor} does not fit "
                f"the new loadings: {questionnaire} items should load most "
                f"strongly on {factor} with sign {expected_sign:+d}, but their "
                f"summed loadings are {sums.round(2).to_dict()}. Factor order "
                "or sign has changed; choose the names again from the loadings."
            )
    return [FACTOR_NAMES[factor] for factor in factor_cols]


def fit_ols_per_version(
    data: pd.DataFrame, outcome: str, predictors: list[str], regression_name: str
) -> pd.DataFrame:
    """Fit one OLS model per task version and list every predictor's estimate.

    Each participant does one task version, so the same model is fitted
    separately to each version's participants. Both the main factor
    regression and its baseline-adjusted sensitivity model use this, so they
    differ only in their outcome and predictors.

    Args:
        data: One row per participant, with a ``version`` column, the
            ``outcome`` column and every ``predictors`` column.
        outcome: The column to predict, for example ``lr_change``.
        predictors: The predictor columns, all entered together.
        regression_name: Which regression is being fitted, for the error
            message if a version is too small.

    Returns:
        One row per (version, predictor), in version order, with
        ``Version``, ``Predictor``, ``N``, ``β``, ``SE``, ``t``, ``p`` and
        the model's ``R²``.

    Raises:
        ValueError: If a version has fewer than ``len(predictors) + 2``
            participants.
    """
    formula = f"{outcome} ~ " + " + ".join(predictors)
    rows = []
    for version in VERSIONS:
        data_v = data[data["version"].astype(str) == version]
        check_version_sample_size(data_v, version, len(predictors), regression_name)
        model = ols(formula, data=data_v).fit()
        for predictor in predictors:
            rows.append(
                {
                    "Version": version,
                    "Predictor": predictor,
                    "N": len(data_v),
                    "β": model.params[predictor],
                    "SE": model.bse[predictor],
                    "t": model.tvalues[predictor],
                    "p": model.pvalues[predictor],
                    "R²": model.rsquared,
                }
            )
    return pd.DataFrame(rows)


def build_updated_factor_results_df(
    lr_pivot: pd.DataFrame,
    updated_factor_scores: pd.DataFrame,
    updated_factor_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit per-version regressions of LR change on the updated EFA factors.

    Args:
        lr_pivot: Per-subject/version LR change, from :func:`build_lr_pivot`.
        updated_factor_scores: Per-subject updated-EFA factor scores.
        updated_factor_cols: The ``ML{i}`` column names.

    Returns:
        A tuple ``(updated_lr_factors, updated_factor_results_df)``: the
        merged LR-change/factor-score frame, and the per-(factor, version)
        regression results with Benjamini-Hochberg q-values over all factors
        x 4 versions.

    Raises:
        ValueError: If a version has fewer than k + 2 participants for k
            factors.
    """
    updated_lr_factors = pd.merge(
        lr_pivot,
        updated_factor_scores,
        left_on="subject_id",
        right_on="assignedSubjectId",
        how="inner",
    )

    results = fit_ols_per_version(
        updated_lr_factors,
        "lr_change",
        updated_factor_cols,
        "Item-level factor regression",
    ).rename(columns={"Predictor": "Factor"})
    results["Sig"] = results["p"].apply(sig_stars)

    # The question is whether any factor relates to learning-rate change in
    # any version, so every factor x version test is one family. Correcting
    # each factor (or each version) separately would control fewer tests than
    # are actually looked at: with 4 factors x 4 versions there are 16 chances
    # of a false positive, not 4.
    updated_factor_results_df = add_fdr_correction(results)

    return updated_lr_factors, updated_factor_results_df


def build_baseline_adjusted_results_df(
    updated_lr_factors: pd.DataFrame, updated_factor_cols: list[str]
) -> pd.DataFrame:
    """Fit the baseline-adjusted sensitivity model for the main factor regression.

    For each task version, fits ``lr_block2 ~ lr_block1 + ML1 + ... + MLk``
    by OLS on the same participants as the main change-score model. This is
    a sensitivity check reported alongside the main result, never instead of
    it.

    Args:
        updated_lr_factors: The merged LR-change/factor-score frame used by
            the main model, from :func:`build_updated_factor_results_df`.
        updated_factor_cols: The ``ML{i}`` column names.

    Returns:
        One row per (version, predictor) with ``Version``, ``Predictor``,
        ``N``, ``β``, ``SE``, ``t``, ``p``, ``p_fdr`` and ``Sig_fdr``. The
        Benjamini-Hochberg family is the factor rows across all versions;
        the ``lr_block1`` rows are not tested, so their ``p_fdr`` and
        ``Sig_fdr`` are empty.

    Raises:
        ValueError: If a version has fewer than k + 3 participants for k
            factors (k factors plus block 1).
    """
    # The main model asks whether a trait predicts how much the learning rate
    # changes (block 2 minus block 1). This model instead asks whether a trait
    # predicts the block 2 learning rate among people with the same block 1
    # learning rate. The two can disagree. Block 1 is itself a noisy estimate,
    # so adjusting for it only partly removes baseline differences, and if a
    # trait relates to block 1, part of that relation is wrongly credited to
    # the trait (Lord's paradox). A disagreement is something to interpret,
    # not an error.
    results = fit_ols_per_version(
        updated_lr_factors,
        "lr_block2",
        ["lr_block1"] + updated_factor_cols,
        "Baseline-adjusted factor regression",
    ).drop(columns="R²")
    return add_fdr_correction(results, results["Predictor"] != "lr_block1")


def count_participants_per_version(updated_lr_factors: pd.DataFrame) -> dict[str, int]:
    """Count the regression participants, and those with a non-converged fit, per version.

    Args:
        updated_lr_factors: The merged LR-change/factor-score frame used by
            the main factor regression, with a ``both_blocks_converged``
            column.

    Returns:
        For each version ``V``, ``n_nonconverged_V`` (participants with at
        least one block whose fit did not converge; they are still in the
        regressions) and ``n_regression_participants_V``.
    """
    by_version = {
        version: updated_lr_factors[
            updated_lr_factors["version"].astype(str) == version
        ]
        for version in VERSIONS
    }
    counts = {
        f"n_nonconverged_{version}": int((~data_v["both_blocks_converged"]).sum())
        for version, data_v in by_version.items()
    }
    counts.update(
        {
            f"n_regression_participants_{version}": len(data_v)
            for version, data_v in by_version.items()
        }
    )
    return counts


def _local_plot_scree(
    eigenvalues: np.ndarray, ax: Optional[plt.Axes] = None, max_factors: Optional[int] = None
) -> plt.Axes:
    """Draw a scree plot (eigenvalue vs. factor number) on ``ax``.

    This is the analysis's own local version of
    ``stats_utils.efa.plotting.plot_scree``. It is kept here, rather than using
    ``stats_utils.plot_scree``, so that the scree plot always has the same
    appearance: a line with circle markers, a dashed red line at eigenvalue 1,
    a light grid and no legend.

    Args:
        eigenvalues: Eigenvalues in descending order.
        ax: Axes to plot on; a new figure/axes is created if ``None``.
        max_factors: If given, only the first ``max_factors`` eigenvalues
            are plotted.

    Returns:
        The axes the scree plot was drawn on.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    if max_factors is not None:
        ev = ev[:max_factors]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(1, len(ev) + 1)
    ax.plot(x, ev, marker="o", linestyle="-", linewidth=2)
    ax.axhline(1, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot")
    ax.grid(alpha=0.3)
    return ax


def _save_current_figure(path: Path, dpi: Optional[int] = None) -> None:
    """Save and close the current matplotlib figure as an SVG (never shown).

    Args:
        path: Destination file path; parent directories are created if
            needed.
        dpi: Optional resolution to pass through to ``savefig``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    plt.gcf().savefig(path, **kwargs)
    plt.close("all")


def _plot_regression_grid_per_questionnaire(
    lr_q: pd.DataFrame, indiv_df: pd.DataFrame, questionnaire_keys: list[str], figures_dir: Path
) -> None:
    """Save one scatter+OLS figure per questionnaire (4 version subplots).

    Args:
        lr_q: Merged LR-change/questionnaire-totals frame.
        indiv_df: Per-(questionnaire, version) results with
            Benjamini-Hochberg q-values, from :func:`build_indiv_df`.
        questionnaire_keys: The 12 questionnaire keys.
        figures_dir: Directory to save the SVGs under.
    """
    for q in questionnaire_keys:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        fig.suptitle(
            f"{Q_LABELS.get(q, q)} vs Learning Rate Change", fontsize=14, fontweight="bold"
        )

        for idx, v in enumerate(VERSIONS):
            ax = axes[idx]
            # `build_indiv_df` has already stopped the run if a version is
            # too small to fit, so every panel has a line.
            data_v = lr_q[lr_q["version"].astype(str) == v]

            ax.scatter(
                data_v[q],
                data_v["lr_change"],
                alpha=0.45,
                edgecolors="white",
                linewidth=0.5,
                s=40,
                color="#1e90ff",
            )
            X = sm.add_constant(data_v[q])
            y = data_v["lr_change"]
            ols_model = sm.OLS(y, X).fit()
            x_sorted = data_v[q].sort_values()
            ax.plot(
                x_sorted,
                ols_model.predict(sm.add_constant(x_sorted)),
                color="#004d99",
                linewidth=1,
            )

            row = indiv_df[(indiv_df["q_key"] == q) & (indiv_df["Version"] == v)].iloc[0]
            p_fdr = row["p_fdr"]
            sig = sig_stars(p_fdr)
            p_str = "p_fdr < .001" if p_fdr < 0.001 else f"p_fdr = {p_fdr:.3f}"

            ax.set_title(VERSION_LABELS[v], fontweight="bold", fontsize=12, pad=20)
            ax.text(
                0.5,
                1.02,
                # DM Sans has no Greek letters, so β is drawn as math text.
                rf"$\beta$ = {row['β']:.3f}, {p_str} {sig}, R² = {row['R²']:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
            )
            ax.set_xlabel(Q_LABELS.get(q, q))
            if idx == 0:
                ax.set_ylabel("LR Change (logit)")
            ax.grid(False)
            ax.margins(x=0.05, y=0.05)
            ax.tick_params(
                axis="both",
                pad=2,
                length=4,
                width=0.5,
                direction="out",
                bottom=True,
                left=True,
                color="grey",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        _save_current_figure(figures_dir / f"lr_change_vs_{q}.svg")


def _plot_factor_grid(
    lr_factors: pd.DataFrame,
    factor_results_df: pd.DataFrame,
    factor_cols: list[str],
    figures_dir: Path,
) -> None:
    """Save the factor-scores-vs-LR-change grid (rows=factors, cols=versions).

    Args:
        lr_factors: Merged LR-change/factor-score frame.
        factor_results_df: Per-(factor, version) results with
            Benjamini-Hochberg q-values, from :func:`build_factor_results_df`.
        factor_cols: The Kaiser ``Factor_i`` column names.
        figures_dir: Directory to save the SVG under.
    """
    fig, axes = plt.subplots(len(factor_cols), 4, figsize=(16, 4 * len(factor_cols)), sharey=True)
    axes = np.atleast_2d(axes)
    # The suptitle is set once, with its final text and size (fontsize=16).
    fig.suptitle(
        "Regression Plots for Factors Predicting Learning Rate Change",
        fontsize=16,
        fontweight="bold",
    )

    for row, factor in enumerate(factor_cols):
        for col, version in enumerate(VERSIONS):
            ax = axes[row, col]
            # `build_factor_results_df` has already stopped the run if a
            # version is too small to fit, so every panel has a line.
            data_v = lr_factors[lr_factors["version"].astype(str) == version]

            ax.scatter(
                data_v[factor],
                data_v["lr_change"],
                alpha=0.45,
                edgecolor="white",
                linewidth=0.5,
                s=40,
                color="#1e90ff",
            )
            X = sm.add_constant(data_v[factor])
            y = data_v["lr_change"]
            model = sm.OLS(y, X).fit()
            x_sorted = data_v[factor].sort_values()
            ax.plot(
                x_sorted,
                model.predict(sm.add_constant(x_sorted)),
                color="#004d99",
                linewidth=1,
            )

            p_fdr = factor_results_df.loc[
                (factor_results_df["Version"] == version)
                & (factor_results_df["Factor"] == factor),
                "p_fdr",
            ].values[0]
            sig = sig_stars(p_fdr)
            p_str = "p_fdr < .001" if p_fdr < 0.001 else f"p_fdr = {p_fdr:.3f}"

            ax.set_title(VERSION_LABELS[version], fontweight="bold", fontsize=12, pad=20)
            ax.text(
                0.5,
                1.02,
                # DM Sans has no Greek letters, so β is drawn as math text.
                rf"$\beta$ = {model.params[factor]:.3f}, {p_str} {sig}, R² = {model.rsquared:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
            )
            ax.set_xlabel(factor)
            if col == 0:
                ax.set_ylabel("LR Change (logit)")
            ax.grid(False)
            ax.margins(x=0.05, y=0.05)
            ax.tick_params(
                axis="both",
                pad=2,
                length=4,
                width=0.5,
                direction="out",
                bottom=True,
                left=True,
                color="grey",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    _save_current_figure(figures_dir / "factor_scores_vs_lr_change.svg")


def plot_item_loadings_by_measure(
    loadings_csv: Path, factor_names: list[str], figure_path: Path
) -> None:
    """Plot each factor's item loadings as bars, coloured by questionnaire.

    One panel is drawn per factor, titled with the factor's name. The panels
    are made here and passed to ``plot_factor_loadings``: when that helper
    makes them itself, a one-factor solution gives a single panel that it
    cannot index, and the run stops with an error.

    Each questionnaire gets its colour from ``QUESTIONNAIRE_COLOURS``.

    Args:
        loadings_csv: The saved selected-solution loadings table, with
            ``item``, ``itemNumber``, ``measure`` and ``ML1`` ...
            ``ML<n>`` columns.
        factor_names: The name of each factor, in ``ML1`` ... ``ML<n>``
            order (from :func:`name_selected_factors`).
        figure_path: Where to save the figure (SVG).

    Raises:
        ValueError: If a questionnaire in the table has no colour in
            ``QUESTIONNAIRE_COLOURS``.
    """
    # The helper colours the k-th questionnaire (in table order) with
    # cmap(k / number of questionnaires). This function turns that fraction
    # back into k, so each questionnaire gets its own fixed colour.
    measures = list(pd.read_csv(loadings_csv)["measure"].unique())
    missing = [m for m in measures if m not in QUESTIONNAIRE_COLOURS]
    if missing:
        raise ValueError(f"No bar colour is set for questionnaires {missing}.")
    colours = [QUESTIONNAIRE_COLOURS[m] for m in measures]

    def colour_by_position(fraction: float) -> str:
        """Return the colour of the questionnaire at ``fraction`` of the list.

        Args:
            fraction: The questionnaire's position divided by the number of
                questionnaires, as passed by ``plot_factor_loadings``.

        Returns:
            The questionnaire's colour as a hex string.
        """
        return colours[round(fraction * len(colours))]

    _, panels = plt.subplots(len(factor_names), 1, figsize=(8, 8), squeeze=False)
    plot_factor_loadings(
        str(loadings_csv),
        factor_names,
        cmap=colour_by_position,
        ax=panels[:, 0],
    )
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.grid(False)
        ax.tick_params(axis="both", which="both", bottom=True, left=True, colors="grey")
        if ax.get_title():
            ax.set_title(ax.get_title(), fontweight="bold")
    fig.subplots_adjust(wspace=1, hspace=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save_current_figure(figure_path)


def _write_table(df: pd.DataFrame, path: Path, index: bool = False, index_label: Optional[str] = None) -> None:
    """Write a table to CSV, creating parent directories as needed.

    Args:
        df: The table to write.
        path: Destination CSV path.
        index: Whether to include the DataFrame's index as a column.
        index_label: Optional column name for the written index.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, index_label=index_label)


# ── Top-level orchestration ─────────────────────────────────────────────────


def run_regression_analysis(
    parameter: str,
    lr_csv: Path,
    model_name: str,
    lr_column: str,
    questionnaire_zip: Path,
    output_dir: Path,
    factor_checks_dir: Path,
) -> dict[str, Any]:
    """Run the full individual-differences regression analysis for one parameter.

    Loads the inputs fresh, checks the scoring key and attention checks against
    the whole questionnaire archive, matches participants and applies the
    exclusions, scores the items from the key, runs the factor analyses and
    regressions, writes every result table and figure under
    ``<output_dir>/<parameter>/``, and returns the results held in memory. The
    factor-analysis checks and the reliability table are written to (or
    compared with the files already in) the shared ``factor_checks_dir``.

    Args:
        parameter: One of ``learning_rate``, ``learning_rate_neg``,
            ``learning_rate_pos`` (used only for output paths and messages;
            the actual inputs are given explicitly below).
        lr_csv: Path to the subject-level learning-rate parameters CSV.
        model_name: Model identifier to select rows for.
        lr_column: Learning-rate column to analyse.
        questionnaire_zip: Path to the questionnaire zip archive.
        output_dir: Root output directory; results are written under
            ``output_dir / parameter``.
        factor_checks_dir: Folder shared by all parameters for the
            factor-analysis checks (factor count, suitability, split half,
            skew-filter list) and the questionnaire reliability table. It
            should lie outside ``output_dir``.

    Returns:
        A dict with the participant counts and analysis tables
        (``num_ppts_LR_cleaned``, ``num_ppts_Q_cleaned``,
        ``num_ppts_with_any_questionnaire``,
        ``num_ppts_completed_all_questionnaires`` -- the number with any
        questionnaire row and the number who completed all 12, which are
        different counts -- ``num_ppts_failed_more_than_one_check`` -- how
        many matched participants the attention checks exclude --
        ``num_male``, ``num_female``, ``mean_age``,
        ``std_age``, ``eigenvalues``, ``n_factors``,
        ``indiv_df``, ``factor_results_df``, ``updated_ev_df``,
        ``updated_selected_n_factors``, ``updated_selected_loadings``,
        ``updated_factor_scores``, ``updated_factor_results_df``,
        ``baseline_adjusted_results_df`` -- the baseline-adjusted sensitivity
        model -- and ``updated_lr_factors`` -- the learning-rate change merged
        with the updated factor scores), plus
        ``figures_written`` (the list of SVG filenames actually saved) and
        ``high_correlation_pair_count`` (the number of item pairs with
        |correlation| > 0.7 found during the updated EFA's item filtering;
        ``high_correlations.svg`` is saved if and only if this is > 0).

    Raises:
        ValueError: If the scoring key or an attention check does not fit the
            data, if a matched participant has two answers to one item, if a
            factor model does not converge or has a Heywood case, if the
            selected factor count is below 1, if the shared checks folder
            holds files that differ from this run's checks, if the
            learning-rate table lacks a True/False ``converged`` column, if
            a participant lacks a block, has a repeated block or has a
            non-finite logit learning rate, or if a task version has too few
            participants for a regression.
    """
    param_dir = output_dir / parameter
    tables_dir = param_dir / "tables"
    figures_dir = param_dir / "figures"

    # Fresh, independent load for this parameter only.
    lr_data = load_lr_data(lr_csv, model_name, lr_column)
    data_q = load_questionnaire_data(questionnaire_zip)
    scoring_key = load_scoring_key(SCORING_KEY_CSV)

    cleaned = clean_and_match_data(lr_data, data_q, QUESTIONNAIRE_KEYS, scoring_key)
    lr_data_new = cleaned["lr_data_new"]
    data_q = cleaned["data_q"]
    q_all = cleaned["q_all"]
    item_scores = cleaned["item_scores"]

    ensure_dm_sans_font()
    # `set_theme` resets the font to seaborn's default, so DM Sans (the font
    # of the publication figures) is set again here. The font sizes stay at
    # seaborn's sizes, which suit these larger figures.
    sns.set_theme(
        font_scale=1, style="whitegrid", rc={"font.family": [DM_SANS_FONT_NAME]}
    )
    biomodal_palette = ["#9CDBD9", "#C0DF16", "#003B49", "#05868E", "#00DAEF", "#50B37B", "#1e90ff"]
    sns.set_palette(biomodal_palette)

    # Initial (Kaiser) factor analysis of questionnaire totals.
    eigenvalues, n_factors, initial_loadings, factor_scores, factor_cols = fit_kaiser_factor_analysis(
        q_all, QUESTIONNAIRE_KEYS
    )
    q_all[factor_cols] = factor_scores

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(initial_loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Factor Loadings for All Questionnaires")
    ax.set_ylabel("Questionnaire")
    plt.tight_layout()
    _save_current_figure(figures_dir / "factor_loadings_all_questionnaires.svg")

    # Correlate only the 12 questionnaire totals, not the factor scores just
    # added to q_all. Selecting the 12 totals by name keeps the matrix correct
    # whatever the number of factors.
    correlation_matrix = q_all[QUESTIONNAIRE_KEYS].corr()
    reversed_corr_matrix = correlation_matrix.iloc[::-1, ::-1].iloc[:, ::-1]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        reversed_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        xticklabels=reversed_corr_matrix.columns,
        yticklabels=reversed_corr_matrix.index,
    )
    plt.title("Correlation Matrix of Questionnaire Scores")
    plt.tight_layout()
    _save_current_figure(figures_dir / "questionnaire_correlation_matrix.svg")

    # Learning-rate change for each participant and task version.
    lr_rw = build_logit_lr_frame(lr_data_new, model_name, lr_column)

    # Individual-questionnaire regressions on LR change.
    lr_pivot = build_lr_pivot(lr_rw)
    indiv_df, lr_q = build_indiv_df(lr_pivot, q_all, QUESTIONNAIRE_KEYS)
    _plot_regression_grid_per_questionnaire(lr_q, indiv_df, QUESTIONNAIRE_KEYS, figures_dir)

    demographics = compute_demographics(lr_q, data_q)

    # Kaiser-factor regressions on LR change.
    factor_results_df = build_factor_results_df(lr_pivot, q_all, factor_cols)
    lr_factors_for_plot = pd.merge(
        lr_pivot,
        q_all[["assignedSubjectId"] + factor_cols],
        left_on="subject_id",
        right_on="assignedSubjectId",
        how="inner",
    )
    _plot_factor_grid(lr_factors_for_plot, factor_results_df, factor_cols, figures_dir)

    # Updated exploratory factor analysis, at the item level.
    updated_q_data, updated_subject_ids = prepare_item_level_data(item_scores)
    updated_q_data, skew_filtered_items = filter_skewed_items(
        updated_q_data, scoring_key
    )
    updated_cleaned_data, high_corr_pairs_df, _removed_items = remove_high_correlation_items(
        updated_q_data
    )

    if len(high_corr_pairs_df) > 0:
        high_corr_items = sorted(
            set(high_corr_pairs_df["item_a"]).union(high_corr_pairs_df["item_b"])
        )
        updated_high_corrs = updated_q_data[high_corr_items].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            updated_high_corrs,
            annot=False,
            cmap="coolwarm",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title("Highly Correlated Updated EFA Items")
        plt.tight_layout()
        _save_current_figure(figures_dir / "high_correlations.svg")

    updated_cleaned_data = impute_missing_items(updated_cleaned_data)
    updated_ev = calculate_item_eigenvalues(updated_cleaned_data)
    updated_ev_df = pd.DataFrame(updated_ev, columns=["Eigenvalues"])

    # The factor count is fixed at SELECTED_N_FACTORS (the scree elbow; see
    # the reasons beside that constant). Parallel analysis and CNG are worked
    # out only to be reported.
    parallel_analysis_thresholds = calculate_parallel_analysis_thresholds(
        updated_cleaned_data.shape[0], updated_cleaned_data.shape[1]
    )
    factor_counts = calculate_factor_count_rules(
        updated_ev, parallel_analysis_thresholds
    )
    updated_selected_n_factors = SELECTED_N_FACTORS

    ax = _local_plot_scree(updated_ev)
    for line in ax.get_lines():
        line.set_color("#1e90ff")
        line.set_markerfacecolor("#1e90ff")
        line.set_markeredgecolor("#1e90ff")
    # The dotted line marks the selected factor count. The parallel-analysis
    # thresholds are not drawn: they are reported in factor_count.csv.
    ax.axvline(updated_selected_n_factors, color="black", linestyle=":", linewidth=1.5)
    plt.tight_layout()
    _save_current_figure(figures_dir / "scree_plot.svg", dpi=300)

    updated_selected_loadings, updated_factor_scores, updated_factor_cols = (
        fit_selected_factor_solution(
            updated_cleaned_data, updated_selected_n_factors, updated_subject_ids
        )
    )
    # Stop here if the factor names no longer fit the loadings, before any
    # output uses them.
    updated_factor_names = name_selected_factors(
        updated_selected_loadings, updated_factor_cols
    )

    # The checks folder is written only after every factor model, including
    # the two split halves, has passed its checks, so a failed fit never
    # leaves a partly written checks folder.
    suitability, kmo_by_item = calculate_suitability(updated_cleaned_data)
    split_half_congruence = calculate_split_half_congruence(
        updated_cleaned_data,
        updated_selected_loadings.loc[:, updated_factor_cols],
    )
    factor_count, factor_count_by_position = build_factor_count_tables(
        updated_cleaned_data.shape[0],
        updated_ev,
        parallel_analysis_thresholds,
        factor_counts,
    )
    write_or_compare_factor_checks(
        {
            "factor_count.csv": factor_count,
            "factor_count_by_position.csv": factor_count_by_position,
            "suitability.csv": suitability,
            "kmo_by_item.csv": kmo_by_item,
            "split_half_congruence.csv": split_half_congruence,
            "skew_filtered_items.csv": skew_filtered_items,
            "questionnaire_reliability.csv": calculate_questionnaire_reliability(
                item_scores, scoring_key, QUESTIONNAIRE_KEYS
            ),
        },
        factor_checks_dir,
    )

    # Tables are written before the figures that read them back from disk
    # (the just-saved CSV is the input to `plot_factor_loadings`).
    _write_table(updated_ev_df, tables_dir / "eigenvalues.csv")
    _write_table(
        updated_selected_loadings, tables_dir / "factor_loadings_selected_solution.csv", index=True
    )
    _write_table(
        updated_factor_scores, tables_dir / "factor_scores_selected_solution.csv"
    )

    updated_factor_score_corr = updated_factor_scores[updated_factor_cols].corr()
    plt.figure(
        figsize=(
            max(5, updated_selected_n_factors * 1.5),
            max(4, updated_selected_n_factors * 1.2),
        )
    )
    sns.heatmap(updated_factor_score_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Updated EFA Selected-Solution Factor Score Correlations")
    plt.tight_layout()
    _save_current_figure(figures_dir / "selected_factor_score_correlations.svg")

    updated_selected_loadings_plot = updated_selected_loadings.set_index("item")[
        updated_factor_cols
    ]
    plt.figure(figsize=(10, max(8, len(updated_selected_loadings_plot) * 0.18)))
    sns.heatmap(
        updated_selected_loadings_plot,
        annot=False,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Factor Loading"},
    )
    plt.title(f"Updated EFA {updated_selected_n_factors}-Factor Loadings")
    plt.ylabel("Item")
    plt.tight_layout()
    _save_current_figure(figures_dir / "selected_factor_loadings.svg")

    plot_item_loadings_by_measure(
        tables_dir / "factor_loadings_selected_solution.csv",
        updated_factor_names,
        figures_dir / "efa_factor_loadings.svg",
    )

    updated_lr_factors, updated_factor_results_df = build_updated_factor_results_df(
        lr_pivot, updated_factor_scores, updated_factor_cols
    )
    _write_table(
        updated_factor_results_df, tables_dir / "lr_change_regression_results.csv"
    )
    # A sensitivity check on the same participants, reported beside the main
    # change-score result above and never in its place.
    baseline_adjusted_results_df = build_baseline_adjusted_results_df(
        updated_lr_factors, updated_factor_cols
    )
    _write_table(
        baseline_adjusted_results_df,
        tables_dir / "lr_change_baseline_adjusted_results.csv",
    )
    # The script does not write summary_df.csv: a regression on the totals
    # factors would use both blocks' rows and so count each participant twice,
    # making its p-values too small. Delete any copy in the folder so it cannot
    # be mistaken for a current result.
    (tables_dir / "summary_df.csv").unlink(missing_ok=True)

    # Remaining new tables.
    counts = {
        "num_ppts_LR_cleaned": cleaned["num_ppts_LR_cleaned"],
        "num_ppts_Q_cleaned": cleaned["num_ppts_Q_cleaned"],
        "num_ppts_with_any_questionnaire": cleaned["num_ppts_with_any_questionnaire"],
        "num_ppts_completed_all_questionnaires": cleaned[
            "num_ppts_completed_all_questionnaires"
        ],
        "num_ppts_failed_more_than_one_check": cleaned[
            "num_ppts_failed_more_than_one_check"
        ],
        "num_male": demographics["num_male"],
        "num_female": demographics["num_female"],
        "mean_age": demographics["mean_age"],
        "std_age": demographics["std_age"],
        "n_factors": n_factors,
        "updated_selected_n_factors": updated_selected_n_factors,
        # Participants whose fits did not converge are kept in the
        # regressions; these columns say how many there are in each version.
        **count_participants_per_version(updated_lr_factors),
    }
    # One column per metric (rather than a single shared "value" column)
    # so integer counts keep an integer dtype instead of being upcast to
    # float64 by a mixed int/float column (which would print e.g. "313.0").
    _write_table(pd.DataFrame([counts]), tables_dir / "counts.csv")
    _write_table(
        pd.DataFrame({"Eigenvalues": eigenvalues}), tables_dir / "initial_eigenvalues.csv"
    )
    _write_table(
        initial_loadings, tables_dir / "initial_factor_loadings.csv", index=True,
        index_label="questionnaire",
    )
    _write_table(indiv_df, tables_dir / "indiv_df.csv")
    _write_table(factor_results_df, tables_dir / "factor_results_df.csv")

    figures_written = sorted(p.name for p in figures_dir.glob("*.svg"))

    return {
        "num_ppts_LR_cleaned": cleaned["num_ppts_LR_cleaned"],
        "num_ppts_Q_cleaned": cleaned["num_ppts_Q_cleaned"],
        "num_ppts_with_any_questionnaire": cleaned["num_ppts_with_any_questionnaire"],
        "num_ppts_completed_all_questionnaires": cleaned[
            "num_ppts_completed_all_questionnaires"
        ],
        "num_ppts_failed_more_than_one_check": cleaned[
            "num_ppts_failed_more_than_one_check"
        ],
        "num_male": demographics["num_male"],
        "num_female": demographics["num_female"],
        "mean_age": demographics["mean_age"],
        "std_age": demographics["std_age"],
        "eigenvalues": eigenvalues,
        "n_factors": n_factors,
        "indiv_df": indiv_df,
        "factor_results_df": factor_results_df,
        "updated_ev_df": updated_ev_df,
        "updated_selected_n_factors": updated_selected_n_factors,
        "updated_selected_loadings": updated_selected_loadings,
        "updated_factor_scores": updated_factor_scores,
        "updated_factor_results_df": updated_factor_results_df,
        "baseline_adjusted_results_df": baseline_adjusted_results_df,
        "updated_lr_factors": updated_lr_factors,
        "figures_written": figures_written,
        "high_correlation_pair_count": len(high_corr_pairs_df),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the regression analysis entrypoint.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]`` when
            ``None``.

    Returns:
        The parsed namespace, with ``parameter`` restricted to
        :data:`PARAMETER_CONFIGS`'s keys.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the individual-differences regression analysis for one "
            "learning-rate parameter."
        )
    )
    parser.add_argument(
        "--parameter", required=True, choices=sorted(PARAMETER_CONFIGS)
    )
    parser.add_argument("--output-dir", default="results/regression")
    parser.add_argument(
        "--factor-checks-dir",
        default="results/factor_analysis_checks",
        help=(
            "Folder shared by all parameters for the factor-analysis checks. "
            "The first run writes them; later runs must reproduce them."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint: run the analysis for the requested parameter.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]`` when
            ``None``.
    """
    args = parse_args(argv)
    config = PARAMETER_CONFIGS[args.parameter]
    result = run_regression_analysis(
        parameter=args.parameter,
        lr_csv=config["lr_csv"],
        model_name=config["model"],
        lr_column=config["lr_column"],
        questionnaire_zip=QUESTIONNAIRE_ZIP,
        output_dir=Path(args.output_dir),
        factor_checks_dir=Path(args.factor_checks_dir),
    )
    print(
        f"Wrote regression analysis for parameter={args.parameter!r} to "
        f"{Path(args.output_dir) / args.parameter} "
        f"({len(result['figures_written'])} figures)."
    )


if __name__ == "__main__":
    main()
