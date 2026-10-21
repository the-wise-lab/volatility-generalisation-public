"""
Data Loading and Preparation Script

This script loads the behavioral data CSVs, applies exclusion criteria,
splits by condition, and saves prepared datasets to disk for parallel fitting.

This is the first step of the pipeline (prepare, fit, model selection and
recovery, regression, figures). It writes one .npz file per condition, plus a
CSV with one row per participant and condition that shows which exclusion
criteria each participant met. The fitting scripts read the .npz files.

Each .npz file holds `choices`, `outcomes_0` and `outcomes_1`, each with one row
per participant and one column per trial. Choices are 1 for the first option and
0 for the second. Outcomes are reward-coded (1 = no explosion, 0 = explosion).
The file also holds `subject_ids` (the participant IDs, in row order) and
`outcome_coding` (a label that records how the outcomes are coded).

Usage:
    # Default workflow: the paths come from the config file
    python prepare_data.py \
                           --output-dir ../data/prepared

    # Equivalent explicit path configuration
    python prepare_data.py --main-data ../data/behavioural_data.csv.zip \
                           --output-dir ../data/prepared
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from volatility_generalisation import prepare_data_for_fitting
from volatility_generalisation.modelling_code.pipeline_config import (
    load_pipeline_config,
)
from volatility_generalisation.modelling_code.condition_metadata import (
    BLOCK_TRIAL_COUNT,
)
from volatility_generalisation.modelling_code.data_preparation import (
    OUTCOME_CODING_REWARD_SAFE,
    convert_empirical_to_reward_coded,
)

console = Console()

# Default for a setting the config file leaves out
DEFAULT_EXCLUDED_SUBJECTS = []

# Which task version and block each condition's trials come from. Version 1
# runs the volatile block first and version 2 the stable block first; versions
# 3 and 4 are the controls, with the same kind of block twice.
CONDITION_VERSION_AND_BLOCK = {
    "VolatileV1": (1, 1),
    "StableV1": (1, 2),
    "StableV2": (2, 1),
    "VolatileV2": (2, 2),
    "VolatileV3": (3, 1),
    "VolatileV3_2": (3, 2),
    "StableV4": (4, 1),
    "StableV4_2": (4, 2),
}

# The exclusion criteria, each a true/false column in the exclusion table. A
# participant is excluded from a condition if they meet any of them.
EXCLUSION_CRITERIA = [
    "on_manual_exclusion_list",
    "low_choice_variability",
    "extreme_mean_choice",
    "low_engagement_in_another_condition",
    "incomplete_trials",
]
QC_AUDIT_FILENAME = "qc_exclusion_audit.csv"
QC_COUNTS_FILENAME = "qc_exclusion_counts.csv"
# Name of the summary row that counts each participant once across conditions
ANY_CONDITION = "any_condition"
# What was measured for each participant in each condition, and the
# thresholds the measurements were compared with
MEASUREMENT_COLUMNS = [
    "input_path", "condition", "subject_id",
    "observed_trials", "expected_trials", "rolling_mean_std", "choice_mean",
    "rolling_window", "std_threshold", "mean_low", "mean_high",
]
QC_AUDIT_COLUMNS = [
    *MEASUREMENT_COLUMNS, *EXCLUSION_CRITERIA, "excluded", "exclusion_reasons",
]


def _write_qc_exclusion_audit(output_dir, exclusion_table):
    """Write the exclusion table: one row per participant and condition.

    The rows are sorted by condition in task order and then by participant, so
    the file is the same each time the script runs on the same data. Any
    earlier copy of the file is overwritten.

    Args:
        output_dir: Folder in which to save the CSV.
        exclusion_table: Exclusion table, as returned by
            `assess_exclusion_criteria`.

    Returns:
        Path to the CSV file.
    """
    condition_order = exclusion_table["condition"].map(
        {name: i for i, name in enumerate(CONDITION_VERSION_AND_BLOCK)}
    )
    audit_data = (
        exclusion_table.assign(_condition=condition_order)
        .sort_values(["_condition", "subject_id"], kind="stable")
        .reindex(columns=QC_AUDIT_COLUMNS)
    )
    audit_path = Path(output_dir) / QC_AUDIT_FILENAME
    audit_data.to_csv(audit_path, index=False)
    return audit_path


def summarise_exclusions(exclusion_table):
    """Count participants, exclusions and each criterion, condition by condition.

    Gives one row per condition, in task order, then an `any_condition` row
    that counts each participant once. One participant can meet several
    criteria, so the criterion counts can add up to more than `excluded`.

    In a condition row, `participants` is everyone with a row for that
    condition, `excluded` those excluded from it, `kept` those saved to its
    `.npz` file, and each criterion column the number who met that criterion
    there. In the `any_condition` row, `participants` counts each participant
    once, `excluded` those excluded from at least one condition, `kept` those
    kept in every condition, and each criterion column those who met it in at
    least one condition.

    Args:
        exclusion_table: Table with one row per participant and condition,
            as returned by `assess_exclusion_criteria`.

    Returns:
        Table with the columns `condition`, `participants`, `excluded`,
        `kept`, and one column per criterion in EXCLUSION_CRITERIA.
    """
    flag_columns = ["excluded", *EXCLUSION_CRITERIA]
    flags = exclusion_table[flag_columns].astype(bool)

    def count_row(condition_name, flags_here):
        """Count participants, exclusions and criteria for one summary row.

        Args:
            condition_name: Label for the row: a condition name, or
                ANY_CONDITION for the row across all conditions.
            flags_here: True/false table with one row per participant (or
                participant and condition) and the columns `excluded` and
                one per criterion in EXCLUSION_CRITERIA.

        Returns:
            Dict with the row's `condition`, `participants`, `excluded`,
            `kept` and one count per criterion.
        """
        counts = flags_here.sum().astype(int)
        return {
            "condition": condition_name,
            "participants": len(flags_here),
            "excluded": counts["excluded"],
            "kept": len(flags_here) - counts["excluded"],
            **{criterion: counts[criterion] for criterion in EXCLUSION_CRITERIA},
        }

    rows = []
    for condition_name in CONDITION_VERSION_AND_BLOCK:
        in_condition = exclusion_table["condition"] == condition_name
        if in_condition.any():
            rows.append(count_row(condition_name, flags[in_condition]))
    # Across conditions, a participant counts as meeting a criterion (or as
    # excluded) if that is true in at least one of their conditions.
    per_participant = flags.groupby(exclusion_table["subject_id"]).any()
    rows.append(count_row(ANY_CONDITION, per_participant))
    columns = ["condition", "participants", "excluded", "kept", *EXCLUSION_CRITERIA]
    return pd.DataFrame(rows, columns=columns)


def validate_input_file(path_str, label):
    """Stop with a clear message if an input data file does not exist.

    Args:
        path_str: Path to the CSV file, as text.
        label: Name of the dataset, used in the error message (for example
            "Main data").

    Raises:
        FileNotFoundError: If there is no file at the path.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} CSV not found: {path}. "
            "Update `scripts/config.toml` or pass the correct path with the CLI flags."
        )


def assess_exclusion_criteria(
    data_dict,
    manual_exclusions,
    rolling_window=10,
    std_threshold=0.2,
    mean_low=1.25,
    mean_high=1.75,
    input_path="",
):
    """Check every exclusion criterion for every participant in every condition.

    Each criterion is checked on its own, before anyone is removed, so the
    table shows all the criteria a participant met, not only the first one
    found. A participant is excluded from a condition if they meet any of:

    - `on_manual_exclusion_list`: their ID is in the config's
      `excluded_subjects` list.
    - `low_choice_variability`: the standard deviation of the rolling mean of
      their choices in this condition is below `std_threshold`, so they
      rarely switched between options.
    - `extreme_mean_choice`: their mean choice in this condition is outside
      `mean_low` to `mean_high`, so they mostly chose one option.
    - `low_engagement_in_another_condition`: one of the two low-engagement
      criteria above is met in another condition. A low-engagement participant is
      removed from every condition.
    - `incomplete_trials`: they have fewer than `BLOCK_TRIAL_COUNT` trials in
      this condition (0 if they have no trials in it at all).

    The default choice bounds (1.25 to 1.75) assume the raw choice column is
    coded 1 or 2, so a mean of 1.5 means the participant chose each option
    equally often.

    Args:
        data_dict: Dictionary mapping each condition name to a table of trials.
        manual_exclusions: Participant IDs to exclude by hand.
        rolling_window: Number of trials in the rolling mean.
        std_threshold: Minimum standard deviation of the rolling mean.
        mean_low: Lower bound for an acceptable mean choice, on the raw 1-or-2
            choice coding.
        mean_high: Upper bound for an acceptable mean choice, on the raw 1-or-2
            choice coding.
        input_path: Path of the input CSV, recorded in the table.

    Returns:
        Table with one row per participant and condition, and the columns
        named in QC_AUDIT_COLUMNS.

    Raises:
        ValueError: If, in any condition, the most trials any participant has
            is not `BLOCK_TRIAL_COUNT`.
    """
    console.print("\n[bold cyan]Checking exclusion criteria...[/bold cyan]")

    # A participant who did a task version should have both of its blocks. To
    # also catch someone with no trials at all in one block, list every
    # participant seen in either block of each version.
    participants_by_version = {}
    for condition_name, data in data_dict.items():
        version = CONDITION_VERSION_AND_BLOCK[condition_name][0]
        participants_by_version.setdefault(version, []).extend(
            data["subject_id"].unique()
        )

    rows = []
    for condition_name, data in data_dict.items():
        # The task design has BLOCK_TRIAL_COUNT trials per block. If the
        # participant with the most trials has a different number, the data do
        # not match the design, so stop rather than exclude everyone.
        trial_counts = data["subject_id"].value_counts()
        if not trial_counts.empty and trial_counts.max() != BLOCK_TRIAL_COUNT:
            most = trial_counts.max()
            holders = sorted(trial_counts[trial_counts == most].index)
            raise ValueError(
                f"Condition '{condition_name}': the most trials any participant has is "
                f"{most}, but the task design has {BLOCK_TRIAL_COUNT} per "
                f"block. Participants with {most}: {holders}"
            )

        trials_by_subject = {
            subject: trials
            for subject, trials in data.groupby("subject_id", sort=False)
        }
        version = CONDITION_VERSION_AND_BLOCK[condition_name][0]
        for subject in dict.fromkeys(participants_by_version[version]):
            choices = trials_by_subject.get(subject, data.iloc[:0])["choice"]
            # The rolling mean over `rolling_window` trials tracks the
            # participant's recent preference. If it barely moves (low standard
            # deviation), they rarely switched options. A participant with no
            # trials here gets NaN, which meets neither choice criterion.
            rolling_mean = choices.rolling(window=rolling_window, min_periods=1).mean()
            rows.append(
                {
                    "input_path": str(input_path),
                    "condition": condition_name,
                    "subject_id": subject,
                    "observed_trials": len(choices),
                    "expected_trials": BLOCK_TRIAL_COUNT,
                    "rolling_mean_std": rolling_mean.std(),
                    "choice_mean": choices.mean(),
                    "rolling_window": rolling_window,
                    "std_threshold": std_threshold,
                    "mean_low": mean_low,
                    "mean_high": mean_high,
                }
            )
    table = pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)

    # Check each criterion separately, so meeting one never hides another.
    table["on_manual_exclusion_list"] = table["subject_id"].isin(manual_exclusions)
    table["low_choice_variability"] = table["rolling_mean_std"] < std_threshold
    table["extreme_mean_choice"] = (table["choice_mean"] < mean_low) | (
        table["choice_mean"] > mean_high
    )
    # Low engagement in any one condition removes the participant from every
    # condition. Count the conditions in which each participant
    # met a low-engagement criterion, then leave out the current condition.
    low_engagement_here = table["low_choice_variability"] | table["extreme_mean_choice"]
    by_subject = low_engagement_here.groupby(table["subject_id"])
    low_engagement_count = by_subject.transform("sum")
    table["low_engagement_in_another_condition"] = (
        low_engagement_count - low_engagement_here > 0
    )
    table["incomplete_trials"] = table["observed_trials"] < BLOCK_TRIAL_COUNT

    met = table[EXCLUSION_CRITERIA]
    table["excluded"] = met.any(axis=1)
    # List the criteria met, for example "low_choice_variability;
    # incomplete_trials". Empty for an included participant.
    table["exclusion_reasons"] = [
        "; ".join(c for c, is_met in zip(EXCLUSION_CRITERIA, flags) if is_met)
        for flags in met.to_numpy()
    ]

    for condition_name, rows_here in table.groupby("condition", sort=False):
        counts = ", ".join(
            f"{criterion} {rows_here[criterion].sum()}"
            for criterion in EXCLUSION_CRITERIA
        )
        console.print(
            f"  [yellow]{condition_name}[/yellow]: {rows_here['excluded'].sum()} of "
            f"{len(rows_here)} participants excluded ({counts})"
        )

    return table


def excluded_from_condition(exclusion_table, condition_name):
    """List the participants the exclusion table excludes from one condition.

    Args:
        exclusion_table: Table returned by `assess_exclusion_criteria`.
        condition_name: Condition name.

    Returns:
        List of participant IDs.
    """
    rows = exclusion_table[
        (exclusion_table["condition"] == condition_name) & exclusion_table["excluded"]
    ]
    return rows["subject_id"].tolist()


def load_and_clean_data(csv_path, limit_subjects=None, new_rlt_only=True):
    """Read a behavioural data CSV and select the study's participants.

    Keeps only participants whose ID contains "new_RLT_" (or, if `new_rlt_only`
    is False, only those whose ID does not). Optionally keeps the first few
    participants of each task version. These choose which data to analyse;
    they are not exclusions. Exclusions, including the manual exclusion list,
    are decided later by `assess_exclusion_criteria`.

    Args:
        csv_path: Path to the behavioural data CSV.
        limit_subjects: Number of participants to keep per task version, taken
            in order of appearance. If None, all participants are kept. If the
            data have no task version column, the limit applies to all
            participants together.
        new_rlt_only: If True, keep only participants with "new_RLT_" in their
            ID. If False, drop those participants.

    Returns:
        Table of trials for the selected participants.
    """
    console.print(f"\n[bold green]Loading dataset from:[/bold green] {csv_path}")

    data = pd.read_csv(csv_path)
    console.print(f"  Initial rows: {len(data)}, subjects: {data['subject_id'].nunique()}")

    # Filter by subject ID prefix
    if new_rlt_only:
        data = data[data['subject_id'].str.contains('new_RLT_', na=False)]
        console.print(f"  After new_RLT_ filter: {len(data)} rows, {data['subject_id'].nunique()} subjects")
    else:
        data = data[~data['subject_id'].str.contains('new_RLT_', na=False)]
        console.print(f"  After excluding new_RLT_ subjects: {len(data)} rows, {data['subject_id'].nunique()} subjects")
    
    if limit_subjects is not None:
        console.print(f"  Limiting to first {limit_subjects} subjects per task_version...")
        subjects_to_keep = []
        if 'task_version' in data.columns:
            for version, group in data.groupby('task_version'):
                # Get unique subjects while preserving their original order of appearance
                unique_subjects = group['subject_id'].drop_duplicates().tolist()
                subjects_to_keep.extend(unique_subjects[:limit_subjects])
        else:
            console.print("  [yellow]Warning: 'task_version' column not found. Limiting overall total subjects instead.[/yellow]")
            unique_subjects = data['subject_id'].drop_duplicates().tolist()
            subjects_to_keep.extend(unique_subjects[:limit_subjects])
            
        data = data[data['subject_id'].isin(subjects_to_keep)]
        console.print(f"  After limiting: {len(data)} rows, {data['subject_id'].nunique()} subjects")

    return data


def drop_repeated_final_trial(data):
    """Drop the repeated log row at the end of each participant's block 2.

    The task logs block 2's final trial twice: every participant's block 2
    has ``BLOCK_TRIAL_COUNT + 1`` rows, and the extra row (the highest
    ``trial_num``) repeats the previous row's outcome and ball colour a
    millisecond later. This drops that row and nothing else. Blocks with
    ``BLOCK_TRIAL_COUNT`` rows or fewer are left unchanged; fewer rows are
    handled later by the incomplete-trials exclusion. Any other extra rows
    stop the run.

    Args:
        data: Trial rows with ``subject_id``, ``task_version``, ``block``,
            ``trial_num``, ``exploded`` and ``ball_color`` columns.

    Returns:
        A copy of ``data`` without the repeated rows.

    Raises:
        ValueError: If a required column is missing, or a participant's block
            has extra rows that are not the documented repeat.
    """
    required = ['subject_id', 'task_version', 'block', 'trial_num', 'exploded', 'ball_color']
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"Cannot check for the repeated final trial: missing columns {missing}."
        )

    rows_to_drop = []
    for (subject_id, task_version, block), rows in data.groupby(
        ['subject_id', 'task_version', 'block'], sort=False
    ):
        n_rows = len(rows)
        if n_rows <= BLOCK_TRIAL_COUNT:
            continue
        where = (f"participant {subject_id}, version {task_version}, block {block}: "
                 f"{n_rows} rows, expected {BLOCK_TRIAL_COUNT}")
        if n_rows > BLOCK_TRIAL_COUNT + 1:
            raise ValueError(f"Too many trials for {where}.")
        if block != 2:
            raise ValueError(f"Extra trial outside block 2 for {where}.")
        last_two = rows.sort_values('trial_num').iloc[-2:]
        previous, final = last_two.iloc[0], last_two.iloc[1]
        if (final['exploded'] != previous['exploded']
                or final['ball_color'] != previous['ball_color']):
            raise ValueError(
                f"Extra final trial does not repeat the trial before it for {where}."
            )
        rows_to_drop.append(last_two.index[1])

    console.print(
        f"  Dropped the repeated final trial of block 2 for "
        f"{len(rows_to_drop)} participant(s)"
    )
    return data.drop(index=rows_to_drop)


def split_by_condition(data):
    """Split the trials into one table per condition (task version and block).

    The result always holds all eight conditions (versions 1-4, two blocks
    each), in the order of CONDITION_VERSION_AND_BLOCK.

    Keeps only the columns subject_id, choice, trial_num, exploded, task_version
    and block. Copies the `exploded` column to a new `outcomes_0` column and
    then drops `exploded`, so the returned tables hold subject_id, choice,
    trial_num, task_version, block and outcomes_0.

    Args:
        data: Table of trials with the columns subject_id, choice, trial_num,
            exploded, task_version and block.

    Returns:
        Dictionary mapping each condition name to its table of trials.
    """
    # Select relevant columns and create outcomes_0
    data = data[['subject_id', 'choice', 'trial_num', 'exploded', 'task_version', 'block']].copy()
    data['outcomes_0'] = data['exploded']
    data = data.drop(columns=['exploded'])
    
    data_dict = {}
    for condition_name in CONDITION_VERSION_AND_BLOCK:
        version, block = CONDITION_VERSION_AND_BLOCK[condition_name]
        data_dict[condition_name] = data[
            (data['task_version'] == version) & (data['block'] == block)
        ]

    console.print(f"\n[bold cyan]Split into {len(data_dict)} conditions:[/bold cyan]")
    for name, df in data_dict.items():
        console.print(f"  {name}: {df['subject_id'].nunique()} subjects, {len(df)} trials")
    
    return data_dict


def prepare_and_save_condition(condition_name, condition_data, subjects_to_exclude,
                               output_dir):
    """Prepare one condition's trials and save them as a model-ready ``.npz`` file.

    Removes the participants excluded from this condition, checks that every
    remaining participant has ``BLOCK_TRIAL_COUNT`` trials numbered without
    gaps, and saves reward-coded arrays.

    Args:
        condition_name: Condition name, used for the file name and messages.
        condition_data: Trial rows for this condition.
        subjects_to_exclude: Participant IDs excluded from this condition, as
            marked in the exclusion table.
        output_dir: Directory for the ``.npz`` file.

    Returns:
        The prepared, reward-coded data dictionary.

    Raises:
        ValueError: If no participants remain, a remaining participant does
            not have ``BLOCK_TRIAL_COUNT`` trials, or a participant's trial
            numbers have gaps or repeats.
    """
    console.print(f"\n[bold yellow]Preparing condition: {condition_name}[/bold yellow]")

    condition_data = condition_data[
        ~condition_data['subject_id'].isin(subjects_to_exclude)
    ].copy()
    if condition_data.empty:
        raise ValueError(
            f"Condition '{condition_name}': no participants remain after exclusions."
        )
    # The exclusion table removes everyone with fewer trials than the design,
    # so a remaining participant with a different count means the table and
    # the trial data disagree.
    trial_counts = condition_data['subject_id'].value_counts()
    wrong_count = sorted(trial_counts.index[trial_counts != BLOCK_TRIAL_COUNT])
    if wrong_count:
        raise ValueError(
            f"Condition '{condition_name}': participants {wrong_count} remain after "
            f"exclusions but do not have {BLOCK_TRIAL_COUNT} trials."
        )

    # Ensure outcomes_0 is numeric
    condition_data['outcomes_0'] = pd.to_numeric(condition_data['outcomes_0'], errors='coerce')
    
    # Reset trial numbers for each subject
    condition_data['trial_num'] = condition_data.groupby('subject_id')['trial_num'].transform(
        lambda x: x - x.min()
    )
    # Each participant's trials must now be numbered 0 to n-1 with no gaps or
    # repeats; a gap would leave an empty, invented trial in the arrays.
    numbering = condition_data.groupby('subject_id')['trial_num']
    n_rows = numbering.size()
    not_consecutive = sorted(n_rows.index[
        (numbering.max() + 1 != n_rows) | (numbering.nunique() != n_rows)
    ])
    if not_consecutive:
        raise ValueError(
            f"Condition '{condition_name}': trial numbers are not consecutive "
            f"for participants {not_consecutive}"
        )
    
    console.print(f"  Subjects after exclusions: {condition_data['subject_id'].nunique()}")
    
    # Prepare for model fitting
    data_prepared = prepare_data_for_fitting(
        condition_data,
        use_jax=False,  # Save as numpy for portability
    )
    
    console.print(
        f"  Data shape: choices={data_prepared['choices'].shape}, "
        f"outcomes_0={data_prepared['outcomes_0'].shape}"
    )
    
    # Convert from explosion coding to reward/safe coding for model-facing arrays
    data_prepared = convert_empirical_to_reward_coded(data_prepared)

    # Extract original subject IDs in the order subjects first appear (matches prepare_data_for_fitting ordering)
    original_subject_ids = condition_data['subject_id'].drop_duplicates().values

    # Save to disk
    output_path = output_dir / f"{condition_name}.npz"
    np.savez_compressed(
        output_path,
        outcomes_0=data_prepared['outcomes_0'],
        outcomes_1=data_prepared['outcomes_1'],
        choices=data_prepared['choices'],
        subject_ids=original_subject_ids,
        outcome_coding=OUTCOME_CODING_REWARD_SAFE,
    )
    
    console.print(f"  [green]✓ Saved to {output_path}[/green]")
    
    return data_prepared


def main():
    """Prepare the behavioural data and save one .npz file per condition.

    Reads the command-line options and the config file, then loads the data
    file, drops the repeated final trial, splits the trials into all eight
    conditions (versions 1-4), checks every exclusion criterion for every
    participant, and saves the prepared arrays for each condition without the
    excluded participants. Finally, it writes the exclusion table (one row per
    participant and condition).

    The config file must exist and be readable. If it is missing or cannot be
    read, the script raises an error before it writes anything. For a setting
    the file leaves out, built-in defaults are used: an empty manual exclusion
    list and the default quality-check thresholds (rolling window of 10 trials,
    minimum standard deviation of 0.2, and mean choice bounds of 1.25 to 1.75).
    The main data file and the output folder have no default: give them on the
    command line or in the config file.

    A participant flagged as low-engagement in any condition is excluded from
    every condition.

    Raises:
        ValueError: If neither the command line nor the config file gives the
            main data file or the output folder.
    """
    parser = argparse.ArgumentParser(
        description="Prepare behavioral data for model fitting"
    )
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/config.toml',
        help='Path to configuration file'
    )
    # CLI Overrides
    parser.add_argument(
        '--main-data',
        type=str,
        default=None,
        help='Path to main dataset CSV (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save prepared data files (overrides config)'
    )
    parser.add_argument(
        '--limit-subjects',
        type=int,
        default=None,
        help='Limit to first N subjects per task_version (prior to exclusions)'
    )
    parser.add_argument(
        '--no-new-rlt',
        action='store_true',
        help='Exclude subjects with "new_RLT_" in their ID (by default only new_RLT_ subjects are included)'
    )
    
    args = parser.parse_args()
    
    console.print("\n[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]   Data Preparation Pipeline[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]\n")
    
    # --- Load Configuration ---
    # A missing or unreadable config file stops the run here, before anything
    # is written.
    console.print(f"[cyan]Loading configuration from {args.config}[/cyan]")
    config = load_pipeline_config(args.config)
    
    # --- Resolve Settings ---
    # Project paths from config
    project_config = config.get('project', {})
    data_config = config.get('data', {})
    
    # Resolve main data path
    if args.main_data:
        main_data_path = args.main_data
    elif 'main_data' in data_config and str(data_config['main_data']).strip():
        path = str(data_config['main_data']).strip()
        if 'data_dir' in project_config and not Path(path).is_absolute():
            main_data_path = str(Path(project_config.get('base_dir', '.')) / project_config['data_dir'] / path)
        else:
            main_data_path = path
    else:
        raise ValueError(
            "No main data file given. Pass --main-data or set `main_data` under [data] "
            f"in {args.config}."
        )
        
    # Resolve output directory
    if args.output_dir:
        output_dir_path = args.output_dir
    elif 'data_dir' in project_config:
         output_dir_path = str(Path(project_config.get('base_dir', '.')) / project_config['data_dir'] / 'prepared')
    else:
        raise ValueError(
            "No output folder given. Pass --output-dir or set `data_dir` under "
            f"[project] in {args.config}."
        )
        
    # Get exclusions and thresholds
    excluded_subjects = data_config.get('excluded_subjects', DEFAULT_EXCLUDED_SUBJECTS)
    thresholds = data_config
    
    # Create output directory
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Output directory:[/bold] {output_dir.absolute()}\n")

    validate_input_file(main_data_path, 'Main data')

    # Quality-check thresholds: each one falls back to its own default if the
    # config file leaves it out.
    mean_low, mean_high = thresholds.get('mean_choice_bounds', [1.25, 1.75])
    criteria_settings = dict(
        manual_exclusions=excluded_subjects,
        rolling_window=thresholds.get('rolling_window', 10),
        std_threshold=thresholds.get('std_threshold', 0.2),
        mean_low=mean_low,
        mean_high=mean_high,
    )

    # ===== LOAD, SPLIT AND CHECK =====
    console.print("\n[bold blue]━━━ ALL CONDITIONS (Versions 1-4) ━━━[/bold blue]")
    main_data = load_and_clean_data(
        main_data_path, limit_subjects=args.limit_subjects,
        new_rlt_only=not args.no_new_rlt,
    )
    main_data = drop_repeated_final_trial(main_data)

    condition_dict = split_by_condition(main_data)
    exclusion_table = assess_exclusion_criteria(
        condition_dict, input_path=main_data_path, **criteria_settings,
    )

    # Prepare each condition without the participants the table excludes
    for condition_name, condition_data in condition_dict.items():
        prepare_and_save_condition(
            condition_name, condition_data,
            excluded_from_condition(exclusion_table, condition_name), output_dir,
        )

    audit_path = _write_qc_exclusion_audit(output_dir, exclusion_table)
    console.print(f"[bold]Exclusion table:[/bold] {audit_path} ({len(exclusion_table)} rows)")
    counts = summarise_exclusions(exclusion_table)
    counts_path = output_dir / QC_COUNTS_FILENAME
    counts.to_csv(counts_path, index=False)
    console.print(f"[bold]Exclusion counts:[/bold] {counts_path}")
    console.print(counts.to_string(index=False))
    console.print("\n[bold green]✓ Data preparation complete![/bold green]\n")
    console.print(f"[bold]Prepared files saved to:[/bold] {output_dir.absolute()}")
    console.print(f"[bold]Total conditions:[/bold] {len(list(output_dir.glob('*.npz')))}\n")


if __name__ == '__main__':
    main()
