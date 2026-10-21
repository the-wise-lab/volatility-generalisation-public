# Data

## Behavioural data

`behavioural_data.csv.zip` holds one CSV file, one row per trial:

| Column | Meaning |
|---|---|
| `subject_id` | Participant ID, for example `new_RLT_v1_s0001` (`v1` is the task version). The same ID is used in every file. |
| `trial_num` | Trial number: 0-119 in block 1 and 120-240 in block 2 (see below). |
| `choice` | The option chosen. |
| `exploded` | The outcome: 1 = explosion, 0 = safe. |
| `task_version` | Task version, 1-4. |
| `block` | Block, 1 or 2. |
| `ball_color` | Colour of the chosen ball. |

`scripts/prepare_data.py` converts the outcomes to reward coding
(1 = safe, 0 = explosion) before fitting.

### The 8 conditions

| Condition | `task_version` | `block` | Block type | Previous block |
|---|---|---|---|---|
| `VolatileV1` | 1 | 1 | Volatile | — |
| `StableV1` | 1 | 2 | Stable | Volatile |
| `StableV2` | 2 | 1 | Stable | — |
| `VolatileV2` | 2 | 2 | Volatile | Stable |
| `VolatileV3` | 3 | 1 | Volatile | — |
| `VolatileV3_2` | 3 | 2 | Volatile | Volatile |
| `StableV4` | 4 | 1 | Stable | — |
| `StableV4_2` | 4 | 2 | Stable | Stable |

The four conditions with a previous block are the second-block conditions
used in the integrated analyses.

### The incomplete-block rule

Each block has 120 trials. The task recorded block 2's last trial twice, so
block 2 has trial numbers 120-240: trial 240 repeats trial 239. Data
preparation removes trial 240. A participant with fewer than 120 trials in a
block is excluded from that condition.

## Questionnaire data

`questionnaire_data.csv.zip` holds one CSV file, one row per answer. The
main columns are:

| Column | Meaning |
|---|---|
| `assignedSubjectId` | Participant ID, the same as `subject_id` above. |
| `age`, `sex` | Demographics. |
| `questionnaireKey` | The questionnaire: `STICSA`, `IUS_12`, `PHQ_8`, `GAD_7`, `LOT_R`, `BIS`, `AES`, `EAT`, `SRDS`, `STAI`, `OCI` or `LSAS`. |
| `questionId` | The item, for example `STICSA__18`. |
| `questionText` | A short label for each scale item (the published wording is copyrighted). The 8 attention checks and the demographic questions keep their full wording. |
| `rawResponse`, `scaleMin`, `scaleMax` | The answer as a whole number, and the range of the scale. |

Do not use the `isAttentionCheck`, `reversesCoded` and `scoredResponse`
columns: they are not correct. How each item is scored is in
`config/questionnaire_scoring_key.csv` (see
[`../config/README.md`](../config/README.md#questionnaire_scoring_keycsv)).

## Data and results files

`prepared/qc_exclusion_audit.csv` has one row for each participant and
condition, including participants who were kept. Each exclusion criterion has
its own true/false column, so you can see every criterion a participant met:

| Column | Meaning |
|---|---|
| `condition`, `subject_id` | The condition and participant. |
| `observed_trials`, `expected_trials` | The participant's trials in this condition, and the 120 required. |
| `rolling_mean_std`, `choice_mean` | The two choice measures. |
| `rolling_window`, `std_threshold`, `mean_low`, `mean_high` | The thresholds used. |
| `on_manual_exclusion_list` | On the manual exclusion list. |
| `low_choice_variability` | `rolling_mean_std` is below `std_threshold`. |
| `extreme_mean_choice` | `choice_mean` is outside `mean_low` to `mean_high`. |
| `low_engagement_in_another_condition` | Either choice criterion is met in another condition. |
| `incomplete_trials` | Fewer than 120 trials. |
| `excluded` | True if any criterion is met. |
| `exclusion_reasons` | The criteria met. |

`prepared/qc_exclusion_counts.csv` has, for each condition, the number of
participants, the number excluded and kept, and the number who met each
criterion. One participant can meet several criteria. The last row,
`any_condition`, counts each participant once.

All other files in `prepared/` are made by `scripts/prepare_data.py`.
