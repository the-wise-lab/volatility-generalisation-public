# 4. Regression analyses

`scripts/regression_analysis.py` relates a fitted learning rate to
questionnaire measures. Run it once for each learning-rate parameter:

```bash
uv run python scripts/regression_analysis.py --parameter learning_rate
```

| `--parameter` | Model | Input |
|---|---|---|
| `learning_rate` | `rescorla_wagner_single_value` | `results/compiled/subject_parameters.csv` |
| `learning_rate_neg` | `dual_lr_single_value` | `results/compiled_dual_lr/subject_parameters.csv` |
| `learning_rate_pos` | `dual_lr_single_value` | `results/compiled_dual_lr/subject_parameters.csv` |

The input files are included in the repository, so this step can run without
refitting the models. The questionnaire answers are read from
`data/questionnaire_data.csv.zip`. Each run writes to
`results/regression/<parameter>/` (change the root folder with
`--output-dir`). The factor-analysis checks are the same for all three runs,
so they go to one shared folder, `results/factor_analysis_checks/` (change it
with `--factor-checks-dir`).

The full list of analysis choices is at the top of
`scripts/regression_analysis.py`.

## Questionnaire scoring

`config/questionnaire_scoring_key.csv` says how each item is scored: forward,
reverse, filler (not scored), or attention check (not scored). Reverse items
are flipped within the scale range, so a high total always means more of the
trait. See [`../config/README.md`](../config/README.md#questionnaire_scoring_keycsv).

Several questionnaires are shortened versions of the published scales, so the
totals are not comparable with published norms.
`questionnaire_reliability.csv` gives Cronbach's alpha and the lowest
item-rest correlation of each questionnaire. A negative item-rest correlation
suggests an item with the wrong scoring direction.

## Attention checks

Participants who fail more than 1 of these 6 checks are excluded:

| Questionnaire | Question | Passing answers |
|---|---|---|
| STICSA | My hands have seven fingers. | "not at all", "a little" |
| PHQ-8 | Difficulty standing on one leg | "not at all", "several days" |
| GAD-7 | Being scared of my own hair | "not at all", "several days" |
| STAI | I feel like I have always been a human being | "very much so", "moderately" |
| AES | I have no feet. | "not at all", "slightly" |
| OCI | I frequently find myself cleaning my boat before going to bed. | "not at all", "a little" |

The IUS-12 check ("I feel surprised when surprising things happen to me") and
the LOT-R check ("I enjoy it when good things happen to me") are not used for
exclusion. A "wrong" answer to these may reflect symptoms, such as blunted
emotions or anhedonia, rather than inattention. Both are still left out of the
scores.

## Factor analysis

An exploratory factor analysis of the questionnaire items gives 4 factors.
The number was chosen from the elbow of the scree plot, and both random halves
of the participants reproduce all 4 factors. Parallel analysis and the
Cattell-Nelson-Gorsuch rule are also reported, but do not decide the number.
Items with an absolute skew of 2 or more are removed before the analysis.

The factors are named after what a high score means:

| Factor | Name | Main items |
|---|---|---|
| ML1 | Negative affect | STICSA, OCI, BIS, STAI, GAD-7, PHQ-8, SRDS |
| ML2 | Social anxiety | LSAS |
| ML3 | Apathy and amotivation | AES, SRDS and STAI, with LOT-R loading negatively |
| ML4 | Tolerance of uncertainty | IUS-12 (loading negatively), some AES and BIS |

Factor order and sign can change if the data change. The script checks that
STICSA, LSAS, AES and IUS-12 still load most strongly on ML1 to ML4, in the
directions above. If they do not, it stops, and the names must be chosen
again from the loadings.

A factor model that does not converge, or that has an item with a
communality of 0.995 or more (a Heywood case), stops the run.

The shared checks folder holds:

| File | Contents |
|---|---|
| `factor_count.csv`, `factor_count_by_position.csv` | Eigenvalues and the factor numbers from parallel analysis and the Cattell-Nelson-Gorsuch rule |
| `suitability.csv`, `kmo_by_item.csv` | Kaiser-Meyer-Olkin measure and Bartlett's test |
| `split_half_congruence.csv` | Tucker congruence of each factor between two random halves of the participants |
| `skew_filtered_items.csv` | Items removed for skew |
| `questionnaire_reliability.csv` | Cronbach's alpha and item-rest correlations |

A second run compares its checks with the files already in the folder and
stops if they differ. Delete the folder to make the checks again.

## Learning-rate regressions

The outcome is the change in logit learning rate from block 1 to block 2.
Each task version is analysed separately (about 80 participants each).

| Table | Model | Role |
|---|---|---|
| `lr_change_regression_results.csv` | Change on all four factors together | Main result |
| `lr_change_baseline_adjusted_results.csv` | Block 2 on block 1 and all four factors | Sensitivity check |
| `indiv_df.csv` | Change on one questionnaire total at a time | Exploratory |
| `factor_results_df.csv` | Change on one factor of the questionnaire totals at a time | Exploratory |

Each table is corrected for multiple comparisons with the Benjamini-Hochberg
false discovery rate, over all tests in that table (`p_fdr`, `Sig_fdr`). In
the baseline-adjusted table, the `lr_block1` rows are not part of the
correction.

The change-score model asks whether a trait predicts how much the learning
rate changes. The baseline-adjusted model asks whether a trait predicts the
block 2 learning rate among people with the same block 1 learning rate.
Because block 1 is measured with noise, the two can disagree (Lord's paradox).

Participants whose fits did not converge are kept. `counts.csv` gives, for
each version, the number of participants in the main regression and how many
of them have a fit that did not converge.

## Outputs

`results/regression/<parameter>/tables/` holds the tables above, plus
`counts.csv` (participant numbers, attention-check exclusions and
demographics), eigenvalues, factor loadings and factor scores.

`results/regression/<parameter>/figures/` holds the figures as SVG files: the
scree plot, factor loadings, the questionnaire correlation matrix, and the
learning-rate change against each questionnaire and each factor.
