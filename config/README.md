# Settings

## `model_bounds.toml`

The bounds and priors of every model parameter. The `[common]` table holds
parameters shared by all models (the softmax `temperature`). Each other table
is named after a model and holds that model's parameters:

| Table | Parameters |
|---|---|
| `[rescorla_wagner_single_value]` | `learning_rate` |
| `[dual_lr_single_value]` | `learning_rate_pos`, `learning_rate_neg` |
| `[pearce_hall_single_value]` | `kappa`, `gamma` |
| `[leaky_beta_single_value]` | `tau`, `decay` |
| `[leaky_beta_asymmetric_single_value]` | `tau_pos`, `tau_neg`, `decay` |

Each parameter has these fields:

| Field | Meaning |
|---|---|
| `min`, `max` | The bounds of the parameter. |
| `prior_mu_mean`, `prior_mu_sigma` | Mean and standard deviation of the prior on the group mean (on the unbounded scale). |
| `prior_sigma_scale` | Scale of the prior on the group standard deviation. |
| `prior_offset_sigma` | Standard deviation of the prior on each participant's offset. |
| `prior_beta_sigma` | Integrated models only: standard deviation of the priors on the current-block, previous-block and interaction effects. If not set, it is equal to `prior_mu_sigma`. |

Two bounds need explanation:

- The lower bound of `temperature` is 0.02. Many participants choose very
  consistently, and a higher bound (0.2) forced the other parameters to
  account for this, which made sampling poor.
- The lower bound of Pearce-Hall `gamma` is 0.2. With a very small `gamma`,
  the model becomes Rescorla-Wagner with a slowly decaying learning rate. This
  gave a second posterior mode, and the chains did not converge.

A change to a bound also moves the prior on the group mean, because a
`prior_mu_mean` of 0 is the middle of the range from `min` to `max`.

Every fit saves the path and a SHA-256 hash of this file in its
`metadata.json`, so each result can be matched to the settings it used. To
use a different file, give `--model-config` to `fit_single.py`,
`run_integrated_model_selection.py`, `run_integrated_model_recovery_fit.py` or
`run_parameter_recovery.py`. Slurm jobs always use this file.

## `questionnaire_scoring_key.csv`

How each of the 136 questionnaire items is scored in the regression analysis.
One row per item:

| Column | Meaning |
|---|---|
| `questionnaire_key` | The questionnaire. |
| `question_id` | The item's `questionId` in the questionnaire data. |
| `item_label` | A short label of the item (the published wording is copyrighted). Attention checks keep their full wording. |
| `scoring` | `forward`, `reverse`, `not_scored_filler` or `attention_check`. |

- `forward` items score the answer as given.
- `reverse` items score `scaleMin + scaleMax - rawResponse`. The EAT is stored
  with 0 = "Always" and 5 = "Never", so both EAT items are reversed.
- `not_scored_filler` (the four LOT-R fillers) and `attention_check` items are
  left out of the totals and the factor analysis.

The regression script stops if this key and the questionnaire data do not
list the same items.

| Questionnaire | Forward | Reverse | Filler | Attention check |
|---|---|---|---|---|
| `STICSA` | 21 | none | none | `STICSA__18` |
| `IUS_12` | 11 | none | none | `IUS_12__11` |
| `PHQ_8` | 8 | none | none | `PHQ_8__AC1` |
| `GAD_7` | 7 | none | none | `GAD_7__AC1` |
| `LOT_R` | 3 | `LOT_R__3`, `__8`, `__10` | `LOT_R__2`, `__5`, `__7`, `__9` | `LOT_R__6` |
| `BIS` | 6 | `BIS__25`, `__26`, `__30` | none | none |
| `AES` | 0 | `AES__33` to `__38`, `__40` | none | `AES__39` |
| `EAT` | 0 | `EAT_51`, `EAT_52` | none | none |
| `SRDS` | 2 | `SRDS__1`, `__3`, `__5` to `__9` | none | none |
| `STAI` | 7 | `STAI__10`, `__11`, `__15`, `__17`, `__20`, `__21`, `__23` | none | `STAI__13` |
| `OCI` | 11 | none | none | `OCI__42` |
| `LSAS` | 19 | none | none | none |
| Total | 95 | 29 | 4 | 8 |

In the LOT-R, an attention check is at position 6, so from item 6 on the item
numbers are one higher than the published numbers. The reverse items
`LOT_R__3`, `__8` and `__10` are published items 3, 7 and 9, and the fillers
are published items 2, 5, 6 and 8.

The scoring directions agree with the published keys: STAI Form Y-2
(Spielberger et al. 1983), LOT-R (Scheier, Carver & Bridges 1994), BIS-11
(Patton et al. 1995), the self-rated AES (Marin et al. 1991), the Zung
Self-Rating Depression Scale (Zung 1965) and EAT-26 (Garner et al. 1982). The
STICSA, IUS-12, PHQ-8, GAD-7, OCI-R and LSAS have no reverse items.
