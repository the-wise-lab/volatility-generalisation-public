# 5. Publication figures

## Main figures

```bash
uv run python scripts/make_publication_figures.py
```

This calculates the task-performance statistics and draws Figures 1 to 3,
each with the tables of the numbers it plots. It needs:

| Item | Input | Option to change it |
|---|---|---|
| Task performance | `data/prepared/` (from [step 1](1-data-preparation.md)) | `--prepared-data-dir` |
| Figure 1: single learning rate | `results/compiled/subject_parameters.csv` (included) | `--single-lr-csv` |
| Figure 2: model comparison and posterior predictive checks | `results/integrated_model_selection/model_comparison/tables/model_selection_compare.csv`, the main model's posterior, and `data/prepared/` | `--comparison-dir` |
| Figure 3: learning rates of the main model | `results/integrated_model_selection/dual_lr_single_value/posterior.nc` and its `metadata.json` | `--posterior-dir` |

The posterior files are not included, because they are large. To draw
Figures 2 and 3, first run the integrated model selection (see
[step 3](3-model-selection-and-recovery.md#integrated-model-selection)). The
script stops with an error that names any missing input.

Outputs go to `results/publication_figures/` (change it with `--output-dir`):
the figures in `figures/` as PDF, SVG and PNG, and their tables in `tables/`.

| Paper item | Figure file | Tables |
|---|---|---|
| Task performance | none (printed while the script runs) | `task-performance-source.csv`, `task-performance-summary.csv`, `task-performance-schedule-check.csv` |
| Figure 1 | `figure-1-single-learning-rate` | `figure-1-source.csv`, `figure-1-comparisons.csv` |
| Figure 2 | `figure-2-model-comparison-and-ppc` | `figure-2-source.csv`, `figure-2-ppc-source.csv` |
| Figure 3 | `figure-3-learning-rates` | `figure-3-source.csv`, `figure-3-population.csv`, `figure-3-effects.csv`, `figure-3-coefficients.csv` |

### The main model

Figures 2 and 3 use the dual learning-rate model (`dual_lr_single_value`).
In the model comparison it is statistically tied with the leaky beta
asymmetric model. It was chosen because it converged, it has fewer
parameters, and its parameters are the learning rates the study is about.

### Task performance

For each participant, over both blocks:

- **Optimal choices**: the proportion of trials on which they chose the ball
  with the higher chance of being safe, tested against 0.5 with a two-sided
  one-sample t-test.
- **Safe outcomes**: the proportion of safe outcomes, beside the best possible
  rate.
- **Participants above chance**: how many chose the better ball more often
  than chance (one-sided exact binomial test, p < .05).

The better ball is found from the task design: it is safe on 80% of trials,
and the balls swap after trial 60 in stable blocks and every 20 trials in
volatile blocks. The script stops if the data do not fit this design.

### Intervals and scales

All posterior intervals are 95% equal-tailed intervals (2.5th to 97.5th
percentile). The 95% confidence intervals in the task-performance tables are
t-based intervals for a mean across participants.

In Figure 3:

- `figure-3-population.csv` gives the learning rate (0 to 1 scale) of a
  typical participant in each condition: the sigmoid of the group mean plus
  the group-level volatility effects. This is not the mean learning rate
  across participants.
- `figure-3-effects.csv` gives the differences between these learning rates:
  volatile minus stable for the current and the previous block, and the
  difference-in-differences for the interaction.
- `figure-3-coefficients.csv` gives the model's group-level coefficients on
  the logit scale. Volatile blocks are coded +0.5 and stable blocks −0.5, so
  each volatility coefficient is a volatile-minus-stable difference. Use this
  table to report the volatility effects as the model estimates them.

## Recovery figures

```bash
uv run python scripts/make_recovery_figures.py --replicates 10
```

This draws Figures S1 to S3 from the compiled recovery tables in
`results/integrated_model_recovery/summary/tables/`, which are included in
the repository. It fits nothing. `--replicates` must match the number of
recovery replicates (10 in the paper).

- **S1, model recovery**: the confusion and inversion matrices.
- **S2, participant parameter recovery**: true against fitted values of every
  parameter of every model, with the median correlation across replicates.
- **S3, group-level effect recovery**: true against fitted group mean and
  volatility effects, with 95% intervals, on the scale before each parameter
  is mapped into its bounds.

Fits that did not converge are shown as open circles in S2 and S3, and are
counted in the S1 table.

Outputs go to `results/recovery_figures/`: `figures/` holds
`figure-s1-model-recovery`, `figure-s2-participant-parameter-recovery` and
`figure-s3-group-effect-recovery` as PDF, SVG and PNG; `tables/` holds
`figure-s1-source.csv`, `figure-s2-source.csv`, `figure-s2-correlations.csv`,
`figure-s3-source.csv` and `figure-s3-correlations.csv`.

To make the recovery tables again from new recovery fits, see
[step 3](3-model-selection-and-recovery.md#integrated-model-recovery).
