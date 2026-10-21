# 2. Model fitting (per condition)

This step fits each of the 5 learning models to each of the 8 conditions
separately, then gathers the results into tables. It needs the prepared data
from [step 1](1-data-preparation.md).

The models are:

- `rescorla_wagner_single_value`: one learning rate;
- `dual_lr_single_value`: separate learning rates for positive and negative
  prediction errors;
- `pearce_hall_single_value`: a learning rate that follows recent surprise;
- `leaky_beta_single_value`: a beta-distribution estimate with memory decay;
- `leaky_beta_asymmetric_single_value`: the same, with separate update rates
  for good and bad outcomes.

Each model is a hierarchical Bayesian model fitted by MCMC. The parameter
bounds and priors are in `config/model_bounds.toml` (see
[`../config/README.md`](../config/README.md)).

## Fit the models

```bash
uv run python scripts/submit_jobs.py
```

This sends one Slurm job for each model and condition (40 jobs). The MCMC and
cluster settings come from `scripts/config.toml`. Useful options:

- `--models` and `--conditions`: fit only some models or conditions.
- `--dry-run`: show the jobs without sending them.
- `--overwrite` or `--skip-existing`: refit, or leave alone, a model and
  condition that already has results.

Check that every job ended with the state `COMPLETED`:

```bash
squeue -u $USER          # jobs still waiting or running
sacct -j <job ID> --format=JobID,State,Elapsed,MaxRSS
```

Each job writes its log to `logs/`.

### Without Slurm

Fit one model to one condition on your own computer:

```bash
uv run python scripts/fit_single.py \
    --model rescorla_wagner_single_value \
    --condition VolatileV1
```

The default settings match those of the paper. For a quick test, add for
example `--num-warmup 500 --num-samples 1000 --num-chains 2`.

## Convergence

Each fit records `"converged": false` in its `metadata.json` if any R-hat is
above 1.01, if R-hat or the effective sample size cannot be computed, or if
there are divergent transitions. An effective sample size below 400 gives a
warning only.

A fit that did not converge is a result, not an error: it keeps all its
outputs and is included in later steps, with its convergence recorded. A
model that cannot be fitted reliably is useful evidence in model comparison.
`fit_single.py` ends with exit code 2 for such a fit, so Slurm shows the job
as `FAILED`. To list these fits:

```bash
grep -l '"converged": false' results/*/metadata.json
```

Other problems, such as a missing or unreadable result, stop the analysis
with an error.

## Compile the results

```bash
uv run python scripts/compile_results.py --overwrite
```

This makes two sets of tables:

| Model | Output folder | Used by |
|---|---|---|
| `rescorla_wagner_single_value` | `results/compiled/` | the `learning_rate` regression and Figure 1 |
| `dual_lr_single_value` | `results/compiled_dual_lr/` | the `learning_rate_neg` and `learning_rate_pos` regressions |

`--overwrite` is needed because these folders already hold the paper's
`subject_parameters.csv`. To compile one model only, give both
`--analysis-model` and `--output-dir`, for example
`--analysis-model dual_lr_single_value --output-dir results/compiled_dual_lr`.

The compile needs a fit of each model in all 8 conditions. Parameters bounded
between 0 and 1 are analysed on the logit scale.

## Outputs

Each fit, in `results/<condition>_<model>/`: `posterior_samples.npz`,
`ppc_results.npz` (posterior predictive checks), `diagnostics.json`,
`parameter_summaries.json`, `metadata.json` and a posterior predictive plot.

Each compiled folder:

- `summary_table.csv`: one row for each condition and model;
- `subject_parameters.csv`: one row for each participant, condition and model;
- `parameter_overview.csv` and `param_dist_*.png`;
- `posterior_predictive_*.csv` and `.png`: posterior predictive checks across
  all fits;
- `parameters/<parameter>/`: for each parameter, a two-way ANOVA of current
  block type by previous block type in the four second-block conditions, with
  follow-up comparisons, cell summaries, normality and equal-variance checks,
  and an interaction plot.
