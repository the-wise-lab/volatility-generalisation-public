# 3. Model selection and recovery

These analyses use the four second-block conditions (`StableV1`,
`VolatileV2`, `VolatileV3_2`, `StableV4_2`), where each participant's
previous block type is known. They need the prepared data from
[step 1](1-data-preparation.md).

- **Integrated model selection** fits each model to all four conditions at
  once. The model has group-level effects of the current block, the previous
  block and their interaction. The models are compared by leave-one-out
  cross-validation over trials (PSIS-LOO). Each participant's first trial is
  left out of the comparison.
- **Integrated model recovery** simulates data from each model in the same
  design and checks whether the same comparison picks the model that made
  the data, and whether the parameters and group-level effects are
  recovered.
- **Parameter recovery** is a simpler check. For each model and condition
  separately, it simulates participants, fits the model back, and compares
  the fitted parameters with the true ones.

All three take their default models and settings from `scripts/config.toml`.
Each `submit_*.py` script sends Slurm jobs; the matching `run_*.py` script
runs one fit on your own computer.

## Integrated model selection

```bash
uv run python scripts/submit_integrated_model_selection.py
uv run python scripts/compile_integrated_model_selection.py
```

The first command sends one Slurm job for each model. When all the jobs have
ended, the second command compares the models. To fit one model on your own
computer:

```bash
uv run python scripts/run_integrated_model_selection.py \
    --model rescorla_wagner_single_value \
    --conditions StableV1 VolatileV2 VolatileV3_2 StableV4_2
```

The `converged` column of `model_selection_loo.csv` shows whether each fit
converged, with the same rules as in
[step 2](2-model-fitting.md#convergence). Fits that did not converge stay in
the comparison. If a fit does not converge, you can try again from a starting
point found by Pathfinder, with `--pathfinder-init`.

`scripts/run_integrated_posterior_predictive_check.py` uses the saved
posteriors (no refitting) to compare the predicted and observed choices of
every model.

Outputs:

- `results/integrated_model_selection/<model>/`: `posterior.nc`,
  `metadata.json`, `diagnostics.json`, and group-level and participant-level
  parameter summaries.
- `results/integrated_model_selection/model_comparison/`: the LOO table
  (`model_selection_loo.csv`), the model comparison (`model_selection_compare.csv`),
  the Pareto k counts (`pareto_k_table.csv`, counted in trials), and figures
  of the ranking and stacking weights.

## Integrated model recovery

```bash
uv run python scripts/submit_integrated_model_recovery.py
uv run python scripts/compile_integrated_model_recovery.py --replicates 10
```

The first command sends one job for each replicate, simulated model and
fitted model. The second command compiles the results. `--replicates` gives
the number of replicates; the compile stops with an error if any fit is
missing or if there are fits it does not expect. To run one fit on your own
computer:

```bash
uv run python scripts/run_integrated_model_recovery_fit.py \
    --replicate 0 \
    --simulated-model rescorla_wagner_single_value \
    --fitted-model dual_lr_single_value
```

For each simulated data set, the winning model is the fitted model with the
highest `elpd_loo`. Fits that did not converge or have high Pareto k values
are included, and are marked in the tables.

The compiled tables, in `results/integrated_model_recovery/summary/tables/`:

- `model_recovery_loo.csv` and `model_recovery_winners.csv`: the score and
  winner of every fit;
- `model_recovery_confusion_matrix.csv` and
  `model_recovery_inversion_matrix.csv`: how often each simulated model is
  recovered;
- `parameter_recovery.csv` and `parameter_recovery_summary.csv`: true and
  fitted participant parameters;
- `group_effect_recovery.csv`: true and fitted group-level effects (group
  mean, current block, previous block, interaction and between-participant
  spread), with 95% equal-tailed intervals, on the scale before each
  parameter is mapped into its bounds.

These tables are included in the repository. Figures S1 to S3 are drawn from
them (see [step 5](5-publication-figures.md#recovery-figures)).

## Parameter recovery

```bash
uv run python scripts/submit_parameter_recovery.py
```

This sends one job for each model and condition. Each job simulates 50
participants with that condition's outcome sequence, then fits the model back.
Use `--models`, `--conditions` and `--dry-run` as in step 2. If results
already exist, add `--overwrite` to replace them or `--skip-existing` to keep
them. To run on your own computer, with small test settings:

```bash
uv run python scripts/run_parameter_recovery.py \
    --models rescorla_wagner_single_value \
    --conditions VolatileV1 \
    --n-subjects 5 --num-warmup 100 --num-samples 100 --num-chains 1
```

Results go to `results/parameter_recovery/<condition>/<model>/`. The
convergence of each fit is recorded in `run_metadata.json` and in
`parameter_recovery_summary.csv`. A fit with one chain always counts as not
converged, because R-hat needs at least two chains.
