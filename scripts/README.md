# Scripts

Every file in this folder, by analysis step. The `submit_*.py` scripts send
Slurm jobs; each job runs the matching `fit_single.py` or `run_*.py` script,
which you can also run on your own computer.

- `README.md`: this list.
- `config.toml`: the analysis settings (file paths, exclusions, MCMC and
  cluster settings).

## 1. Data preparation ([guide](../docs/1-data-preparation.md))

- `prepare_data.py`: applies the exclusions and writes one data file for each
  condition.

## 2. Model fitting ([guide](../docs/2-model-fitting.md))

- `submit_jobs.py`: sends one job for each model and condition.
- `fit_single.py`: fits one model to one condition.
- `compile_results.py`: gathers the fits into tables and runs the ANOVAs.

## 3. Model selection and recovery ([guide](../docs/3-model-selection-and-recovery.md))

- `submit_integrated_model_selection.py`: sends one job for each model.
- `run_integrated_model_selection.py`: fits one integrated model.
- `compile_integrated_model_selection.py`: compares the models.
- `run_integrated_posterior_predictive_check.py`: posterior predictive checks
  of the integrated models.
- `submit_integrated_model_recovery.py`: sends the model recovery jobs.
- `run_integrated_model_recovery_fit.py`: one model recovery fit.
- `compile_integrated_model_recovery.py`: gathers the model recovery results.
- `submit_parameter_recovery.py`: sends one parameter recovery job for each
  model and condition.
- `run_parameter_recovery.py`: one parameter recovery run.

## 4. Regression analyses ([guide](../docs/4-regression-analyses.md))

- `regression_analysis.py`: relates one learning-rate parameter to the
  questionnaire measures.

## 5. Figures ([guide](../docs/5-publication-figures.md))

- `make_publication_figures.py`: task-performance statistics and Figures 1
  to 3.
- `make_recovery_figures.py`: Figures S1 to S3.
