# How Past Experiences Shape New Beliefs: Generalisation and Adaptation in Human Reinforcement Learning  

Phoenix Pavlik, Tin Nok Isaac Choy, Marko Dimitrovski & Toby Wise

----

## Analysis steps

1. **Prepare the data**: apply the exclusions and split the data by
   condition. See [`docs/1-data-preparation.md`](docs/1-data-preparation.md).
2. **Fit the models** to each condition. See
   [`docs/2-model-fitting.md`](docs/2-model-fitting.md).
3. **Compare the integrated models and check recovery.** See
   [`docs/3-model-selection-and-recovery.md`](docs/3-model-selection-and-recovery.md).
4. **Relate learning rates to questionnaire measures.** See
   [`docs/4-regression-analyses.md`](docs/4-regression-analyses.md).
5. **Make the paper figures.** See
   [`docs/5-publication-figures.md`](docs/5-publication-figures.md).

Model fitting can be slow, and so scripts are provided to enable parallelisation on a Slurm-managed high-performance computing cluster. See
[`docs/hpc-parallelisation.md`](docs/hpc-parallelisation.md).

## Getting started

The Python environment is managed using [uv](https://docs.astral.sh/uv/getting-started/installation/). After installing `uv`, run the following commands to set up the environment and run the analysis:

```bash
uv sync --locked
uv run python scripts/prepare_data.py
uv run python scripts/submit_jobs.py               # needs Slurm
uv run python scripts/compile_results.py --overwrite
uv run python scripts/regression_analysis.py --parameter learning_rate
```

Wait for the Slurm jobs to end before you compile the results. Without Slurm, fit each
model and condition locally with `scripts/fit_single.py` (see
[`docs/2-model-fitting.md`](docs/2-model-fitting.md)).

## Data and results in this repository

- `data/behavioural_data.csv.zip`: the task data.
- `data/questionnaire_data.csv.zip`: the questionnaire answers.
- `data/prepared/qc_exclusion_audit.csv` and `qc_exclusion_counts.csv`: who
  was excluded, and why.
- `results/compiled/subject_parameters.csv` and
  `results/compiled_dual_lr/subject_parameters.csv`: the fitted participant
  parameters, so that the regressions can run without refitting.
- `results/publication_figures/tables/` and `results/recovery_figures/tables/`:
  every number plotted in the paper figures.
- `results/integrated_model_recovery/summary/tables/`: the recovery results
  that Figures S1 to S3 are drawn from.

Posterior samples are not included here due to their large size, but can be regenerated as outlined above. See
[`data/README.md`](data/README.md) for the contents of each data file.

## Paper figures and tables

| Paper item | Command | Output folder |
| --- | --- | --- |
| Task-performance statistics, Figures 1 to 3 | `uv run python scripts/make_publication_figures.py` | `results/publication_figures/` |
| Figures S1 to S3 (recovery) | `uv run python scripts/make_recovery_figures.py --replicates 10` | `results/recovery_figures/` |
| Regression tables | `uv run python scripts/regression_analysis.py --parameter learning_rate` (also `learning_rate_neg` and `learning_rate_pos`) | `results/regression/<parameter>/` |
| Factor-analysis checks | the same regression commands | `results/factor_analysis_checks/` |

Figures S1 to S3 can be drawn from the included tables. Figures 2 and 3 need
the integrated model fits, which you must make first (see
[`docs/3-model-selection-and-recovery.md`](docs/3-model-selection-and-recovery.md)).
[`docs/5-publication-figures.md`](docs/5-publication-figures.md) lists the
files for each figure.
