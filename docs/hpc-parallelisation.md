# Running on a computing cluster

The model fits are slow, so the `submit_*.py` scripts send them as Slurm jobs
that run at the same time. The work is split in two ways:

- **Across jobs:** each job fits one model to one condition (or, for the
  integrated analyses, one model or one recovery fit).
- **Within a job:** the MCMC chains run in parallel, one on each CPU.

All cluster settings are in `scripts/config.toml`:

```toml
[mcmc]
num_chains = 4

[hpc]
cpus_per_task = 4    # keep equal to num_chains
fit_time = "02:00:00"
fit_memory = "12G"
integrated_selection_time = "04:00:00"
integrated_selection_memory = "32G"
integrated_recovery_time = "01:00:00"
integrated_recovery_memory = "16G"
parameter_recovery_time = "01:00:00"
parameter_recovery_memory = "6G"
```

Each kind of job has its own time and memory limit:

| Jobs | Script | Settings |
| --- | --- | --- |
| Per-condition fits | `submit_jobs.py` | `fit_memory`, `fit_time` |
| Integrated model selection | `submit_integrated_model_selection.py` | `integrated_selection_memory`, `integrated_selection_time` |
| Integrated model recovery | `submit_integrated_model_recovery.py` | `integrated_recovery_memory`, `integrated_recovery_time` |
| Parameter recovery | `submit_parameter_recovery.py` | `parameter_recovery_memory`, `parameter_recovery_time` |

These limits suit this study's data set, with some spare capacity. A larger
data set, or more MCMC samples, may need more. To change a limit for one
submission, use `--mem` and `--time`. In total, all the cluster jobs need
roughly 150 CPU-hours.

Use at least 4 chains, because R-hat is less reliable with fewer. If a job
does not use all its CPUs, check that `cpus_per_task` equals `num_chains`:
with fewer CPUs than chains, the chains run one after another. If a job runs
out of memory, increase its memory setting.

## Without a cluster

Each fit can run on one computer with the matching `run_*.py` script, or
`fit_single.py` for the per-condition fits (see
[step 2](2-model-fitting.md#without-slurm)). The chains run in parallel when
the computer has at least as many CPU cores as chains. The fits then run one
after another, so the whole analysis takes much longer.
