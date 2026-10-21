from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_submit_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "submit_integrated_model_recovery.py"
    spec = importlib.util.spec_from_file_location("submit_integrated_model_recovery", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_pending_jobs_skips_completed_runs(tmp_path: Path) -> None:
    module = load_submit_module()
    completed = tmp_path / "replicate_000" / "rescorla_wagner_single_value" / "dual_lr_single_value"
    completed.mkdir(parents=True)
    (completed / "fit_summary.json").write_text("{}\n", encoding="utf-8")

    jobs = [
        {"replicate": 0, "simulated_model": "rescorla_wagner_single_value", "fitted_model": "dual_lr_single_value"},
        {"replicate": 0, "simulated_model": "dual_lr_single_value", "fitted_model": "dual_lr_single_value"},
    ]

    pending = module.build_pending_jobs(jobs, tmp_path, overwrite=False)

    assert pending == [jobs[1]]


def test_generate_slurm_script_includes_recovery_args(tmp_path: Path) -> None:
    """Check model-recovery arguments and the cache exports.

    Args:
        tmp_path: Temporary project folder from pytest.
    """

    module = load_submit_module()
    args = SimpleNamespace(
        time="01:00:00",
        mem="8G",
        cpus=2,
        partition="cpu",
        conditions=["StableV1", "VolatileV2"],
        model_order=["rescorla_wagner_single_value", "dual_lr_single_value"],
        data_dir="data/prepared",
        output_dir="results/integrated_model_recovery",
        n_subjects=10,
        base_seed=42,
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        target_accept_prob=0.95,
        max_tree_depth=10,
        thinning=1,
        loo_drop_initial_trials=1,
        array_concurrency=5,
    )
    job_list_file = tmp_path / "scripts" / ".integrated_recovery_job_list.txt"
    job_list_file.parent.mkdir(parents=True)

    script = module.generate_slurm_script(
        job_list_file=job_list_file,
        n_jobs=2,
        args=args,
        project_dir=tmp_path,
        python_command="python",
    )

    assert "#SBATCH --job-name=integrated_model_recovery" in script
    assert "#SBATCH --array=1-2%5" in script
    assert "run_integrated_model_recovery_fit.py" in script
    assert '--models rescorla_wagner_single_value dual_lr_single_value' in script
    assert '--conditions StableV1 VolatileV2' in script
    assert '--loo-drop-initial-trials 1' in script

    assert (
        'export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"\n'
        'export XDG_CACHE_HOME="${TMPDIR:-/tmp}/cache_${SLURM_JOB_ID}"'
    ) in script


def test_generate_slurm_script_propagates_custom_loo_drop_initial_trials(tmp_path: Path) -> None:
    module = load_submit_module()
    args = SimpleNamespace(
        time="01:00:00",
        mem="8G",
        cpus=2,
        partition="cpu",
        conditions=["StableV1", "VolatileV2"],
        model_order=["rescorla_wagner_single_value", "dual_lr_single_value"],
        data_dir="data/prepared",
        output_dir="results/integrated_model_recovery",
        n_subjects=10,
        base_seed=42,
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        target_accept_prob=0.95,
        max_tree_depth=10,
        thinning=1,
        loo_drop_initial_trials=3,
        array_concurrency=None,
    )
    job_list_file = tmp_path / "scripts" / ".integrated_recovery_job_list.txt"
    job_list_file.parent.mkdir(parents=True)

    script = module.generate_slurm_script(
        job_list_file=job_list_file,
        n_jobs=2,
        args=args,
        project_dir=tmp_path,
        python_command="python",
    )

    assert '--loo-drop-initial-trials 3' in script


@pytest.mark.parametrize("flag", ["--models", "--simulated-models", "--fitted-models"])
def test_parse_args_rejects_removed_models(flag: str, monkeypatch) -> None:
    module = load_submit_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["submit_integrated_model_recovery.py", flag, "rescorla_wagner"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_main_uses_second_block_defaults_when_conditions_not_provided(tmp_path: Path, monkeypatch) -> None:
    module = load_submit_module()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname = 'temp'\n", encoding="utf-8")
    config_path = Path(__file__).resolve().parents[2] / "scripts" / "config.toml"

    args = SimpleNamespace(
        config=str(config_path),
        models=None,
        simulated_models=None,
        fitted_models=None,
        conditions=None,
        data_dir="data/prepared",
        output_dir="results/integrated_model_recovery",
        replicates=1,
        n_subjects=10,
        base_seed=42,
        time="01:00:00",
        mem="8G",
        cpus=2,
        partition="cpu",
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        target_accept_prob=0.95,
        max_tree_depth=10,
        thinning=1,
        loo_drop_initial_trials=1,
        skip_existing=False,
        overwrite=False,
        array_concurrency=None,
        dry_run=True,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.chdir(project_dir)

    module.main()

    assert args.model_order == [
        "rescorla_wagner_single_value",
        "dual_lr_single_value",
        "pearce_hall_single_value",
        "leaky_beta_single_value",
        "leaky_beta_asymmetric_single_value",
    ]
    assert args.simulated_models == args.model_order
    assert args.fitted_models == args.model_order
    assert args.conditions == ["StableV1", "VolatileV2", "VolatileV3_2", "StableV4_2"]


def test_dry_run_does_not_create_persistent_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = load_submit_module()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname = 'temp'\n", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text("[mcmc]\nseed = 1\n", encoding="utf-8")

    args = SimpleNamespace(
        config=str(config_path),
        models=["rescorla_wagner_single_value", "dual_lr_single_value"],
        simulated_models=["rescorla_wagner_single_value"],
        fitted_models=["dual_lr_single_value"],
        conditions=["StableV1"],
        data_dir="data/prepared",
        output_dir="results/integrated_model_recovery",
        replicates=1,
        n_subjects=10,
        base_seed=42,
        time="01:00:00",
        mem="8G",
        cpus=2,
        partition="cpu",
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        target_accept_prob=0.95,
        max_tree_depth=10,
        thinning=1,
        loo_drop_initial_trials=1,
        skip_existing=False,
        overwrite=False,
        array_concurrency=None,
        dry_run=True,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called during dry-run")

    monkeypatch.setattr(module.subprocess, "run", fail_if_called)
    monkeypatch.chdir(project_dir)

    module.main()

    printed = capsys.readouterr().out
    expected_row = "0\trescorla_wagner_single_value\tdual_lr_single_value"
    assert f"Job list (not written):\n{expected_row}\n" in printed
    assert not (project_dir / "results").exists()
    assert not (project_dir / "logs").exists()
    assert not (project_dir / "scripts" / ".integrated_model_recovery_job_list.txt").exists()


def _dry_run_args(config_path: Path, **overrides) -> SimpleNamespace:
    """Build the command-line options for a dry run, with some values replaced.

    Args:
        config_path: Path to the settings file.
        **overrides: Options to replace, for example ``base_seed=0``.

    Returns:
        The command-line options, as they would come from ``parse_args``.
    """
    values = dict(
        config=str(config_path),
        models=["rescorla_wagner_single_value", "dual_lr_single_value"],
        simulated_models=None,
        fitted_models=None,
        conditions=["StableV1"],
        data_dir="data/prepared",
        output_dir="results/integrated_model_recovery",
        replicates=1,
        n_subjects=10,
        base_seed=None,
        time=None,
        mem=None,
        cpus=None,
        partition=None,
        num_warmup=None,
        num_samples=None,
        num_chains=None,
        target_accept_prob=None,
        max_tree_depth=None,
        thinning=None,
        loo_drop_initial_trials=1,
        skip_existing=False,
        overwrite=False,
        array_concurrency=None,
        dry_run=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_main_with(module, monkeypatch, tmp_path: Path, config_text: str, **overrides):
    """Run ``main`` in a small project with a settings file.

    Args:
        module: The loaded submit script.
        monkeypatch: pytest tool used to set the command-line options and folder.
        tmp_path: Temporary folder from pytest.
        config_text: Text of the settings file.
        **overrides: Command-line options to replace.

    Returns:
        The command-line options after ``main`` has filled in the defaults.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        "[project]\nname = 'temp'\n", encoding="utf-8"
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_text, encoding="utf-8")
    args = _dry_run_args(config_path, **overrides)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.chdir(project_dir)
    module.main()
    return args


def test_explicit_zero_values_are_kept(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_submit_module()

    args = _run_main_with(
        module,
        monkeypatch,
        tmp_path,
        "[mcmc]\nseed = 7\nnum_warmup = 50\n",
        base_seed=0,
        num_warmup=0,
    )

    printed = capsys.readouterr().out
    assert args.base_seed == 0
    assert args.num_warmup == 0
    assert "--base-seed 0" in printed
    assert "--num-warmup 0" in printed


# Resource cases. The settings for this job family, the other two families and
# the shared `time` / `memory` all differ, so reading the wrong key, or ignoring
# `--mem` / `--time`, prints the wrong #SBATCH line.
FAMILY_RESOURCE_SETTINGS = (
    '[hpc]\nmemory = "64G"\ntime = "04:00:00"\n'
    'integrated_recovery_memory = "24G"\nintegrated_recovery_time = "02:30:00"\n'
    'integrated_selection_memory = "40G"\nintegrated_selection_time = "05:00:00"\n'
    'parameter_recovery_memory = "40G"\nparameter_recovery_time = "05:00:00"\n'
)
RESOURCE_CASES = [
    pytest.param(FAMILY_RESOURCE_SETTINGS, {}, "24G", "02:30:00", id="family-keys"),
    pytest.param(
        FAMILY_RESOURCE_SETTINGS,
        {"mem": "32G", "time": "02:00:00"},
        "32G",
        "02:00:00",
        id="command-line",
    ),
    pytest.param(
        '[hpc]\nmemory = "48G"\ntime = "03:00:00"\n',
        {},
        "48G",
        "03:00:00",
        id="shared-keys-only",
    ),
    pytest.param("", {}, "16G", "04:00:00", id="built-in-fallbacks"),
]


@pytest.mark.parametrize(
    "hpc_text, overrides, expected_mem, expected_time", RESOURCE_CASES
)
def test_slurm_time_and_memory_come_from_the_recovery_settings(
    tmp_path: Path,
    monkeypatch,
    capsys,
    hpc_text: str,
    overrides: dict,
    expected_mem: str,
    expected_time: str,
) -> None:
    module = load_submit_module()

    _run_main_with(module, monkeypatch, tmp_path, hpc_text, **overrides)

    printed = capsys.readouterr().out
    assert f"#SBATCH --mem={expected_mem}" in printed.splitlines()
    assert f"#SBATCH --time={expected_time}" in printed.splitlines()


@pytest.mark.parametrize(
    "option, override, config_text",
    [
        ("num_chains", {"num_chains": 0}, "[mcmc]\nnum_chains = 4\n"),
        ("num_samples", {}, "[mcmc]\nnum_samples = 0\n"),
        ("cpus", {"cpus": 0}, "[hpc]\ncpus_per_task = 4\n"),
        ("thinning", {"thinning": 0}, "[mcmc]\nthinning = 1\n"),
        ("max_tree_depth", {}, "[mcmc]\nmax_tree_depth = 0\n"),
        ("array_concurrency", {"array_concurrency": 0}, "[mcmc]\nseed = 1\n"),
    ],
)
def test_values_below_one_stop_with_the_option_named(
    tmp_path: Path, monkeypatch, option: str, override: dict, config_text: str
) -> None:
    module = load_submit_module()

    with pytest.raises(ValueError, match=option.replace("_", "-")):
        _run_main_with(module, monkeypatch, tmp_path, config_text, **override)


def test_empty_model_list_stops(tmp_path: Path, monkeypatch) -> None:
    module = load_submit_module()

    with pytest.raises(ValueError, match="model list is empty"):
        _run_main_with(
            module, monkeypatch, tmp_path, "[fitting]\nmodels = []\n", models=None
        )
