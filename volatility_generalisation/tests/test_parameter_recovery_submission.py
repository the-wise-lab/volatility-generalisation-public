from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_submit_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "submit_parameter_recovery.py"
    spec = importlib.util.spec_from_file_location("submit_parameter_recovery", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_parameter_recovery.py"
    spec = importlib.util.spec_from_file_location("run_parameter_recovery", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_job_pairs_aborts_when_outputs_exist_without_flag(tmp_path):
    module = load_submit_module()
    run_dir = tmp_path / "VolatileV1" / "rescorla_wagner_single_value"
    run_dir.mkdir(parents=True)
    (run_dir / "parameter_recovery.csv").write_text("done\n", encoding="utf-8")

    try:
        module.build_job_pairs(["rescorla_wagner_single_value"], ["VolatileV1"], tmp_path)
    except RuntimeError as exc:
        assert "Existing parameter recovery outputs found" in str(exc)
    else:
        raise AssertionError("Expected build_job_pairs to abort on existing outputs")


def test_build_job_pairs_can_skip_existing_outputs(tmp_path):
    module = load_submit_module()
    run_dir = tmp_path / "VolatileV1" / "rescorla_wagner_single_value"
    run_dir.mkdir(parents=True)
    # A finished run has its two tables, diagnostics and run details, and no figures.
    for name in (
        "parameter_recovery.csv",
        "parameter_recovery_summary.csv",
        "diagnostics.json",
        "run_metadata.json",
    ):
        (run_dir / name).write_text("done\n", encoding="utf-8")

    pairs = module.build_job_pairs(
        ["rescorla_wagner_single_value", "dual_lr_single_value"],
        ["VolatileV1"],
        tmp_path,
        skip_existing=True,
    )

    assert pairs == [("dual_lr_single_value", "VolatileV1")]


def test_submit_parameter_recovery_defaults_only_include_retained_models() -> None:
    module = load_submit_module()

    assert module.FALLBACK_MODELS == [
        "rescorla_wagner_single_value",
        "dual_lr_single_value",
        "pearce_hall_single_value",
        "leaky_beta_single_value",
        "leaky_beta_asymmetric_single_value",
    ]


def test_submit_parameter_recovery_parser_rejects_removed_models(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_submit_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["submit_parameter_recovery.py", "--models", "rescorla_wagner", "--dry-run"],
    )

    with pytest.raises(SystemExit):
        module.setup_args()


def test_run_parameter_recovery_defaults_only_include_retained_models() -> None:
    module = load_run_module()

    assert module.FALLBACK_MODELS == [
        "rescorla_wagner_single_value",
        "dual_lr_single_value",
        "pearce_hall_single_value",
        "leaky_beta_single_value",
        "leaky_beta_asymmetric_single_value",
    ]


def test_run_parameter_recovery_parser_rejects_removed_models(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_run_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_parameter_recovery.py", "--models", "rescorla_wagner"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_validate_prepared_condition_files_aborts_for_missing_inputs(tmp_path):
    module = load_submit_module()
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    (data_dir / "VolatileV1.npz").write_text("done\n", encoding="utf-8")

    try:
        module.validate_prepared_condition_files(data_dir, ["VolatileV1", "StableV1"])
    except RuntimeError as exc:
        message = str(exc)
        assert "Missing prepared condition files" in message
        assert "StableV1" in message
        assert "prepare_data.py" in message
    else:
        raise AssertionError("Expected validate_prepared_condition_files to abort")


def test_generate_slurm_script_includes_partition_and_worker_args(tmp_path):
    """Check parameter-recovery arguments and the cache exports.

    Args:
        tmp_path: Temporary project folder from pytest.
    """

    module = load_submit_module()
    args = SimpleNamespace(
        time="01:00:00",
        mem="8G",
        cpus=2,
        partition="cpu",
        data_dir="data/prepared",
        output_dir="results/parameter_recovery",
        config=str(tmp_path / "custom_settings.toml"),
        n_subjects=50,
        num_warmup=2000,
        num_samples=4000,
        num_chains=4,
        seed=42,
        target_accept_prob=0.95,
        max_tree_depth=10,
        thinning=1,
    )
    project_dir = tmp_path
    job_list_file = tmp_path / "scripts" / ".parameter_recovery_job_list.txt"
    job_list_file.parent.mkdir(parents=True)

    script = module.generate_slurm_script(
        job_list_file=job_list_file,
        n_jobs=1,
        args=args,
        project_dir=project_dir,
        python_command="python",
    )

    assert "#SBATCH --job-name=parameter_recovery" in script
    assert "#SBATCH --partition=cpu" in script
    assert "run_parameter_recovery.py" in script
    assert "--n-subjects 50" in script
    assert "--target-accept-prob 0.95" in script
    assert f'--config "{(tmp_path / "custom_settings.toml").resolve()}"' in script

    assert (
        'export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"\n'
        'export XDG_CACHE_HOME="${TMPDIR:-/tmp}/cache_${SLURM_JOB_ID}"'
    ) in script


def test_resolve_settings_switches_to_interruptible_partition_when_requested():
    module = load_submit_module()
    args = SimpleNamespace(
        models=None,
        conditions=None,
        data_dir="data/prepared",
        output_dir="results/parameter_recovery",
        n_subjects=None,
        time=None,
        mem=None,
        cpus=None,
        partition=None,
        interruptible_cpu=True,
        num_warmup=None,
        num_samples=None,
        num_chains=None,
        seed=None,
        target_accept_prob=None,
        max_tree_depth=None,
        thinning=None,
    )

    resolved = module.resolve_settings(args, {"hpc": {"partition": "cpu"}})

    assert resolved.partition == "interruptible_cpu"


# Resource cases. The settings for this job family, the other two families and
# the shared `time` / `memory` all differ, so reading the wrong key, or ignoring
# `--mem` / `--time`, prints the wrong #SBATCH line.
FAMILY_RESOURCE_SETTINGS = (
    '[hpc]\nmemory = "64G"\ntime = "04:00:00"\n'
    'parameter_recovery_memory = "24G"\nparameter_recovery_time = "02:30:00"\n'
    'integrated_selection_memory = "40G"\nintegrated_selection_time = "05:00:00"\n'
    'integrated_recovery_memory = "40G"\nintegrated_recovery_time = "05:00:00"\n'
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
def test_dry_run_writes_nothing(
    tmp_path, monkeypatch, capsys, hpc_text, overrides, expected_mem, expected_time
):
    module = load_submit_module()
    project_dir = tmp_path / "project"
    (project_dir / "scripts").mkdir(parents=True)
    (project_dir / "data" / "prepared").mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        "[project]\nname = 'temp'\n", encoding="utf-8"
    )
    (project_dir / "data" / "prepared" / "VolatileV1.npz").write_text(
        "data\n", encoding="utf-8"
    )
    # Only a custom settings file exists; there is no scripts/config.toml. It
    # sets a plain Python command, because the built-in one needs uv, and adds
    # the resource case's lines to the same [hpc] table.
    config_path = tmp_path / "custom_settings.toml"
    config_path.write_text(
        '[hpc]\npython_command = "python"\n' + hpc_text.removeprefix("[hpc]\n"),
        encoding="utf-8",
    )
    resource_options = []
    for name, value in overrides.items():
        resource_options += [f"--{name}", value]
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_parameter_recovery.py",
            "--config",
            str(config_path),
            "--models",
            "rescorla_wagner_single_value",
            "--conditions",
            "VolatileV1",
            "--dry-run",
            *resource_options,
        ],
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called during a dry run")

    monkeypatch.setattr(module.subprocess, "run", fail_if_called)

    module.main()

    printed = capsys.readouterr().out
    assert "rescorla_wagner_single_value VolatileV1" in printed
    assert "--array=1-1" in printed
    # The SLURM script is printed inside a bordered panel, so each line is
    # padded with spaces up to the border.
    assert re.search(rf"#SBATCH --mem={expected_mem}\s", printed)
    assert re.search(rf"#SBATCH --time={expected_time}\s", printed)
    assert not (project_dir / "logs").exists()
    assert not (project_dir / "results").exists()
    assert not (project_dir / "scripts" / ".parameter_recovery_job_list.txt").exists()
    assert not (project_dir / "scripts" / ".temp_parameter_recovery_submit.sh").exists()
