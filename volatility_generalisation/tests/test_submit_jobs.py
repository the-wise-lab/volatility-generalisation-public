from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_submit_jobs_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "submit_jobs.py"
    spec = importlib.util.spec_from_file_location("submit_jobs", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_job_pairs_flags_existing_outputs(tmp_path: Path) -> None:
    module = load_submit_jobs_module()
    completed = tmp_path / "VolatileV1_rescorla_wagner_single_value"
    completed.mkdir(parents=True)
    (completed / "metadata.json").write_text("{}\n", encoding="utf-8")

    try:
        module.build_job_pairs(
            ["rescorla_wagner_single_value", "dual_lr_single_value"],
            ["VolatileV1"],
            tmp_path,
        )
    except RuntimeError as error:
        assert "rescorla_wagner_single_value x VolatileV1" in str(error)
    else:
        raise AssertionError("Expected build_job_pairs to flag existing outputs")


def test_build_job_pairs_skips_existing_outputs(tmp_path: Path) -> None:
    module = load_submit_jobs_module()
    completed = tmp_path / "VolatileV1_rescorla_wagner_single_value"
    completed.mkdir(parents=True)
    (completed / "metadata.json").write_text("{}\n", encoding="utf-8")

    pairs = module.build_job_pairs(
        ["rescorla_wagner_single_value", "dual_lr_single_value"],
        ["VolatileV1"],
        tmp_path,
        skip_existing=True,
    )

    assert pairs == [("dual_lr_single_value", "VolatileV1")]


def test_build_job_pairs_overwrite_keeps_existing_outputs(tmp_path: Path) -> None:
    module = load_submit_jobs_module()
    completed = tmp_path / "VolatileV1_rescorla_wagner_single_value"
    completed.mkdir(parents=True)
    (completed / "metadata.json").write_text("{}\n", encoding="utf-8")

    pairs = module.build_job_pairs(
        ["rescorla_wagner_single_value", "dual_lr_single_value"],
        ["VolatileV1"],
        tmp_path,
        overwrite=True,
    )

    assert pairs == [
        ("rescorla_wagner_single_value", "VolatileV1"),
        ("dual_lr_single_value", "VolatileV1"),
    ]


def test_submit_jobs_defaults_only_include_retained_models() -> None:
    module = load_submit_jobs_module()

    assert module.FALLBACK_MODELS == [
        "rescorla_wagner_single_value",
        "dual_lr_single_value",
        "pearce_hall_single_value",
        "leaky_beta_single_value",
        "leaky_beta_asymmetric_single_value",
    ]


def test_submit_jobs_parser_rejects_removed_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_submit_jobs_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["submit_jobs.py", "--models", "rescorla_wagner", "--dry-run"],
    )

    with pytest.raises(SystemExit):
        module.setup_args()


def test_submit_jobs_validation_rejects_removed_models() -> None:
    module = load_submit_jobs_module()

    with pytest.raises(ValueError, match="Unsupported model"):
        module.validate_models(["rescorla_wagner", "dual_lr_single_value"])


def test_generate_slurm_script_passes_root_model_config(tmp_path: Path) -> None:
    """Check the model settings path, each task's cache, and the shared uv cache.

    Args:
        tmp_path: Temporary project and cache folder from pytest.
    """

    module = load_submit_jobs_module()

    args = SimpleNamespace(
        script_dir="scripts",
        time="01:00:00",
        mem="8G",
        cpus=4,
        partition="cpu",
        warmup=100,
        samples=200,
        chains=4,
    )

    script = module.generate_slurm_script(
        job_list_file=tmp_path / ".job_list.txt",
        n_jobs=1,
        args=args,
        project_dir=tmp_path,
        modules=[],
        pre_run_commands=[],
        python_command="uv run --frozen python",
    )

    assert 'MODEL_CONFIG="$PROJECT_DIR/config/model_bounds.toml"' in script
    assert '--model-config "$MODEL_CONFIG"' in script

    cache_exports = "\n".join(
        line
        for line in script.splitlines()
        if line.startswith(("export UV_CACHE_DIR=", "export XDG_CACHE_HOME="))
    )
    assert (
        script.index("export UV_CACHE_DIR=")
        < script.index("export XDG_CACHE_HOME=")
        < script.index("# Run Fitting Script")
    )
    for job_id, temporary_base, uv_cache in [
        ("101", str(tmp_path / "with spaces"), ""),
        ("102", "", "/shared/custom uv"),
    ]:
        result = subprocess.run(
            ["bash", "-c", cache_exports + "\nprintenv XDG_CACHE_HOME UV_CACHE_DIR"],
            env={
                "PATH": os.defpath,
                "SLURM_JOB_ID": job_id,
                "SLURM_ARRAY_JOB_ID": "100",
                "TMPDIR": temporary_base,
                "XDG_CACHE_HOME": "/shared/cache",
                "UV_CACHE_DIR": uv_cache,
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.stdout.splitlines() == [
            f"{temporary_base or '/tmp'}/cache_{job_id}",
            uv_cache or "/shared/cache/uv",
        ]


def _make_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make a small project folder with a settings file, and work inside it.

    Args:
        tmp_path: Temporary folder from pytest.
        monkeypatch: pytest tool used to change the working folder.

    Returns:
        Path to the project folder.
    """
    project_dir = tmp_path / "project"
    (project_dir / "scripts").mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        "[project]\nname = 'temp'\n", encoding="utf-8"
    )
    (project_dir / "scripts" / "config.toml").write_text(
        "[mcmc]\nseed = 1\n", encoding="utf-8"
    )
    monkeypatch.chdir(project_dir)
    return project_dir


def _set_command_line(monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    """Set the command line for submit_jobs.py with one model and one condition.

    Args:
        monkeypatch: pytest tool used to change ``sys.argv``.
        *extra: More options to add, for example ``--dry-run``.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_jobs.py",
            "--models",
            "rescorla_wagner_single_value",
            "--conditions",
            "VolatileV1",
            *extra,
        ],
    )


FIT_AND_SHARED_RESOURCES = (
    '[hpc]\nmemory = "64G"\ntime = "04:00:00"\n'
    'fit_memory = "12G"\nfit_time = "01:00:00"\n'
)


@pytest.mark.parametrize(
    "hpc_settings, resource_options, expected_mem, expected_time",
    [
        pytest.param(
            FIT_AND_SHARED_RESOURCES, [], "12G", "01:00:00", id="fit-defaults"
        ),
        pytest.param(
            FIT_AND_SHARED_RESOURCES,
            ["--mem", "32G", "--time", "02:00:00"],
            "32G",
            "02:00:00",
            id="command-line-overrides",
        ),
        pytest.param(
            '[hpc]\nmemory = "48G"\ntime = "03:00:00"\n',
            [],
            "48G",
            "03:00:00",
            id="shared-settings-only",
        ),
        pytest.param("", [], "16G", "04:00:00", id="built-in-fallbacks"),
    ],
)
def test_dry_run_writes_nothing_and_prints_the_job_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    hpc_settings: str,
    resource_options: list[str],
    expected_mem: str,
    expected_time: str,
) -> None:
    """Check that a dry run prints fit resources and cache settings without writing.

    The order is: command-line option, then fit-specific setting, then shared
    setting, then built-in fallback. The expected values are written out here
    rather than taken from submit_jobs.py.

    Args:
        tmp_path: Temporary folder from pytest.
        monkeypatch: pytest tool used to change the working folder and command
            line.
        capsys: pytest tool that captures printed text.
        hpc_settings: ``[hpc]`` table added to the settings file.
        resource_options: ``--mem`` and ``--time`` options, if any.
        expected_mem: Memory the batch script should request.
        expected_time: Time limit the batch script should request.
    """
    module = load_submit_jobs_module()
    project_dir = _make_project(tmp_path, monkeypatch)
    config_path = project_dir / "scripts" / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + hpc_settings, encoding="utf-8"
    )
    _set_command_line(monkeypatch, "--dry-run", *resource_options)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called during a dry run")

    monkeypatch.setattr(module.subprocess, "run", fail_if_called)

    module.main()

    printed = capsys.readouterr().out
    assert "rescorla_wagner_single_value VolatileV1" in printed
    assert "#SBATCH --array=1-1" in printed
    assert f"#SBATCH --mem={expected_mem}\n" in printed
    assert f"#SBATCH --time={expected_time}\n" in printed
    assert "#SBATCH --cpus-per-task=4\n" in printed
    assert "--xla_force_host_platform_device_count=4" in printed
    assert "--num-chains 4 " in printed
    assert 'export XDG_CACHE_HOME="${TMPDIR:-/tmp}/cache_${SLURM_JOB_ID}"' in printed
    assert not (project_dir / "logs").exists()
    assert not (project_dir / "results").exists()
    assert not (project_dir / "scripts" / ".job_list.txt").exists()
    assert not (project_dir / "scripts" / ".temp_submit.sh").exists()


@pytest.mark.parametrize(
    "sbatch_error",
    [
        subprocess.CalledProcessError(1, ["sbatch"]),
        FileNotFoundError("sbatch"),
    ],
)
def test_main_exits_with_an_error_when_sbatch_fails_or_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sbatch_error: Exception
) -> None:
    module = load_submit_jobs_module()
    _make_project(tmp_path, monkeypatch)
    _set_command_line(monkeypatch)

    def fake_run(*_args, **_kwargs):
        raise sbatch_error

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code not in (0, None)


@pytest.mark.parametrize(
    "config_text, error_type",
    [
        (None, FileNotFoundError),
        ("[mcmc]\nseed = 1\nseed = 2\n", ValueError),
    ],
)
def test_main_stops_before_writing_when_config_is_missing_or_broken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_text: str | None,
    error_type: type[Exception],
) -> None:
    module = load_submit_jobs_module()
    project_dir = _make_project(tmp_path, monkeypatch)
    config_path = project_dir / "scripts" / "config.toml"
    if config_text is None:
        config_path.unlink()
    else:
        config_path.write_text(config_text, encoding="utf-8")
    _set_command_line(monkeypatch, "--config", str(config_path))

    with pytest.raises(error_type, match="config.toml"):
        module.main()

    assert not (project_dir / "logs").exists()
    assert not (project_dir / "scripts" / ".job_list.txt").exists()
    assert not (project_dir / "scripts" / ".temp_submit.sh").exists()
