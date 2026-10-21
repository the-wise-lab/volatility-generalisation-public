"""Checks that keep the Markdown documentation accurate.

Checks, over every Markdown file in the repository (excluding dotfile
directories and the gitignored `plans/` directory, which quotes stale facts
on purpose):

- Every ``scripts/<name>.py`` mentioned exists, and every ``--flag``
  used with a script inside a fenced code block appears in that script's
  ``--help`` output.
- Every relative Markdown link resolves to an existing file.
- ``scripts/README.md`` names every file in ``scripts/`` (ignoring
  ``__pycache__`` and dotfiles left behind by the submit scripts), and
  nothing else.
- No ``name.py:NN`` or ``name.toml:NN`` line references appear, because line
  numbers drift. Code is named instead: every `` `function_name` in
  `path/to/file.py` `` reference must point to an existing file that defines
  that function, class or constant at top level. When several names share one
  file (`` `a` and `b` in `file.py` ``), every name is checked.
- None of the known stale facts have crept back in.

No allow-lists: every script mention, flag, and link found is checked. This
test does not depend on git: it walks the filesystem directly, so it also
runs correctly against a plain extracted copy of the repository.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from functools import cache
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.documentation

STALE_PATTERNS = [
    r"96 (parallel|SLURM|array) jobs",
    r"behavioural_data_new",
    r"12 models",
    r"controls\.csv",
    r"notebooks/",
]

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)
_SCRIPT_RE = re.compile(r"scripts/([A-Za-z0-9_.-]+\.py)")
_FLAG_RE = re.compile(r"(?<!\S)(--[A-Za-z][A-Za-z0-9-]*)")
_BARE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(py|toml|sh|md)$")

_LINE_REFERENCE_RE = re.compile(r"\b[\w./-]+\.(?:py|toml):\d+")
_NAMED_REFERENCE_RE = re.compile(
    r"((?:`\w+`(?:,\s+|\s+and\s+|\s+or\s+|,\s+and\s+))*`\w+`)\s+in\s+`([\w./-]+\.py)`"
)
_BACKTICKED_NAME_RE = re.compile(r"`(\w+)`")

_SKIP_TOP_LEVEL_DIRS = {"plans"}


def _collect_markdown_files(root: Path = REPO_ROOT) -> list[Path]:
    """Find every visible Markdown file under a folder, without using git.

    Walks the folder from the top down and removes unwanted subfolders from
    the walk before entering them, so nothing inside ``.venv``, ``.git`` or
    similar folders is ever listed. Skips every folder whose name starts with
    ``.`` and the top-level ``plans/`` folder, which is gitignored and
    deliberately quotes stale facts. A ``plans/`` folder deeper in the tree is
    kept. Links to folders are not followed.

    Args:
        root: Folder to search. Defaults to the repository root.

    Returns:
        Sorted absolute paths of every ``*.md`` file to check.
    """
    found: list[Path] = []
    for folder, subfolders, file_names in os.walk(root, topdown=True, followlinks=False):
        at_top_level = Path(folder) == root
        # Changing the list in place tells os.walk not to enter the folders
        # removed from it.
        subfolders[:] = [
            name
            for name in subfolders
            if not name.startswith(".")
            and not (at_top_level and name in _SKIP_TOP_LEVEL_DIRS)
        ]
        found.extend(
            Path(folder) / name
            for name in file_names
            if name.endswith(".md") and not name.startswith(".")
        )
    return sorted(found)


MARKDOWN_FILES = _collect_markdown_files()


def test_markdown_collection_found_readme(tmp_path: Path, monkeypatch) -> None:
    """The collector finds `README.md`, and never enters a folder it should skip.

    The real repository cannot show unnecessary searching, because the files
    it finds are right either way. So a small invented folder tree is searched
    here, and the folders the search enters are recorded.
    """
    assert REPO_ROOT / "README.md" in MARKDOWN_FILES

    for relative in [
        "README.md",
        "docs/guide.md",
        "docs/plans/nested.md",  # a `plans` folder below the top level is kept
        "docs/notes.txt",  # not Markdown
        "docs/.cache/hidden.md",  # hidden folder
        ".hidden/secret.md",  # hidden folder
        "plans/old.md",  # top-level `plans` folder
        "plans/deeper/older.md",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("text\n", encoding="utf-8")
    # A link to a folder: its files must be listed once, under their real folder.
    (tmp_path / "linked").symlink_to(tmp_path / "docs", target_is_directory=True)

    entered: list[Path] = []
    real_walk = os.walk

    def recording_walk(*args, **kwargs):
        """Run the real ``os.walk``, noting each folder it enters."""
        for folder, subfolders, file_names in real_walk(*args, **kwargs):
            entered.append(Path(folder))
            yield folder, subfolders, file_names

    monkeypatch.setattr(os, "walk", recording_walk)
    found = _collect_markdown_files(tmp_path)

    assert found == [
        tmp_path / "README.md",
        tmp_path / "docs" / "guide.md",
        tmp_path / "docs" / "plans" / "nested.md",
    ]
    # The skipped folders were never entered, not just filtered out afterwards.
    assert sorted(entered) == [tmp_path, tmp_path / "docs", tmp_path / "docs" / "plans"]


def _join_logical_lines(block: str) -> list[str]:
    """Join backslash-continued shell lines within one fenced code block.

    Args:
        block: Raw text inside one fenced code block.

    Returns:
        One string per logical command, with continuation backslashes
        resolved.
    """
    logical: list[str] = []
    buffer = ""
    for raw_line in block.split("\n"):
        line = raw_line.rstrip()
        if line.endswith("\\"):
            buffer += line[:-1] + " "
        else:
            buffer += line
            logical.append(buffer)
            buffer = ""
    if buffer:
        logical.append(buffer)
    return logical


def _script_flags(text: str) -> dict[str, set[str]]:
    """Collect every script/flag pair used inside a Markdown file's code blocks.

    Args:
        text: Full Markdown source of one file.

    Returns:
        Mapping of script filename (under ``scripts/``) to the set of
        ``--flag`` tokens used alongside it in a fenced code block.
    """
    script_flags: dict[str, set[str]] = {}
    for block in _CODE_BLOCK_RE.findall(text):
        for line in _join_logical_lines(block):
            script_match = _SCRIPT_RE.search(line)
            if script_match:
                script = script_match.group(1)
                script_flags.setdefault(script, set()).update(_FLAG_RE.findall(line))
    return script_flags


@cache
def _help_text(script: str) -> str:
    """Run one script's ``--help`` once per process and cache successful output.

    Args:
        script: Script filename under ``scripts/``.

    Returns:
        The concatenated stdout and stderr of ``<script> --help``.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"scripts/{script} --help exited {result.returncode}:\n{result.stderr}"
    return result.stdout + result.stderr


def _flag_present(flag: str, help_text: str) -> bool:
    """Check whether a flag appears in help text as a whole token.

    Args:
        flag: The ``--flag`` string to look for.
        help_text: Combined stdout/stderr of the script's ``--help``.

    Returns:
        True if ``flag`` appears bordered by anything other than a word
        character or hyphen (so ``--model`` does not match inside
        ``--model-config``).
    """
    pattern = re.compile(rf"(?<![\w-]){re.escape(flag)}(?![\w-])")
    return pattern.search(help_text) is not None


@pytest.mark.parametrize(
    "md_file", MARKDOWN_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_referenced_scripts_and_flags_are_real(md_file: Path) -> None:
    """Every mentioned script exists, and every code-block flag is real.

    Args:
        md_file: One collected Markdown file.
    """
    text = md_file.read_text()
    for script in set(_SCRIPT_RE.findall(text)):
        assert (
            REPO_ROOT / "scripts" / script
        ).is_file(), f"{md_file}: mentions scripts/{script}, which does not exist"
    for script, flags in _script_flags(text).items():
        if not flags:
            continue
        help_text = _help_text(script)
        for flag in sorted(flags):
            assert _flag_present(
                flag, help_text
            ), f"{md_file}: scripts/{script} has no {flag} flag (per --help)"


def test_cached_help_still_checks_each_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later document's invalid flag still fails after help is cached.

    Args:
        tmp_path: Directory for two small Markdown files and a script.
        monkeypatch: Restores the repository root and subprocess runner.
    """
    script = "example.py"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / script).touch()
    valid_doc = tmp_path / "valid.md"
    invalid_doc = tmp_path / "invalid.md"
    valid_doc.write_text("```sh\npython scripts/example.py --valid\n```\n")
    invalid_doc.write_text("```sh\npython scripts/example.py --invalid\n```\n")

    help_run = Mock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="--valid\n", stderr=""
        )
    )
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "run", help_run)

    _help_text.cache_clear()
    try:
        test_referenced_scripts_and_flags_are_real(valid_doc)
        with pytest.raises(AssertionError, match="has no --invalid flag"):
            test_referenced_scripts_and_flags_are_real(invalid_doc)
        help_run.assert_called_once_with(
            [sys.executable, str(scripts_dir / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
    finally:
        _help_text.cache_clear()


@pytest.mark.parametrize(
    "md_file", MARKDOWN_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_relative_links_resolve(md_file: Path) -> None:
    """Every relative Markdown link resolves to an existing file.

    Args:
        md_file: One collected Markdown file.
    """
    for target in _LINK_RE.findall(md_file.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        resolved = (md_file.parent / path_part).resolve()
        assert resolved.is_file(), f"{md_file}: link target {target!r} does not exist"


def test_scripts_readme_index_is_exact() -> None:
    """`scripts/README.md` lists every file in `scripts/`, and nothing else."""
    scripts_dir = REPO_ROOT / "scripts"
    actual_files = {
        entry.name
        for entry in scripts_dir.iterdir()
        if entry.is_file() and not entry.name.startswith(".")
    }

    readme_text = (scripts_dir / "README.md").read_text()
    mentioned_files = {
        span
        for span in re.findall(r"`([^`]+)`", readme_text)
        if _BARE_FILENAME_RE.match(span)
    }

    missing = actual_files - mentioned_files
    extra = mentioned_files - actual_files
    assert not missing, f"scripts/README.md is missing: {sorted(missing)}"
    assert not extra, f"scripts/README.md mentions non-existent files: {sorted(extra)}"


def _reference_problems(text: str, repo_root: Path) -> list[str]:
    """Find line references and unresolvable named code references in text.

    Args:
        text: Full Markdown source of one file.
        repo_root: Folder that the referenced file paths are relative to.

    Returns:
        One message per problem: each ``name.py:NN`` or ``name.toml:NN``
        reference, and each `` `name` in `path.py` `` reference whose file is
        missing or does not define ``name`` at top level as a function, class or
        constant.
    """
    problems = [
        f"line reference {match!r}; name the function or setting instead"
        for match in _LINE_REFERENCE_RE.findall(text)
    ]
    references = [
        (name, path)
        for names, path in _NAMED_REFERENCE_RE.findall(text)
        for name in _BACKTICKED_NAME_RE.findall(names)
    ]
    for name, path in references:
        source = repo_root / path
        if not source.is_file():
            problems.append(f"`{name}` in `{path}`: file does not exist")
        elif not re.search(
            rf"^(?:(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b"
            rf"|{re.escape(name)}\s*(?::[^=\n]*)?=)",
            source.read_text(),
            re.M,
        ):
            problems.append(f"`{name}` in `{path}`: not defined at top level")
    return problems


@pytest.mark.parametrize(
    "md_file", MARKDOWN_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_code_references_are_by_name_and_exist(md_file: Path) -> None:
    """No line references remain, and every named function exists.

    Args:
        md_file: One collected Markdown file.
    """
    problems = _reference_problems(md_file.read_text(), REPO_ROOT)
    assert not problems, f"{md_file}: {problems}"


def test_reference_check_catches_both_mistakes(tmp_path: Path) -> None:
    """A line reference and a missing function name are each reported.

    Args:
        tmp_path: Folder holding a small module with one function.
    """
    (tmp_path / "mod.py").write_text("def present():\n    pass\n\nLIMIT = 3\n")
    good = "`present` in `mod.py`"
    assert _reference_problems(good, tmp_path) == []
    assert _reference_problems("`LIMIT` in `mod.py`", tmp_path) == []
    shared = "`{}` and `present` in `mod.py`"
    assert _reference_problems(shared.format("LIMIT"), tmp_path) == []
    assert _reference_problems(shared.format("absent"), tmp_path)
    assert _reference_problems("see `mod.py:12`", tmp_path)
    assert _reference_problems("`absent` in `mod.py`", tmp_path)
    assert _reference_problems("`present` in `other.py`", tmp_path)


@pytest.mark.parametrize("pattern", STALE_PATTERNS)
def test_no_stale_facts(pattern: str) -> None:
    """No collected Markdown file contains a known stale fact.

    Args:
        pattern: One stale-fact regular expression to search for.
    """
    regex = re.compile(pattern)
    offenders = [
        md_file for md_file in MARKDOWN_FILES if regex.search(md_file.read_text())
    ]
    assert not offenders, f"stale pattern {pattern!r} found in: {offenders}"
