"""Checks that every function in the analysis code is fully described.

The checks read every Python file under `volatility_generalisation/` (except
its `tests/` folder) and in `scripts/`, without running them. They list every
problem found, not only the first. The rules:

1. Every module, class, function, method and nested function has a docstring.
2. A function with parameters (other than `self` and `cls`) has an `Args:`
   section with an entry for each one, including `*args` and `**kwargs`.
3. A function that returns a value has a `Returns:` section. A function that
   yields values has a `Yields:` section instead, and needs no `Returns:`.
4. A function that raises an error itself has a `Raises:` section that names
   at least one exception type.
5. No docstring uses NumPy-style underlined section headings.
6. No file mentions deleted files, plan contract numbers, or (in the regression
   files) the word "legacy".

There is no allow-list, and no dependence on git.
"""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest

pytestmark = pytest.mark.documentation

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "volatility_generalisation"
TESTS_DIR = PACKAGE_DIR / "tests"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Names of the five docstring rules. Each rule is tested on its own.
R_DOCSTRING = "missing docstring"
R_ARGS = "missing Args entry"
R_RETURNS = "missing Returns or Yields section"
R_RAISES = "missing Raises section"
R_NUMPY = "NumPy-style section"
DOCSTRING_RULES = [
    R_DOCSTRING,
    R_ARGS,
    R_RETURNS,
    R_RAISES,
    R_NUMPY,
]

# Text that must not appear in the code. This file is never searched.
STALE_PATTERNS = [
    re.compile("regression_analyses"),
    re.compile("ModellingResults"),
    re.compile("legacy_equivalence"),
    re.compile(r"\b(A?C)-\d{2}\b"),
]
LEGACY_PATTERN = re.compile("legacy", re.IGNORECASE)

_FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFINITION_TYPES = (*_FUNCTION_TYPES, ast.ClassDef)
_SCOPE_TYPES = (*_DEFINITION_TYPES, ast.Lambda)
_UNDERLINE = re.compile(r"^-{3,}$")
_EXCEPTION_ENTRY = re.compile(r"^[A-Za-z_][\w.]*:")


class Problem(NamedTuple):
    """One problem: the rule broken, where it is, and a plain-English message."""

    rule: str
    path: str
    line: int
    name: str
    message: str


def _collect_in_scope_files() -> list[Path]:
    """Find every Python file whose docstrings are checked, without using git.

    These are all files under `volatility_generalisation/` except its `tests/`
    folder, plus every Python file directly inside `scripts/`. Folders named
    `__pycache__` are skipped.

    Returns:
        Sorted absolute paths of the files to check.
    """
    found = [
        path
        for path in PACKAGE_DIR.rglob("*.py")
        if "__pycache__" not in path.parts and TESTS_DIR not in path.parents
    ]
    found.extend(SCRIPTS_DIR.glob("*.py"))
    return sorted(found)


IN_SCOPE_FILES = _collect_in_scope_files()
# Two test files that must also be free of stale references.
STALE_CHECK_EXTRA_FILES = [
    TESTS_DIR / "test_regression_analysis.py",
    TESTS_DIR / "test_docs.py",
]
# Files in which the word "legacy" must not appear.
LEGACY_WORD_FILES = [
    SCRIPTS_DIR / "regression_analysis.py",
    TESTS_DIR / "test_regression_analysis.py",
]


def test_files_were_found() -> None:
    """The file search finds files, so an empty list cannot pass silently."""
    assert IN_SCOPE_FILES, "No Python files were found to check."
    assert any(SCRIPTS_DIR in path.parents for path in IN_SCOPE_FILES)
    assert not any(TESTS_DIR in path.parents for path in IN_SCOPE_FILES)


def _walk_definitions(node: ast.AST, prefix: str = "") -> Iterator[tuple[ast.AST, str]]:
    """List every class, function and method below a node, however deeply nested.

    Args:
        node: The syntax tree node to search below.
        prefix: Qualified name of the enclosing definition, with a trailing dot.

    Yields:
        Pairs of the definition node and its qualified name, such as
        `outer.<locals>.inner` for a function inside another function.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEFINITION_TYPES):
            name = prefix + child.name
            yield child, name
            separator = "." if isinstance(child, ast.ClassDef) else ".<locals>."
            yield from _walk_definitions(child, name + separator)
        else:
            yield from _walk_definitions(child, prefix)


def _parameter_names(function: ast.AST) -> list[str]:
    """List the parameters a function takes, other than `self` and `cls`.

    Args:
        function: A function or method node.

    Returns:
        Parameter names without leading stars (`*args` appears as `args`).
    """
    a = function.args
    parameters = [*a.posonlyargs, *a.args, *a.kwonlyargs]
    parameters.extend(p for p in (a.vararg, a.kwarg) if p)
    return [p.arg for p in parameters if p.arg not in ("self", "cls")]


def _own_body_kinds(function: ast.AST) -> set[str]:
    """Find what a function's own body does, leaving out nested definitions.

    Nested functions, classes and lambdas are not searched. They are checked
    on their own, as separate definitions.

    Args:
        function: A function or method node.

    Returns:
        A set that may hold `"return"` (a `return` with a value other than
        `None`), `"yield"` (`yield` or `yield from`) and `"raise"` (any `raise`,
        including a bare `raise` that passes on a caught error).
    """
    kinds: set[str] = set()
    stack = list(function.body)
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Return) and current.value is not None:
            value = current.value
            if not (isinstance(value, ast.Constant) and value.value is None):
                kinds.add("return")
        elif isinstance(current, (ast.Yield, ast.YieldFrom)):
            kinds.add("yield")
        elif isinstance(current, ast.Raise):
            kinds.add("raise")
        if not isinstance(current, _SCOPE_TYPES):
            stack.extend(ast.iter_child_nodes(current))
    return kinds


def _section_lines(docstring_lines: list[str], title: str) -> list[str] | None:
    """Pick out the lines that belong to one docstring section.

    The lines must come from a cleaned docstring (`ast.get_docstring` cleans
    it), so the base indentation is zero. A section starts at a line that is
    exactly the title and a colon, at the base indentation. A heading inside an
    indented block (such as `Examples:`) does not count. Neither does a heading
    with text after it on the same line (`Returns: The value.`). The section
    ends at the next line at the base indentation.

    Args:
        docstring_lines: The lines of one cleaned docstring.
        title: Section title without the colon, such as `Args`.

    Returns:
        The lines inside the section, or None if there is no such section or
        it has no text in it.
    """
    for start, line in enumerate(docstring_lines):
        if line.rstrip() != f"{title}:":
            continue
        body: list[str] = []
        for later in docstring_lines[start + 1 :]:
            if later.strip() and later[0] != " ":
                break
            body.append(later)
        return body if any(text.strip() for text in body) else None
    return None


def _has_entry(section_body: list[str] | None, pattern: re.Pattern[str]) -> bool:
    """Say whether a section has an entry that matches a pattern.

    An entry is a line at the section's first indentation level.

    Args:
        section_body: The lines inside a section, or None if it is missing.
        pattern: Regular expression to match against each entry without its
            indentation.

    Returns:
        True if at least one entry matches.
    """
    content = [line for line in section_body or [] if line.strip()]
    if not content:
        return False
    indent = len(content[0]) - len(content[0].lstrip())
    return any(
        pattern.match(line.strip())
        for line in content
        if len(line) - len(line.lstrip()) == indent
    )


def _numpy_headings(docstring_lines: list[str]) -> list[str]:
    """Find NumPy-style headings: a line of text followed by a line of dashes.

    Args:
        docstring_lines: The lines of one docstring.

    Returns:
        The heading text of each NumPy-style section found.
    """
    return [
        line.strip()
        for line, below in zip(docstring_lines, docstring_lines[1:])
        if line.strip()
        and not _UNDERLINE.match(line.strip())
        and _UNDERLINE.match(below.strip())
    ]


def _check_definition(node: ast.AST, name: str, line: int, path: str) -> list[Problem]:
    """Check the docstring of one module, class, function or method.

    Args:
        node: The module, class or function node.
        name: Qualified name to show in each problem.
        line: Line number to show in each problem.
        path: File name to show in each problem.

    Returns:
        Every problem found for this definition.
    """
    problems: list[Problem] = []

    def report(rule: str, message: str) -> None:
        """Record one problem for this definition.

        Args:
            rule: Which rule the problem breaks.
            message: Plain-English description of the problem.
        """
        problems.append(Problem(rule, path, line, name, message))

    docstring = ast.get_docstring(node)
    if not docstring:
        report(R_DOCSTRING, "no docstring")
        return problems
    lines = docstring.splitlines()
    for heading in _numpy_headings(lines):
        report(R_NUMPY, f"NumPy-style heading {heading!r}; use Google style")
    if not isinstance(node, _FUNCTION_TYPES):
        return problems

    args_body = _section_lines(lines, "Args")
    for parameter in _parameter_names(node):
        entry = re.compile(rf"^\*{{0,2}}{re.escape(parameter)}\s*(\(.*\))?:")
        if not _has_entry(args_body, entry):
            report(R_ARGS, f"no Args entry for {parameter!r}")

    kinds = _own_body_kinds(node)
    if "yield" in kinds:
        if _section_lines(lines, "Yields") is None:
            report(R_RETURNS, "no Yields section")
    elif "return" in kinds and _section_lines(lines, "Returns") is None:
        report(R_RETURNS, "no Returns section")
    if "raise" in kinds:
        if not _has_entry(_section_lines(lines, "Raises"), _EXCEPTION_ENTRY):
            report(R_RAISES, "no Raises section naming an exception type")
    return problems


def check_source(source: str, path: str) -> list[Problem]:
    """Check every docstring in one piece of Python source against the rules.

    Args:
        source: Full text of a Python file.
        path: Name to show for the file in each problem.

    Returns:
        Every problem found, in the order of the definitions. The list is
        empty if all rules are met.
    """
    tree = ast.parse(source, filename=path)
    problems = _check_definition(tree, "<module>", 1, path)
    for node, name in _walk_definitions(tree):
        problems.extend(_check_definition(node, name, node.lineno, path))
    return problems


def find_forbidden_text(
    text: str, path: str, patterns: list[re.Pattern[str]]
) -> list[Problem]:
    """Find every match of some patterns in a text, one problem per match.

    Args:
        text: Full text of a file.
        path: Name to show for the file in each problem.
        patterns: Regular expressions that must not appear in the text.

    Returns:
        One problem for each match, in file order.
    """
    return [
        Problem("forbidden text", path, number, "<text>", f"found {m.group(0)!r}")
        for number, line in enumerate(text.splitlines(), start=1)
        for pattern in patterns
        for m in pattern.finditer(line)
    ]


def _fail_with_list(heading: str, problems: list[Problem]) -> None:
    """Fail the current test with one readable list of problems.

    Args:
        heading: One-line description of what was checked.
        problems: The problems to list, as `path:line name: message`. Nothing
            happens if this is empty.
    """
    if problems:
        listing = "\n".join(
            f"  {p.path}:{p.line} {p.name}: {p.message}" for p in problems
        )
        pytest.fail(f"{heading}: {len(problems)} problem(s)\n{listing}", pytrace=False)


def _search_files(files: list[Path], patterns: list[re.Pattern[str]]) -> list[Problem]:
    """Search some files for forbidden text.

    Args:
        files: Absolute paths of the files to search. Each must exist.
        patterns: Regular expressions that must not appear.

    Returns:
        Every match found, across all the files.
    """
    problems: list[Problem] = []
    for path in files:
        name = path.relative_to(REPO_ROOT).as_posix()
        assert path.is_file(), f"{name} does not exist"
        text = path.read_text(encoding="utf-8")
        problems.extend(find_forbidden_text(text, name, patterns))
    return problems


@cache
def _docstring_problems() -> tuple[Problem, ...]:
    """Check every in-scope file once, and keep the result for later tests.

    Returns:
        Every docstring problem across all in-scope files.
    """
    problems: list[Problem] = []
    for path in IN_SCOPE_FILES:
        name = path.relative_to(REPO_ROOT).as_posix()
        problems.extend(check_source(path.read_text(encoding="utf-8"), name))
    return tuple(problems)


@pytest.mark.parametrize("rule", DOCSTRING_RULES)
def test_docstring_rule(rule: str) -> None:
    """Every in-scope definition meets one docstring rule.

    Args:
        rule: Name of the rule to check.
    """
    problems = [problem for problem in _docstring_problems() if problem.rule == rule]
    _fail_with_list(f"Rule '{rule}'", problems)


def test_no_stale_references() -> None:
    """No file mentions a deleted file or test, or a plan contract number."""
    files = [*IN_SCOPE_FILES, *STALE_CHECK_EXTRA_FILES]
    _fail_with_list("Stale references", _search_files(files, STALE_PATTERNS))


def test_regression_files_do_not_say_legacy() -> None:
    """The regression script and its test do not use the word "legacy"."""
    problems = _search_files(LEGACY_WORD_FILES, [LEGACY_PATTERN])
    _fail_with_list("The word 'legacy' in the regression files", problems)


_MODULE_HEADER = '"""Sample module."""\n'

_CORRECT_SAMPLE = _MODULE_HEADER + '''
def combine(first, second, *rest, mode="sum", **options):
    """Combine values.

    Args:
        first: The first value.
        second (int): The second value.
        *rest: More values.
        mode: How to combine them.
        **options: Extra settings.

    Returns:
        The combined value.

    Raises:
        ValueError: If the mode is not known.
    """
    if mode != "sum":
        raise ValueError(mode)
    return first + second


def count_up(limit):
    """Count from zero.

    Args:
        limit: Where to stop.

    Yields:
        Each whole number below the limit.
    """
    yield limit


def outer():
    """Define an inner function that returns a value; outer does not."""
    def inner():
        """Give a number.

        Returns:
            The number three.
        """
        return 3

    if inner() == 3:
        return None
    return
'''


def _function(header: str, docstring: str, body: str) -> str:
    """Build the source code of one function with the given docstring.

    Args:
        header: The `def` line without its colon, such as `def f(a, b)`.
        docstring: Docstring text, with `\n` between lines and no indentation.
        body: The statement that forms the function body.

    Returns:
        Python source code for the function, after a blank line.
    """
    indented = docstring.replace("\n", "\n    ")
    return f'\n{header}:\n    """{indented}\n    """\n    {body}\n'


# Each row is one function with one fault: the rule it breaks, the name the
# problem should give, the `def` line, the docstring text and the body.
_FAULTY_FUNCTIONS = [
    (R_NUMPY, "f", "def f()", "Do.\n\nNotes\n-----\nNumPy style.", "pass"),
    (R_ARGS, "f", "def f(a, b)", "Do.\n\nArgs:\n    a: One.", "pass"),
    (R_ARGS, "f", "def f(a)", "Do.", "pass"),
    (R_RETURNS, "f", "def f()", "Do.", "return 3"),
    (R_RETURNS, "f", "def f()", "Do.\n\nReturns:", "return 3"),
    (R_RETURNS, "f", "def f()", "Do.\n\nTip:\n    Returns:\n        3.", "return 3"),
    (R_RETURNS, "f", "async def f()", "Do.", "return 3"),
    (R_RETURNS, "f", "def f()", "Do.", "yield 1"),
    (R_RAISES, "f", "def f()", "Do.", "raise E"),
    (R_RAISES, "f", "def f()", "Do.\n\nRaises:\n    It failed.", "raise E"),
]
# Rows of the same kind, for faults that are not in a plain function.
_FAULTY_STRUCTURES = [
    (R_DOCSTRING, "outer.<locals>.inner", "def outer():\n 'Do.'\n def inner(): pass"),
    (R_DOCSTRING, "C", "class C:\n value = 1"),
    (R_DOCSTRING, "C.m", "class C:\n 'A class.'\n def m(self): pass"),
]
_WRONG_SAMPLES = [
    (rule, name, _function(header, docstring, body))
    for rule, name, header, docstring, body in _FAULTY_FUNCTIONS
] + _FAULTY_STRUCTURES


def test_correct_google_docstrings_pass() -> None:
    """A sample with correct Google-style docstrings has no problems."""
    assert check_source(_CORRECT_SAMPLE, "sample.py") == []


@pytest.mark.parametrize(("expected_rule", "expected_name", "body"), _WRONG_SAMPLES)
def test_wrong_docstrings_are_flagged(
    expected_rule: str, expected_name: str, body: str
) -> None:
    """A sample with one docstring fault is flagged for that fault only.

    Args:
        expected_rule: The rule the fault breaks.
        expected_name: The definition that should be named in the problem.
        body: Python source code with one fault, without a module docstring.
    """
    problems = check_source(_MODULE_HEADER + body, "sample.py")
    found = {(problem.rule, problem.name) for problem in problems}
    assert found == {(expected_rule, expected_name)}, problems


def test_missing_module_docstring_is_flagged() -> None:
    """A file that does not start with a docstring is flagged."""
    problems = check_source("value = 1\n", "sample.py")
    found = [(problem.rule, problem.name) for problem in problems]
    assert found == [(R_DOCSTRING, "<module>")]


@pytest.mark.parametrize(
    "text",
    [
        "see regression_analyses/ for more",
        "import ModellingResults",
        "def test_legacy_equivalence():",
        "# C-04 says so",
        "# AC-12 says so",
    ],
)
def test_stale_references_are_flagged(text: str) -> None:
    """A stale reference in a line of text is flagged once.

    Args:
        text: One line of sample text with a stale reference.
    """
    problems = find_forbidden_text(text, "sample.py", STALE_PATTERNS)
    assert len(problems) == 1, problems


def test_ordinary_text_is_not_flagged_as_stale() -> None:
    """Text that only looks a little like a reference is not flagged."""
    text = "# Ordinary comment about C-4, ABC-04 and an older format"
    assert find_forbidden_text(text, "sample.py", STALE_PATTERNS) == []


def test_legacy_word_is_flagged_in_any_capitalisation() -> None:
    """The word "legacy" is found, whether or not it is capitalised."""
    text = "Legacy scripts and the legacy tests"
    assert len(find_forbidden_text(text, "sample.py", [LEGACY_PATTERN])) == 2
