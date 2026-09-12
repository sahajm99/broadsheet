"""Structural checks over the committed `notebooks/analysis.ipynb`.

The notebook is committed **with** its outputs, because the corpus it reads is
licensed and cannot be in CI (decision P1): the outputs are the only evidence in
the repository that the narrative ran. That makes two things worth asserting on
every push.

- Every code cell has output and none of it is a traceback, so a notebook that
  was regenerated and never executed, or executed with an error, fails here
  instead of being read as a result.
- No output holds a sentence from an article. The corpus is not redistributable
  (D1) and the notebook prints headlines, dates and document numbers only; a
  stray `print(doc.text)` would leak licensed text into a public repository, and
  a committed output is published the moment it is pushed. The canary is a
  sentence from the TEXT of `FT911-1`, the first article in the collection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks/analysis.ipynb"

# A sentence from the TEXT of FT911-1. If it ever appears in an output, some cell
# printed the body of an article.
CANARY = "Exactly 50 years ago yesterday"

# Fields an output can carry text in.
TEXT_KEYS = ("text", "evalue", "ename", "traceback")


def _load() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(
            f"{NOTEBOOK.name} is not present; build it with "
            "`uv run python notebooks/build_notebook.py` and execute it with "
            "`uv run jupyter nbconvert --to notebook --execute --inplace "
            "notebooks/analysis.ipynb`"
        )
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def notebook() -> dict:
    return _load()


@pytest.fixture(scope="module")
def code_cells(notebook: dict) -> list[tuple[int, dict]]:
    return [
        (i, cell)
        for i, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    ]


def _output_text(output: dict) -> str:
    """Every string an output carries, joined -- stream text, values, tracebacks
    and the text/plain of a display bundle. Image payloads are skipped."""
    parts: list[str] = []
    for key in TEXT_KEYS:
        value = output.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    data = output.get("data", {})
    for mime, value in data.items():
        if mime.startswith("image/"):
            continue
        parts.append(value if isinstance(value, str) else "".join(map(str, value)))
    return "\n".join(parts)


def test_notebook_is_present_and_v4(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "the notebook has no cells"


def test_there_are_code_and_markdown_cells(notebook: dict, code_cells) -> None:
    markdown = [c for c in notebook["cells"] if c.get("cell_type") == "markdown"]
    assert len(code_cells) >= 20, "the notebook lost most of its code cells"
    assert len(markdown) >= 10, "the notebook lost most of its commentary"


def test_every_code_cell_has_output(code_cells) -> None:
    """A cell with no output was never executed, or prints nothing worth reading."""
    empty = [i for i, cell in code_cells if not cell.get("outputs")]
    assert not empty, f"code cells with no output: {empty}"


def test_no_code_cell_has_an_error_output(code_cells) -> None:
    failures = []
    for i, cell in code_cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                failures.append(f"cell {i}: {output.get('ename')}: {output.get('evalue')}")
    assert not failures, "the notebook was executed with errors:\n" + "\n".join(failures)


def test_every_code_cell_was_executed(code_cells) -> None:
    unexecuted = [i for i, cell in code_cells if cell.get("execution_count") is None]
    assert not unexecuted, f"code cells with no execution count: {unexecuted}"


def test_no_output_contains_article_text(code_cells) -> None:
    """The corpus is licensed and is not redistributed (D1)."""
    leaks = [
        i
        for i, cell in code_cells
        for output in cell.get("outputs", [])
        if CANARY in _output_text(output)
    ]
    assert not leaks, (
        f"article text from FT911-1 appears in the output of cells {leaks}; "
        "the notebook may print headlines, dates and document numbers only"
    )


def test_the_canary_is_not_anywhere_in_the_file() -> None:
    """Cheap belt and braces: the sentence must not be in the source either."""
    if not NOTEBOOK.exists():
        pytest.skip("notebooks/analysis.ipynb is not present")
    assert CANARY not in NOTEBOOK.read_text(encoding="utf-8")


def test_the_file_has_lf_endings_and_is_small() -> None:
    if not NOTEBOOK.exists():
        pytest.skip("notebooks/analysis.ipynb is not present")
    data = NOTEBOOK.read_bytes()
    assert b"\r\n" not in data, (
        "the notebook has CRLF endings; run "
        "`uv run python notebooks/build_notebook.py --normalise-eol`"
    )
    assert len(data) < 1_500_000, f"the notebook is {len(data):,} bytes, over the 1.5 MB budget"
