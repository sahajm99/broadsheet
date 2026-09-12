"""The course tokenizer: lowercase, split on non-alphanumerics, discard any
token that contains a digit; then optionally stopword-filter and stem."""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.constants import STOPWORDS_PATH
from pipeline.porter import stem as porter_stem

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HAS_DIGIT_RE = re.compile(r"\d")


def tokenize(text: str) -> list[str]:
    """Alphanumeric runs of the lowercased text, minus anything with a digit.

    Dropping digit-bearing tokens is the course rule, so `"3"`, `"1991"` and
    `"B2"` all disappear while `"jet"` and `"s"` (from `"jet's"`) survive.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if not _HAS_DIGIT_RE.search(t)]


def load_stopwords(path: Path = STOPWORDS_PATH) -> frozenset[str]:
    """The course stopword list: one word per line, indented in the file as
    issued, so every line is stripped and lowercased; blanks are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return frozenset(word for word in (line.strip().lower() for line in lines) if word)


def analyze(text: str, stopwords: frozenset[str] | None, stem: bool) -> list[str]:
    """Tokenize, drop stopwords when a list is given, stem when asked.

    Stopwords are removed before stemming, which is what the course pipeline
    does: the list is a list of surface words, not of stems.
    """
    tokens = tokenize(text)
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    if stem:
        tokens = [porter_stem(t) for t in tokens]
    return tokens
