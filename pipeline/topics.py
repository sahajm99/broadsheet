"""TREC topics and relevance judgments.

The topic file is the Robust 2004 set (301-450, 601-700) in the usual SGML-ish
shape, but it is not one shape: the 301-450 block writes `<title> text` on the
tag line and labels the other fields (`<desc> Description:`), while the
601-700 block uses bare tags with the text on the following lines. Both are
read here, because the evaluation runs over all 250 topics.

The qrels file is `topic 0 docno rel`, already filtered to `FT911-` documents
(decision D3). Relevance is binarised: TREC used 2 for "highly relevant" in
some years and every metric in this project is binary (D5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.constants import QRELS_PATH, TOPICS_PATH

FIELDS = {
    "title": "Title only",
    "title_desc": "Title + description",
    "title_desc_narr": "Title + description + narrative",
}

_TOP_RE = re.compile(r"<top>(.*?)</top>", re.DOTALL)
_NUM_RE = re.compile(r"<num>\s*(?:Number:)?\s*(\d+)")
# Each field runs from its tag to the next tag, so a title on its own line and
# a title on the tag line are the same case; the optional label is dropped.
_TITLE_RE = re.compile(r"<title>(.*?)(?=<desc>|<narr>|\Z)", re.DOTALL)
_DESC_RE = re.compile(r"<desc>\s*(?:Description:)?(.*?)(?=<narr>|\Z)", re.DOTALL)
_NARR_RE = re.compile(r"<narr>\s*(?:Narrative:)?(.*?)\Z", re.DOTALL)


@dataclass(frozen=True)
class Topic:
    """One TREC topic with its three query fields, whitespace collapsed."""

    num: int
    title: str
    desc: str
    narr: str


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _field(pattern: re.Pattern[str], block: str, name: str, num: str) -> str:
    match = pattern.search(block)
    if match is None:
        raise ValueError(f"topic {num} has no <{name}>")
    return _collapse(match.group(1))


def load_topics(path: Path = TOPICS_PATH) -> list[Topic]:
    """Every `<top>` block in the file, ascending by topic number."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    topics: list[Topic] = []
    for block in _TOP_RE.findall(raw):
        num = _NUM_RE.search(block)
        if num is None:
            raise ValueError(f"{path}: a <top> block has no <num>")
        topics.append(
            Topic(
                num=int(num.group(1)),
                title=_field(_TITLE_RE, block, "title", num.group(1)),
                desc=_field(_DESC_RE, block, "desc", num.group(1)),
                narr=_field(_NARR_RE, block, "narr", num.group(1)),
            )
        )
    topics.sort(key=lambda t: t.num)
    return topics


def query_text(topic: Topic, field: str) -> str:
    """The query string for one topic under one field combination."""
    if field == "title":
        return topic.title
    if field == "title_desc":
        return f"{topic.title} {topic.desc}"
    if field == "title_desc_narr":
        return f"{topic.title} {topic.desc} {topic.narr}"
    raise ValueError(f"unknown field {field!r}; expected one of {sorted(FIELDS)}")


def load_qrels(
    path: Path = QRELS_PATH, docnos: set[str] | None = None
) -> dict[int, dict[str, int]]:
    """`topic -> docno -> 1 if judged relevant else 0`.

    Any positive TREC grade counts as relevant. When `docnos` is given, only
    judgments for documents in the collection are kept, and a topic whose
    judgments all fall outside it disappears rather than arriving empty.
    """
    qrels: dict[int, dict[str, int]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4:
            raise ValueError(f"{path}: expected 'topic 0 docno rel', got {line!r}")
        topic, _, docno, rel = parts
        if docnos is not None and docno not in docnos:
            continue
        qrels.setdefault(int(topic), {})[docno] = 1 if int(rel) > 0 else 0
    return qrels
