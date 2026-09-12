"""Read the TREC SGML records of the FT 1991 corpus into `Doc` objects.

The corpus shape is fixed and regular: every `<DOC>` carries `<DOCNO>`,
`<PROFILE>`, `<DATE>`, `<HEADLINE>`, `<TEXT>`, `<PUB>` and `<PAGE>`, with
`<BYLINE>` and `<DATELINE>` on some records. Only DOCNO, DATE, HEADLINE and
TEXT are kept; the rest is metadata that the index does not need.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.constants import RAW_DIR

_DOC_RE = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
_DOCNO_RE = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.DOTALL)
_DATE_RE = re.compile(r"<DATE>(.*?)</DATE>", re.DOTALL)
_HEADLINE_RE = re.compile(r"<HEADLINE>(.*?)</HEADLINE>", re.DOTALL)
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

# "FT  14 MAY 91 / " — the paper-and-date stamp the FT puts on every headline.
_FT_PREFIX_RE = re.compile(r"^\s*FT\s+\d{1,2}\s+[A-Z]{3}\s+\d{2}\s*/\s*")
_WS_RE = re.compile(r"\s+")

HEADLINE_LIMIT = 200

_MISSING_RAW = (
    "No corpus files found in {path}. The FT 1991 corpus is licensed and is not "
    "committed to this repository; see data/raw/SOURCE.md for what to put there."
)


@dataclass(frozen=True)
class Doc:
    """One article: `date` is ISO `YYYY-MM-DD`, `headline` is cleaned."""

    docno: str
    date: str
    headline: str
    text: str


def iso_date(yymmdd: str) -> str:
    """`"910514"` -> `"1991-05-14"`. The corpus is 1991; two-digit years
    90-99 are 19xx, everything else 20xx."""
    yymmdd = yymmdd.strip()
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
    century = "19" if int(yy) >= 90 else "20"
    return f"{century}{yy}-{mm}-{dd}"


def clean_headline(raw: str) -> str:
    """Drop the `FT  14 MAY 91 /` stamp, unwrap the line breaks the SGML puts
    in mid-headline, and cut over-long headlines to 200 characters."""
    headline = _FT_PREFIX_RE.sub("", html.unescape(raw))
    headline = _WS_RE.sub(" ", headline).strip()
    return headline[:HEADLINE_LIMIT]


def _clean_text(raw: str) -> str:
    return html.unescape(raw).strip()


def parse_file(path: Path) -> list[Doc]:
    """Every `<DOC>` record in one corpus file, in file order."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    docs = []
    for record in _DOC_RE.findall(raw):
        docno = _DOCNO_RE.search(record)
        date = _DATE_RE.search(record)
        headline = _HEADLINE_RE.search(record)
        text = _TEXT_RE.search(record)
        if docno is None or date is None or text is None:
            continue
        docs.append(
            Doc(
                docno=docno.group(1).strip(),
                date=iso_date(date.group(1)),
                headline=clean_headline(headline.group(1)) if headline else "",
                text=_clean_text(text.group(1)),
            )
        )
    return docs


def _docno_key(docno: str) -> int:
    return int(docno.split("-")[1])


def load_corpus(raw_dir: Path = RAW_DIR) -> list[Doc]:
    """Every article in the corpus, sorted by document number.

    Raises `FileNotFoundError` naming `data/raw/SOURCE.md` when the corpus is
    not in place, because it is licensed and cannot be shipped with the code.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(_MISSING_RAW.format(path=raw_dir))
    files = sorted(p for p in raw_dir.iterdir() if p.is_file())
    docs: list[Doc] = []
    for path in files:
        docs.extend(parse_file(path))
    if not docs:
        raise FileNotFoundError(_MISSING_RAW.format(path=raw_dir))
    docs.sort(key=lambda d: _docno_key(d.docno))
    return docs
