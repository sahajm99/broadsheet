"""The index the browser downloads, and the fixtures that keep it honest.

Three exports live here, and all three exist because the page runs its own
search rather than asking a server:

- `export_search_index` writes `search/index.json.gz`, the whole `stem_stop`
  index as gap-encoded postings plus one citation line per document. It ships
  stems, document numbers, dates and headlines and **never** article TEXT
  (D1/D2): a stemmed bag of words with term frequencies cannot reconstruct an
  article, and a headline with a date is a citation, which is what a search
  result is.
- `export_golden` writes the parity fixtures. Task 7's TypeScript ranks in the
  browser; these files are the pipeline's answer for the same 250 queries, so
  a port that disagrees fails a test instead of quietly ranking differently
  from every number on the page (W3, X2).
- `export_stopwords` hands the browser the same 523-word list the index was
  built with (D6). One list, one file, no second copy to drift.

Nothing is pruned from the index. A term with df 1 is 4 bytes of JSON and is
the difference between a search that finds a rare name and one that shrugs.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.constants import BM25_B, BM25_K1, GOLDEN_DIR, SIZE_LIMIT
from pipeline.index import Index
from pipeline.io import compact_json, gzip_json, round6, write_bytes, write_json
from pipeline.parse import Doc
from pipeline.rank import Scorer, run_topics
from pipeline.topics import Topic

SCHEMA = "search_index.v1"

# The golden fixtures are test data, not a page download, so the site's
# 200,000-byte budget does not apply to them: 250 topics x 10 results is
# comfortably bigger and is fetched by nobody.
GOLDEN_LIMIT = 2_000_000

GOLDEN_RUNS = {
    "top10_bm25_title.json": "bm25",
    "top10_tfidf_raw_title.json": "tfidf_raw",
}
TOPIC_TITLES = "topic_titles.json"
GOLDEN_TOP_N = 10


# --------------------------------------------------------- the gz payload ---


def _gap_encode(postings: list[tuple[int, int]]) -> list[int]:
    """`[(doc, tf), ...]` -> `[gap, tf, gap, tf, ...]`.

    The first gap is the first document index itself, so a term in documents
    0, 3 and 4 becomes `[0, tf, 3, tf, 1, tf]`. Gaps are small even in a wide
    posting list, which is most of why the gzip is megabytes and not tens of
    them, and decoding is a running sum -- one pass, no lookups.
    """
    flat: list[int] = []
    previous = 0
    for doc_idx, tf in postings:
        flat.append(doc_idx - previous)
        flat.append(tf)
        previous = doc_idx
    return flat


def _check_order(index: Index, docs: list[Doc]) -> None:
    """Refuse a `docs` list that is not the one the index was built from.

    Everything here addresses documents by position -- the gap-encoded
    postings, the `docs` array, `doc_len` -- so a mismatched list would not
    fail, it would silently attach the wrong headline to every result.
    """
    if len(docs) != len(index.docnos) or any(
        doc.docno != docno for doc, docno in zip(docs, index.docnos)
    ):
        raise ValueError(
            "docs and index disagree about document order; the search index "
            "addresses documents by position and cannot be built from a "
            "different ordering"
        )


def build_search_index(index: Index, docs: list[Doc]) -> dict:
    """The complete `search_index.v1` object for one index.

    `docs` supplies the display metadata and must be the same list, in the
    same order, that the index was built from.
    """
    _check_order(index, docs)
    return {
        "schema": SCHEMA,
        "n_docs": index.n_docs,
        "avgdl": round6(index.avgdl),
        "k1": BM25_K1,
        "b": BM25_B,
        "docs": [{"no": d.docno, "d": d.date, "h": d.headline} for d in docs],
        "doc_len": list(index.doc_len),
        # Sorted so the file has one canonical order regardless of the order
        # the corpus happened to introduce terms in.
        "terms": {
            term: _gap_encode(index.postings[term]) for term in sorted(index.postings)
        },
    }


def export_search_index(index: Index, docs: list[Doc], out: Path) -> dict:
    """Write the gz and report both sizes: `{"index_gz", "index_json"}`.

    `postings.json` prints both -- the compressed number is what a visitor
    pays (GitHub Pages does not compress unknown types, W2) and the raw number
    is what the browser holds once it has inflated it.
    """
    obj = build_search_index(index, docs)
    payload = gzip_json(obj)
    write_bytes(out, payload)
    return {"index_gz": len(payload), "index_json": len(compact_json(obj))}


# ------------------------------------------------------------- the golden ---


def _top10(
    index: Index,
    topics: list[Topic],
    stopwords: frozenset[str],
    ranker: str,
) -> dict[str, list[dict]]:
    """One ranker's top ten per topic, title field, topics with hits only."""
    run = run_topics(
        index, Scorer(index, ranker), topics, "title", stopwords, GOLDEN_TOP_N
    )
    return {
        str(topic.num): [
            {"no": docno, "score": round6(score)} for docno, score in run[topic.num]
        ]
        for topic in topics
        if run[topic.num]
    }


def export_golden(
    index: Index,
    docs: list[Doc],
    topics: list[Topic],
    stopwords: frozenset[str],
    golden_dir: Path = GOLDEN_DIR,
) -> dict[str, int]:
    """Write the three parity fixtures; return `{filename: bytes}`.

    Keys are topic numbers as strings, in ascending numeric order, because
    that is how the TypeScript test iterates them and a JSON object has no
    order of its own to fall back on. `docs` is taken only to check that the
    index really is the one built from this corpus, since the DOCNOs written
    here are what the browser's answers will be compared against.
    """
    _check_order(index, docs)
    golden_dir = Path(golden_dir)
    written = {
        TOPIC_TITLES: write_json(
            golden_dir / TOPIC_TITLES,
            {str(topic.num): topic.title for topic in topics},
            limit=GOLDEN_LIMIT,
        )
    }
    for name, ranker in GOLDEN_RUNS.items():
        written[name] = write_json(
            golden_dir / name,
            _top10(index, topics, stopwords, ranker),
            limit=GOLDEN_LIMIT,
        )
    return written


def export_stopwords(stopwords: frozenset[str], path: Path) -> int:
    """Write the stopword list the browser must use, sorted; return bytes."""
    return write_json(Path(path), sorted(stopwords), limit=SIZE_LIMIT)
