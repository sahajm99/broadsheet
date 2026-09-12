"""The inverted index: term -> [(doc index, term frequency)].

Documents are addressed by their position in the `docs` list, not by DOCNO, so
a posting is two small integers and the browser index in Task 6 can store them
as gap-encoded arrays. `docnos[i]` maps an index back to `FT911-...`.

Only the TEXT field is indexed. The headline is display metadata: including it
would double-count the words an editor chose to repeat in the lede and would
make the document-length distribution disagree with the article body.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pipeline.constants import TREATMENTS
from pipeline.parse import Doc
from pipeline.tokenize import analyze


@dataclass
class Index:
    """One treatment's complete index over the corpus.

    `postings[term]` is ascending by document index and every `tf` is > 0.
    `doc_len[i]` is the number of terms kept in document `i` (after stopword
    removal and stemming), which is what BM25 length-normalises against.
    """

    docnos: list[str]
    doc_len: list[int]
    postings: dict[str, list[tuple[int, int]]]
    n_docs: int
    avgdl: float
    total_tokens: int
    stem: bool
    stop: bool
    _cf: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def df(self, term: str) -> int:
        """In how many documents `term` occurs."""
        return len(self.postings.get(term, ()))

    def cf(self, term: str) -> int:
        """How many times `term` occurs in the whole collection."""
        if not self._cf:
            self._cf = {
                t: sum(tf for _, tf in plist) for t, plist in self.postings.items()
            }
        return self._cf.get(term, 0)


def build_index(docs: list[Doc], stopwords: frozenset[str] | None, stem: bool) -> Index:
    """Index the TEXT field of every document under one treatment."""
    accumulator: dict[str, dict[int, int]] = defaultdict(dict)
    docnos: list[str] = []
    doc_len: list[int] = []

    for doc_idx, doc in enumerate(docs):
        terms = analyze(doc.text, stopwords, stem)
        docnos.append(doc.docno)
        doc_len.append(len(terms))
        for term in terms:
            per_doc = accumulator[term]
            per_doc[doc_idx] = per_doc.get(doc_idx, 0) + 1

    # Documents are visited in order, so each term's dict is already ascending
    # by document index; freezing to tuples keeps a posting immutable.
    postings = {term: list(per_doc.items()) for term, per_doc in accumulator.items()}
    total_tokens = sum(doc_len)
    n_docs = len(docs)
    return Index(
        docnos=docnos,
        doc_len=doc_len,
        postings=postings,
        n_docs=n_docs,
        avgdl=total_tokens / n_docs if n_docs else 0.0,
        total_tokens=total_tokens,
        stem=stem,
        stop=bool(stopwords),
    )


def build_all(docs: list[Doc], stopwords: frozenset[str]) -> dict[str, Index]:
    """One index per entry of `TREATMENTS`, in that order."""
    return {
        key: build_index(docs, stopwords if stop else None, stem)
        for key, (stem, stop) in TREATMENTS.items()
    }
