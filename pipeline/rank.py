"""The three rankers: the course tf-idf cosine, log tf-idf cosine and BM25.

Scoring is postings-driven. A query touches only the postings of its own
terms, accumulating into a dict of `doc_idx -> partial score`, so cost is the
length of those posting lists and not the size of the collection. For the two
cosine rankers the per-document norms are computed once when the `Scorer` is
built, in one pass over every posting.

The operation order below is load-bearing: Task 7 ports these three functions
to TypeScript and a parity test compares top-10s document by document. A score
is accumulated per document as the sum, over the *distinct* query terms in
order of first appearance, of that term's contribution; for the cosine rankers
the sum is divided once, at the end, by `norm_d * norm_q`. Weights use
`math.log10`, BM25's idf uses `math.log`, and nothing here uses numpy: plain
Python floats are IEEE doubles, which is what JavaScript numbers are too.
"""

from __future__ import annotations

import math
from bisect import bisect_left

from pipeline.constants import BM25_B, BM25_K1, TOP_N
from pipeline.index import Index
from pipeline.tokenize import analyze
from pipeline.topics import Topic, query_text

RANKERS = {
    "tfidf_raw": "tf-idf cosine (course weighting)",
    "tfidf_log": "log tf-idf cosine",
    "bm25": "BM25",
}

_COSINE = ("tfidf_raw", "tfidf_log")


class Scorer:
    """One ranker bound to one index, with idf and document norms cached.

    Build it once per (index, ranker, k1, b) and reuse it for every query: the
    constructor walks all postings for the cosine rankers, which costs about
    as much as a hundred queries.
    """

    def __init__(
        self, index: Index, ranker: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> None:
        if ranker not in RANKERS:
            raise ValueError(
                f"unknown ranker {ranker!r}; expected one of {list(RANKERS)}"
            )
        self.index = index
        self.ranker = ranker
        self.k1 = k1
        self.b = b
        self._idf: dict[str, float] = {}
        self._norm: list[float] = self._doc_norms() if ranker in _COSINE else []

    # -- weights -----------------------------------------------------------

    def idf(self, term: str) -> float:
        """Inverse document frequency; 0.0 for a term outside the vocabulary.

        `log10(N / df)` for the two cosine rankers, the BM25-plus-one variant
        `ln((N - df + 0.5) / (df + 0.5) + 1)` for BM25, which is never
        negative and so cannot subtract score for a very common term.
        """
        cached = self._idf.get(term)
        if cached is not None:
            return cached
        df = self.index.df(term)
        if df == 0:
            value = 0.0
        elif self.ranker == "bm25":
            n = self.index.n_docs
            value = math.log((n - df + 0.5) / (df + 0.5) + 1)
        else:
            value = math.log10(self.index.n_docs / df)
        self._idf[term] = value
        return value

    def _tf_weight(self, tf: int) -> float:
        """The tf half of a cosine weight, used for documents and queries."""
        if self.ranker == "tfidf_log":
            return 1 + math.log10(tf)
        return float(tf)

    def _doc_norms(self) -> list[float]:
        """Euclidean norm of every document vector, in one pass over postings."""
        norm_sq = [0.0] * self.index.n_docs
        for term, plist in self.index.postings.items():
            idf = self.idf(term)
            if idf == 0.0:
                continue
            for doc_idx, tf in plist:
                w = self._tf_weight(tf) * idf
                norm_sq[doc_idx] += w * w
        return [math.sqrt(v) for v in norm_sq]

    def _query_tf(self, query_terms: list[str]) -> dict[str, int]:
        """Distinct query terms that are in the vocabulary, in query order,
        mapped to their query term frequency. Dicts keep insertion order, so
        this fixes the accumulation order the TypeScript port must mirror."""
        qtf: dict[str, int] = {}
        for term in query_terms:
            if term in qtf:
                qtf[term] += 1
            elif term in self.index.postings:
                qtf[term] = 1
        return qtf

    # -- scoring -----------------------------------------------------------

    def score(self, query_terms: list[str]) -> dict[int, float]:
        """`doc_idx -> score` for every document that scores above zero."""
        qtf = self._query_tf(query_terms)
        if not qtf:
            return {}
        if self.ranker == "bm25":
            return self._score_bm25(qtf)
        return self._score_cosine(qtf)

    def _score_cosine(self, qtf: dict[str, int]) -> dict[int, float]:
        postings = self.index.postings
        dot: dict[int, float] = {}
        norm_q_sq = 0.0
        for term, q in qtf.items():
            idf = self.idf(term)
            w_q = self._tf_weight(q) * idf
            norm_q_sq += w_q * w_q
            if w_q == 0.0:
                continue
            for doc_idx, tf in postings[term]:
                w_d = self._tf_weight(tf) * idf
                dot[doc_idx] = dot.get(doc_idx, 0.0) + w_q * w_d
        norm_q = math.sqrt(norm_q_sq)
        if norm_q == 0.0:
            return {}
        norm = self._norm
        scores = {}
        for doc_idx, value in dot.items():
            norm_d = norm[doc_idx]
            if norm_d == 0.0 or value <= 0.0:
                continue
            scores[doc_idx] = value / (norm_d * norm_q)
        return scores

    def _score_bm25(self, qtf: dict[str, int]) -> dict[int, float]:
        postings = self.index.postings
        doc_len = self.index.doc_len
        avgdl = self.index.avgdl
        k1, b = self.k1, self.b
        scores: dict[int, float] = {}
        for term, q in qtf.items():
            idf = self.idf(term)
            if idf == 0.0:
                continue
            for doc_idx, tf in postings[term]:
                dl = doc_len[doc_idx]
                contribution = (
                    q * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
                )
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contribution
        return {d: s for d, s in scores.items() if s > 0.0}

    def rank(
        self, query_terms: list[str], top_n: int = TOP_N
    ) -> list[tuple[int, float]]:
        """The top `top_n` documents, highest score first, ties by doc index."""
        scores = self.score(query_terms)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_n]

    # -- explanation -------------------------------------------------------

    def _tf(self, term: str, doc_idx: int) -> int:
        """Term frequency of `term` in one document, 0 when it does not occur."""
        plist = self.index.postings.get(term)
        if not plist:
            return 0
        pos = bisect_left(plist, doc_idx, key=lambda posting: posting[0])
        if pos < len(plist) and plist[pos][0] == doc_idx:
            return plist[pos][1]
        return 0

    def explain(self, query_terms: list[str], doc_idx: int) -> list[dict]:
        """One row per distinct query term that the document actually has.

        Rows are in query order and their `contribution` values sum to
        `score(query_terms)[doc_idx]`, so the site can show the arithmetic of
        a result without the reader having to trust it. Terms outside the
        vocabulary, and terms absent from this document, are left out.
        """
        qtf = self._query_tf(query_terms)
        if not qtf:
            return []
        rows: list[dict] = []
        if self.ranker == "bm25":
            k1, b = self.k1, self.b
            dl = self.index.doc_len[doc_idx]
            avgdl = self.index.avgdl
            for term, q in qtf.items():
                tf = self._tf(term, doc_idx)
                if tf == 0:
                    continue
                idf = self.idf(term)
                rows.append(
                    {
                        "term": term,
                        "qtf": q,
                        "tf": tf,
                        "df": self.index.df(term),
                        "idf": idf,
                        "contribution": q
                        * idf
                        * tf
                        * (k1 + 1)
                        / (tf + k1 * (1 - b + b * dl / avgdl)),
                    }
                )
            return rows

        norm_q = math.sqrt(
            sum((self._tf_weight(q) * self.idf(term)) ** 2 for term, q in qtf.items())
        )
        denom = self._norm[doc_idx] * norm_q
        if denom == 0.0:
            # Every query term is in every document, so no document scores at
            # all and there is no arithmetic to show.
            return []
        for term, q in qtf.items():
            tf = self._tf(term, doc_idx)
            if tf == 0:
                continue
            idf = self.idf(term)
            w_q = self._tf_weight(q) * idf
            w_d = self._tf_weight(tf) * idf
            rows.append(
                {
                    "term": term,
                    "qtf": q,
                    "tf": tf,
                    "df": self.index.df(term),
                    "idf": idf,
                    "contribution": w_q * w_d / denom,
                }
            )
        return rows


def run_topics(
    index: Index,
    scorer: Scorer,
    topics: list[Topic],
    field: str,
    stopwords: frozenset[str] | None,
    top_n: int = TOP_N,
) -> dict[int, list[tuple[str, float]]]:
    """One run: `topic number -> [(docno, score)]`, ranked.

    The query is analysed exactly as the index was built, which is the whole
    point of passing the index in: a stemmed index gets stemmed query terms
    and an index built with stopwords removed gets them removed from the
    query too.
    """
    stops = stopwords if index.stop else None
    docnos = index.docnos
    runs: dict[int, list[tuple[str, float]]] = {}
    for topic in topics:
        terms = analyze(query_text(topic, field), stops, index.stem)
        runs[topic.num] = [
            (docnos[doc_idx], score) for doc_idx, score in scorer.rank(terms, top_n)
        ]
    return runs
