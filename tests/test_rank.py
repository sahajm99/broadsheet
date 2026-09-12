"""Ranker tests on a three-document hand example.

Every expected number is derived from the formulas in the brief, independently
of the implementation: the cosine scores are closed forms (3/sqrt(10), 1/2,
sqrt(9/26) for `tfidf_raw`) and the BM25 figures are the ones worked out by
hand in the brief. The TypeScript port in Task 7 is checked against the same
numbers, so they may not be adjusted to fit an implementation.
"""

import math
from pathlib import Path

import pytest

from pipeline.index import build_index
from pipeline.parse import Doc, load_corpus
from pipeline.rank import RANKERS, Scorer, run_topics
from pipeline.topics import load_topics

FIXTURE = Path(__file__).parent / "fixtures" / "mini"

TEXTS = [
    "apple apple banana",
    "apple cherry",
    "banana banana banana cherry cherry",
]
QUERY = ["apple", "banana"]


@pytest.fixture(scope="module")
def index():
    docs = [
        Doc(docno=f"D{i}", date="1991-05-14", headline=f"H{i}", text=text)
        for i, text in enumerate(TEXTS)
    ]
    return build_index(docs, None, stem=False)


def test_rankers_are_the_three_named_runs():
    assert list(RANKERS) == ["tfidf_raw", "tfidf_log", "bm25"]


def test_unknown_ranker_is_rejected(index):
    with pytest.raises(ValueError, match="ranker"):
        Scorer(index, "lmdirichlet")


def test_tfidf_raw_idf_is_log10_n_over_df(index):
    scorer = Scorer(index, "tfidf_raw")
    assert scorer.idf("apple") == pytest.approx(math.log10(1.5), abs=1e-9)
    assert scorer.idf("apple") == pytest.approx(0.176091, abs=1e-6)
    assert scorer.idf("durian") == 0.0


def test_tfidf_raw_scores_match_the_closed_form(index):
    scores = Scorer(index, "tfidf_raw").score(QUERY)
    assert set(scores) == {0, 1, 2}
    assert scores[0] == pytest.approx(3 / math.sqrt(10), abs=1e-6)
    assert scores[0] == pytest.approx(0.948683, abs=1e-6)
    assert scores[1] == pytest.approx(0.5, abs=1e-6)
    assert scores[2] == pytest.approx(math.sqrt(9 / 26), abs=1e-6)
    assert scores[2] == pytest.approx(0.588348, abs=1e-6)


def test_tfidf_raw_rank_order(index):
    ranked = Scorer(index, "tfidf_raw").rank(QUERY)
    assert [doc for doc, _ in ranked] == [0, 2, 1]
    assert ranked[0][1] == pytest.approx(0.948683, abs=1e-6)


def test_tfidf_log_scores(index):
    scores = Scorer(index, "tfidf_log").score(QUERY)
    assert scores[0] == pytest.approx(0.991551, abs=1e-6)
    assert scores[1] == pytest.approx(0.5, abs=1e-6)
    assert scores[2] == pytest.approx(0.530627, abs=1e-6)


def test_bm25_idf_and_scores(index):
    scorer = Scorer(index, "bm25")
    assert index.avgdl == pytest.approx(10 / 3)
    assert scorer.idf("apple") == pytest.approx(math.log(1.6), abs=1e-9)
    assert scorer.idf("apple") == pytest.approx(0.470004, abs=1e-6)
    scores = scorer.score(QUERY)
    assert scores[0] == pytest.approx(1.155010, abs=1e-5)
    assert scores[0] == pytest.approx(1.155008, abs=1e-6)
    assert scores[1] == pytest.approx(0.561961, abs=1e-6)
    assert scores[2] == pytest.approx(0.667102, abs=1e-6)
    assert [doc for doc, _ in scorer.rank(QUERY)] == [0, 2, 1]


def test_bm25_parameters_are_honoured(index):
    default = Scorer(index, "bm25").score(QUERY)
    flat = Scorer(index, "bm25", k1=1.2, b=0.0).score(QUERY)
    assert flat[0] != pytest.approx(default[0], abs=1e-9)
    # With b = 0 there is no length normalisation: the denominator is tf + k1.
    tf, k1 = 2.0, 1.2
    apple = 1 * math.log(1.6) * tf * (k1 + 1) / (tf + k1)
    banana = 1 * math.log(1.6) * 1 * (k1 + 1) / (1 + k1)
    assert flat[0] == pytest.approx(apple + banana, abs=1e-9)


def test_terms_outside_the_vocabulary_are_skipped(index):
    for ranker in RANKERS:
        scorer = Scorer(index, ranker)
        assert scorer.score(["apple", "durian", "banana"]) == pytest.approx(
            scorer.score(QUERY)
        )
        assert scorer.score(["durian"]) == {}
        assert scorer.score([]) == {}
        assert scorer.rank(["durian"]) == []


def test_repeated_query_terms_count_through_qtf(index):
    for ranker in RANKERS:
        scorer = Scorer(index, ranker)
        once = scorer.score(["apple", "banana"])
        twice = scorer.score(["apple", "apple", "banana"])
        # d0 is apple-heavy and d2 banana-heavy, so weighting apple up must
        # move d0 up relative to d2 under every ranker.
        assert twice[0] / twice[2] > once[0] / once[2]


def test_top_n_truncates_without_reordering(index):
    scorer = Scorer(index, "bm25")
    assert scorer.rank(QUERY, top_n=2) == scorer.rank(QUERY)[:2]
    assert len(scorer.rank(QUERY, top_n=1)) == 1


def test_ties_are_broken_by_document_index():
    # A fourth document without the query terms keeps df below N, so the
    # cosine idf stays above zero and the three identical documents tie.
    texts = ["apple banana", "apple banana", "apple banana", "cherry"]
    docs = [
        Doc(docno=f"D{i}", date="1991-05-14", headline="", text=text)
        for i, text in enumerate(texts)
    ]
    index = build_index(docs, None, stem=False)
    for ranker in RANKERS:
        ranked = Scorer(index, ranker).rank(QUERY)
        assert [doc for doc, _ in ranked] == [0, 1, 2]
        assert len({round(s, 12) for _, s in ranked}) == 1


def test_a_term_in_every_document_earns_no_cosine_score():
    # log10(N / df) is zero when df == N, so the course weighting cannot rank
    # a query whose every term is universal. BM25 still can.
    docs = [
        Doc(docno=f"D{i}", date="1991-05-14", headline="", text="apple banana")
        for i in range(3)
    ]
    index = build_index(docs, None, stem=False)
    assert Scorer(index, "tfidf_raw").score(QUERY) == {}
    assert Scorer(index, "tfidf_log").score(QUERY) == {}
    assert Scorer(index, "tfidf_raw").explain(QUERY, 0) == []
    assert len(Scorer(index, "bm25").score(QUERY)) == 3
    assert len(Scorer(index, "bm25").explain(QUERY, 0)) == 2


@pytest.mark.parametrize("ranker", list(RANKERS))
def test_explain_rows_sum_to_the_score(index, ranker):
    scorer = Scorer(index, ranker)
    rows = scorer.explain(QUERY, 0)
    assert [row["term"] for row in rows] == ["apple", "banana"]
    assert [row["tf"] for row in rows] == [2, 1]
    assert [row["qtf"] for row in rows] == [1, 1]
    assert [row["df"] for row in rows] == [2, 2]
    assert all(row["idf"] == pytest.approx(scorer.idf(row["term"])) for row in rows)
    total = sum(row["contribution"] for row in rows)
    assert total == pytest.approx(scorer.score(QUERY)[0], abs=1e-9)


def test_explain_omits_terms_the_document_does_not_have(index):
    rows = Scorer(index, "bm25").explain(["apple", "durian", "cherry"], 0)
    assert [row["term"] for row in rows] == ["apple"]
    assert Scorer(index, "bm25").explain(["cherry"], 0) == []


def test_run_topics_maps_topic_numbers_to_docnos():
    docs = load_corpus(FIXTURE)
    topics = load_topics(FIXTURE / "topics.txt")
    index = build_index(docs, None, stem=False)
    runs = run_topics(index, Scorer(index, "bm25"), topics, "title", None)
    assert set(runs) == {1, 2}
    # Topic 1 is "jet engine"; unstemmed, only FT911-1 matches either word.
    assert [docno for docno, _ in runs[1]] == ["FT911-1"]
    # Topic 2 is "bank money": FT911-2 alone.
    assert [docno for docno, _ in runs[2]] == ["FT911-2"]
    assert all(isinstance(score, float) for _, score in runs[1])


def test_run_topics_analyses_the_query_like_the_index():
    docs = load_corpus(FIXTURE)
    topics = load_topics(FIXTURE / "topics.txt")
    stopwords = frozenset({"about", "any", "documents", "the", "a", "and", "to", "s"})
    stemmed = build_index(docs, stopwords, stem=True)
    runs = run_topics(stemmed, Scorer(stemmed, "bm25"), topics, "title", stopwords)
    # "jet engine" stems to jet/engin, and FT911-10 has "jets" -> "jet".
    assert [docno for docno, _ in runs[1]] == ["FT911-1", "FT911-10"]
    # The description adds "jets" only, which the stemmed index already has.
    long_runs = run_topics(
        stemmed, Scorer(stemmed, "bm25"), topics, "title_desc_narr", stopwords
    )
    assert [d for d, _ in long_runs[1]] == [d for d, _ in runs[1]]


def test_run_topics_truncates_to_top_n():
    docs = load_corpus(FIXTURE)
    topics = load_topics(FIXTURE / "topics.txt")
    index = build_index(docs, None, stem=False)
    runs = run_topics(index, Scorer(index, "bm25"), topics, "title", None, top_n=1)
    assert all(len(v) <= 1 for v in runs.values())
