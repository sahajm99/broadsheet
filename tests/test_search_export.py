"""The browser search index and the golden parity fixtures, on the mini corpus.

Two things are checked here and nowhere else. First, that the gzip the browser
downloads inflates to exactly the shape Task 7's TypeScript reader expects --
key order, gap encoding, and above all that no article TEXT is in it, which is
the licensing rule (D1) and cannot be left to a reviewer's eye. Second, that
the golden fixtures the TypeScript parity test compares against are written in
the form that test will read: string topic numbers, DOCNO and score.
"""

import gzip
import json
from pathlib import Path

import pytest

from pipeline.index import build_all
from pipeline.parse import load_corpus
from pipeline.search_export import (
    SCHEMA,
    build_search_index,
    export_golden,
    export_search_index,
    export_stopwords,
)
from pipeline.topics import load_topics

FIXTURE = Path(__file__).parent / "fixtures" / "mini"
STOPWORDS = frozenset({"the", "and", "a", "to", "s"})


@pytest.fixture(scope="module")
def docs():
    return load_corpus(FIXTURE)


@pytest.fixture(scope="module")
def indexes(docs):
    return build_all(docs, STOPWORDS)


@pytest.fixture(scope="module")
def topics():
    return load_topics(FIXTURE / "topics.txt")


@pytest.fixture(scope="module")
def written(tmp_path_factory, indexes, docs):
    """The gz written once, with the byte sizes the pipeline reports."""
    out = tmp_path_factory.mktemp("search") / "index.json.gz"
    sizes = export_search_index(indexes["stem_stop"], docs, out)
    return out, sizes


@pytest.fixture(scope="module")
def obj(written):
    path, _ = written
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


# ------------------------------------------------------------ the payload ---


def test_gz_round_trips_to_the_schema(obj):
    assert obj["schema"] == SCHEMA == "search_index.v1"
    assert obj["n_docs"] == 7


def test_key_order_is_the_contract(obj):
    assert list(obj) == [
        "schema",
        "n_docs",
        "avgdl",
        "k1",
        "b",
        "docs",
        "doc_len",
        "terms",
    ]


def test_bm25_parameters_travel_with_the_index(obj, indexes):
    assert obj["k1"] == 1.2
    assert obj["b"] == 0.75
    assert obj["avgdl"] == pytest.approx(indexes["stem_stop"].avgdl)


def test_docs_are_citations_in_corpus_order(obj):
    assert obj["docs"][0] == {
        "no": "FT911-1",
        "d": "1991-05-14",
        "h": "Jets over Cranwell",
    }
    assert [row["no"] for row in obj["docs"]] == [
        "FT911-1",
        "FT911-2",
        "FT911-3",
        "FT911-4",
        "FT911-5",
        "FT911-6",
        "FT911-10",
    ]
    assert set(obj["docs"][0]) == {"no", "d", "h"}


def test_doc_len_is_one_number_per_document(obj, indexes):
    assert len(obj["doc_len"]) == 7
    assert obj["doc_len"] == indexes["stem_stop"].doc_len


def test_postings_are_gap_encoded(obj):
    # "jet" is in doc 0 three times (jet, jets, jet's) and doc 6 once; the
    # first gap is the document index itself, later gaps are deltas.
    assert obj["terms"]["jet"] == [0, 3, 6, 1]


def test_every_term_of_the_index_is_present(obj, indexes):
    assert set(obj["terms"]) == set(indexes["stem_stop"].postings)


def test_gaps_reconstruct_the_posting_lists(obj, indexes):
    for term, flat in obj["terms"].items():
        assert len(flat) % 2 == 0
        doc_idx = 0
        rebuilt = []
        for i in range(0, len(flat), 2):
            doc_idx += flat[i]
            rebuilt.append((doc_idx, flat[i + 1]))
        assert rebuilt == indexes["stem_stop"].postings[term]


def test_no_article_text_is_shipped(obj, docs):
    blob = json.dumps(obj)
    # "text" can legitimately be a stem; what may never appear is a field
    # whose value is prose.
    assert '"text":"' not in blob
    assert "text" not in obj
    assert "roared" not in blob  # a word from doc 1's TEXT, not its headline
    for doc in docs:
        assert doc.text not in blob


def test_sizes_are_reported_for_meta_and_postings(written):
    path, sizes = written
    assert set(sizes) == {"index_gz", "index_json"}
    assert sizes["index_gz"] == path.stat().st_size
    assert sizes["index_json"] > sizes["index_gz"] > 0


def test_writing_twice_gives_identical_bytes(tmp_path, indexes, docs):
    first = tmp_path / "a.json.gz"
    second = tmp_path / "b.json.gz"
    export_search_index(indexes["stem_stop"], docs, first)
    export_search_index(indexes["stem_stop"], docs, second)
    assert first.read_bytes() == second.read_bytes()


def test_build_search_index_refuses_a_mismatched_doc_list(indexes, docs):
    with pytest.raises(ValueError, match="document order"):
        build_search_index(indexes["stem_stop"], list(reversed(docs)))


# ----------------------------------------------------------- the fixtures ---


@pytest.fixture(scope="module")
def golden(tmp_path_factory, indexes, docs, topics):
    out = tmp_path_factory.mktemp("golden")
    export_golden(indexes["stem_stop"], docs, topics, STOPWORDS, out)
    return out


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_topic_titles_cover_every_topic(golden, topics):
    titles = load(golden / "topic_titles.json")
    assert len(titles) == len(topics) == 2
    assert titles == {"1": "jet engine", "2": "bank money"}


@pytest.mark.parametrize(
    "name", ["top10_bm25_title.json", "top10_tfidf_raw_title.json"]
)
def test_top10_fixtures_are_docno_and_score(golden, name):
    top10 = load(golden / name)
    assert set(top10) <= {"1", "2"}
    assert "1" in top10
    for rows in top10.values():
        assert 1 <= len(rows) <= 10
        for row in rows:
            assert set(row) == {"no", "score"}
            assert row["no"].startswith("FT911-")
            assert row["score"] == round(row["score"], 6)
        scores = [row["score"] for row in rows]
        assert scores == sorted(scores, reverse=True)


def test_bm25_finds_the_jet_documents(golden):
    top10 = load(golden / "top10_bm25_title.json")
    # "jet engine" -> jet, engin: only doc 1 and doc 10 have either stem.
    assert {row["no"] for row in top10["1"]} == {"FT911-1", "FT911-10"}


def test_fixtures_are_sorted_by_topic_number(golden):
    for name in ("topic_titles.json", "top10_bm25_title.json"):
        text = (golden / name).read_text(encoding="utf-8")
        keys = [int(key) for key in json.loads(text)]
        assert keys == sorted(keys)
        assert "\r" not in text
        assert text.endswith("\n")


# ---------------------------------------------------------- the stopwords ---


def test_stopwords_are_written_sorted(tmp_path):
    path = tmp_path / "stopwords.json"
    export_stopwords(STOPWORDS, path)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == sorted(STOPWORDS)
    assert "\r" not in text
    assert text.endswith("\n")
