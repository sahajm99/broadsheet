"""Vocabulary statistics, the log-log fits, and the three JSON writers."""

import gzip
import json
import re
from pathlib import Path

import pytest

from pipeline.constants import TREATMENTS
from pipeline.index import Index, build_all, build_index
from pipeline.io import now_iso, round6, write_gz, write_json
from pipeline.parse import load_corpus
from pipeline.vocab_stats import (
    DOC_LEN_BINS,
    POSTING_BINS,
    doc_len_hist,
    doc_len_summary,
    funnel,
    heaps_fit,
    heaps_points,
    longest,
    posting_hist,
    posting_summary,
    stem_groups,
    top_terms,
    write_corpus,
    write_postings,
    write_vocab,
    zipf_fit,
    zipf_points,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini"
STOPWORDS = frozenset({"the", "and", "a", "to", "s"})


@pytest.fixture(scope="module")
def docs():
    return load_corpus(FIXTURE)


@pytest.fixture(scope="module")
def indexes(docs):
    return build_all(docs, STOPWORDS)


@pytest.fixture(scope="module")
def index(indexes):
    return indexes["stem_stop"]


# --------------------------------------------------------------------- io ---


def test_write_json_shape_and_size_guard(tmp_path):
    path = tmp_path / "nested" / "out.json"
    size = write_json(path, {"schema": "x.v1", "n": 1})
    assert path.exists()
    raw = path.read_bytes()
    assert size == len(raw)
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8")) == {"schema": "x.v1", "n": 1}
    with pytest.raises(ValueError) as excinfo:
        write_json(tmp_path / "big.json", {"a": ["x" * 100] * 100}, limit=50)
    assert "big.json" in str(excinfo.value)


def test_write_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", {"x": float("nan")})


def test_write_gz_is_deterministic(tmp_path):
    obj = {"schema": "search_index.v1", "terms": {"jet": [0, 3, 6, 1]}}
    a = tmp_path / "a.gz"
    b = tmp_path / "b.gz"
    assert write_gz(a, obj) == write_gz(b, obj)
    assert a.read_bytes() == b.read_bytes()
    assert json.loads(gzip.decompress(a.read_bytes()).decode("utf-8")) == obj


def test_round6_and_now_iso():
    assert round6(1 / 3) == 0.333333
    assert round6(2) == 2.0
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", now_iso())


# ----------------------------------------------------------------- funnel ---


def test_funnel_is_four_ordered_shrinking_steps(docs):
    rows = funnel(docs, STOPWORDS)
    assert [r["step"] for r in rows] == [
        "raw_alpha",
        "minus_digits",
        "minus_stopwords",
        "stemmed",
    ]
    assert all(r["label"] for r in rows)
    types = [r["n_types"] for r in rows]
    tokens = [r["n_tokens"] for r in rows]
    assert types == sorted(types, reverse=True)
    assert tokens[0] >= tokens[1] >= tokens[2] == tokens[3]
    raw = [t for d in docs for t in re.findall(r"[a-z0-9]+", d.text.lower())]
    assert rows[0]["n_tokens"] == len(raw)
    assert rows[0]["n_types"] == len(set(raw))
    assert rows[1]["n_tokens"] == len([t for t in raw if not re.search(r"\d", t)])


def test_funnel_matches_the_indexes(docs, indexes):
    rows = {r["step"]: r for r in funnel(docs, STOPWORDS)}
    assert rows["minus_digits"]["n_tokens"] == indexes["nostem_nostop"].total_tokens
    assert rows["minus_stopwords"]["n_tokens"] == indexes["nostem_stop"].total_tokens
    assert rows["stemmed"]["n_types"] == len(indexes["stem_stop"].postings)


# ------------------------------------------------------------------ heaps ---


def test_heaps_points_increase_and_end_at_the_total(docs, indexes):
    points = heaps_points(docs, STOPWORDS)
    assert points
    assert [p["n_tokens"] for p in points] == sorted({p["n_tokens"] for p in points})
    assert [p["n_types"] for p in points] == sorted(p["n_types"] for p in points)
    assert points[-1]["n_tokens"] == indexes["nostem_stop"].total_tokens
    assert points[-1]["n_types"] == len(indexes["nostem_stop"].postings)


def test_heaps_fit_recovers_a_synthetic_power_law():
    points = [
        {"n_tokens": n, "n_types": 5.0 * n**0.6}
        for n in (1000, 3000, 10_000, 30_000, 100_000, 1_000_000)
    ]
    fit = heaps_fit(points)
    assert fit["beta"] == pytest.approx(0.6, abs=1e-6)
    assert fit["k"] == pytest.approx(5.0, rel=1e-6)


# ------------------------------------------------------------------- zipf ---


def _power_law_index(n_terms: int) -> Index:
    """An index whose collection frequencies are exactly ``1000 / rank``."""
    postings = {f"t{r:04d}": [(0, 1000.0 / r)] for r in range(1, n_terms + 1)}
    return Index(
        docnos=["FT911-1"],
        doc_len=[1],
        postings=postings,
        n_docs=1,
        avgdl=1.0,
        total_tokens=1,
        stem=True,
        stop=True,
    )


def test_zipf_fit_recovers_slope_minus_one():
    fit = zipf_fit(_power_law_index(1000))
    assert fit["slope"] == pytest.approx(-1.0, abs=1e-6)
    assert fit["intercept"] == pytest.approx(3.0, abs=1e-6)


def test_zipf_points_are_ranked_by_frequency(index):
    points = zipf_points(index)
    assert points[0]["rank"] == 1
    ranks = [p["rank"] for p in points]
    assert ranks == sorted(set(ranks))
    assert ranks[-1] <= len(index.postings)
    freqs = [p["freq"] for p in points]
    assert freqs == sorted(freqs, reverse=True)
    assert all(p["term"] in index.postings for p in points)


# -------------------------------------------------------------- term lists ---


def test_top_terms_are_ordered_by_collection_frequency(index):
    rows = top_terms(index, n=5)
    assert len(rows) == 5
    assert [r["cf"] for r in rows] == sorted((r["cf"] for r in rows), reverse=True)
    for row in rows:
        assert row["df"] == index.df(row["term"])
        assert row["cf"] == sum(tf for _, tf in index.postings[row["term"]])
        assert row["cf"] >= row["df"]


def test_stem_groups_collect_surface_forms(docs, index):
    groups = stem_groups(docs, STOPWORDS, index)
    assert groups
    assert [g["n_words"] for g in groups] == sorted(
        (g["n_words"] for g in groups), reverse=True
    )
    for group in groups:
        assert len(group["words"]) == min(group["n_words"], 8)
        assert group["stem"] in index.postings
        assert [w["cf"] for w in group["words"]] == sorted(
            (w["cf"] for w in group["words"]), reverse=True
        )
    # doc3 has "exploration", "explorers" and "explore", all stemming to "explor".
    by_stem = {g["stem"]: g for g in groups}
    assert by_stem["explor"]["n_words"] == 3
    assert {w["word"] for w in by_stem["explor"]["words"]} == {
        "exploration",
        "explorers",
        "explore",
    }


def test_longest_lists_the_widest_posting_lists(index):
    rows = longest(index, n=3)
    assert len(rows) == 3
    assert [r["df"] for r in rows] == sorted((r["df"] for r in rows), reverse=True)
    assert rows[0]["df"] == max(index.df(t) for t in index.postings)


# ------------------------------------------------------------- histograms ---


def test_posting_hist_counts_every_term(index):
    hist = posting_hist(index)
    assert hist["bins"] == POSTING_BINS
    assert len(hist["counts"]) == len(POSTING_BINS)
    assert sum(hist["counts"]) == len(index.postings)


def test_posting_summary(index):
    summary = posting_summary(index)
    assert summary["n_terms"] == len(index.postings)
    assert summary["n_postings"] == sum(len(p) for p in index.postings.values())
    assert summary["max_len"] == max(len(p) for p in index.postings.values())
    assert summary["mean_len"] == pytest.approx(
        summary["n_postings"] / summary["n_terms"], abs=1e-6
    )
    assert 0.0 <= summary["df1_share"] <= 1.0


def test_doc_len_hist_counts_every_document(index):
    hist = doc_len_hist(index)
    assert hist["bins"] == DOC_LEN_BINS
    assert len(hist["counts"]) == len(DOC_LEN_BINS)
    assert sum(hist["counts"]) == index.n_docs


def test_doc_len_hist_last_bin_catches_the_tail():
    base = build_index(load_corpus(FIXTURE), None, stem=False)
    huge = Index(
        docnos=["a", "b"],
        doc_len=[4, 9_999],
        postings=base.postings,
        n_docs=2,
        avgdl=5001.5,
        total_tokens=10_003,
        stem=False,
        stop=False,
    )
    counts = doc_len_hist(huge)["counts"]
    assert counts[0] == 1
    assert counts[-1] == 1
    assert sum(counts) == 2


def test_doc_len_summary(index):
    summary = doc_len_summary(index)
    assert summary["max"] == max(index.doc_len)
    assert summary["mean"] == pytest.approx(index.avgdl, abs=1e-6)
    assert summary["p10"] <= summary["median"] <= summary["p90"] <= summary["max"]


# ---------------------------------------------------------------- writers ---


def test_write_corpus(docs, index, tmp_path):
    path = write_corpus(docs, index, n_files=2, out_dir=tmp_path)
    assert path == tmp_path / "corpus.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["schema"] == "corpus.v1"
    assert obj["n_docs"] == 7
    assert obj["n_files"] == 2
    assert obj["date_min"] == "1991-05-14"
    assert obj["date_max"] == "1991-05-15"
    assert obj["total_tokens_alpha"] > 0
    assert obj["doc_len"]["bins"] == DOC_LEN_BINS
    assert sum(obj["doc_len"]["counts"]) == 7
    assert set(obj["doc_len_summary"]) == {"mean", "median", "p10", "p90", "max"}
    assert obj["generated_at"].endswith("Z")


def test_write_vocab(docs, indexes, tmp_path):
    path = write_vocab(docs, STOPWORDS, indexes, out_dir=tmp_path)
    assert path == tmp_path / "vocab.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["schema"] == "vocab.v1"
    assert [r["step"] for r in obj["funnel"]] == [
        "raw_alpha",
        "minus_digits",
        "minus_stopwords",
        "stemmed",
    ]
    assert set(obj["treatments"]) == set(TREATMENTS)
    for key in TREATMENTS:
        assert set(obj["treatments"][key]) == {"n_types", "n_tokens"}
    assert all(set(r) == {"term", "df", "cf"} for r in obj["top_terms"])
    assert set(obj["heaps_fit"]) == {"k", "beta"}
    assert set(obj["zipf_fit"]) == {"slope", "intercept"}
    assert all(set(p) == {"n_tokens", "n_types"} for p in obj["heaps"])
    assert all(set(p) == {"rank", "freq", "term"} for p in obj["zipf"])
    assert all(set(g) == {"stem", "n_words", "words"} for g in obj["stem_groups"])
    assert len(obj["stem_groups"]) <= 12


def test_write_postings(index, tmp_path):
    path = write_postings(index, {"index_gz": 10, "index_json": 20}, out_dir=tmp_path)
    assert path == tmp_path / "postings.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["schema"] == "postings.v1"
    assert obj["hist"]["bins"] == POSTING_BINS
    assert set(obj["summary"]) == {
        "n_terms",
        "n_postings",
        "mean_len",
        "median_len",
        "max_len",
        "df1_share",
    }
    assert len(obj["longest"]) <= 20
    assert obj["size_bytes"] == {"index_gz": 10, "index_json": 20}


def test_writers_round_floats_to_six_places(docs, indexes, tmp_path):
    write_vocab(docs, STOPWORDS, indexes, out_dir=tmp_path)
    text = (tmp_path / "vocab.json").read_text(encoding="utf-8")
    for decimals in re.findall(r"-?\d+\.(\d+)", text):
        assert len(decimals) <= 6, decimals
