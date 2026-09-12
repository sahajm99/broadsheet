"""The whole evaluation matrix and the seven JSON writers, on the mini corpus.

The mini fixture has 7 documents, 2 topics and 4 judgments, which is far too
small for the numbers to mean anything -- that is deliberate. What is checked
here is shape: that the 9 runs, 4 treatments, 25 grid cells and 8 comparisons
are all present and consistent with each other, that the per-topic arrays line
up with `topics_eval`, and that every file the site reads parses and carries
its schema string. The real numbers are produced once, on the real corpus, and
recorded in the task report.
"""

import json
from pathlib import Path

import pytest

from pipeline.constants import BM25_B, BM25_K1, GRID_B, GRID_K1, TREATMENTS
from pipeline.index import build_all
from pipeline.parse import load_corpus
from pipeline.rank import RANKERS
from pipeline.runs import (
    COMPARISONS,
    RUN_KEYS,
    build_runs,
    write_comparisons,
    write_grid,
    write_per_topic,
    write_pr_curves,
    write_runs,
    write_topics,
    write_treatments,
)
from pipeline.topics import FIELDS, load_qrels, load_topics

FIXTURE = Path(__file__).parent / "fixtures" / "mini"
STOPWORDS = frozenset({"the", "and", "a", "to", "s"})


@pytest.fixture(scope="module")
def indexes():
    return build_all(load_corpus(FIXTURE), STOPWORDS)


@pytest.fixture(scope="module")
def topics():
    return load_topics(FIXTURE / "topics.txt")


@pytest.fixture(scope="module")
def qrels(indexes):
    return load_qrels(FIXTURE / "qrels.txt", set(indexes["stem_stop"].docnos))


@pytest.fixture(scope="module")
def results(indexes, topics, qrels):
    return build_runs(indexes, topics, qrels, STOPWORDS)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------- build_runs ---


def test_run_keys_are_rankers_crossed_with_fields():
    assert RUN_KEYS == [f"{r}|{f}" for r in RANKERS for f in FIELDS]
    assert len(RUN_KEYS) == 9
    assert RUN_KEYS[0] == "tfidf_raw|title"
    assert RUN_KEYS[-1] == "bm25|title_desc_narr"


def test_nine_main_runs_each_with_metrics_and_per_topic(results):
    assert list(results["main"]) == RUN_KEYS
    for key, run in results["main"].items():
        for metric in ("map", "p10", "ndcg10", "rprec", "recall100", "judged10"):
            assert set(run["metrics"][metric]) == {"mean", "lo", "hi"}, key
        assert run["per_topic"]["ap"].shape == (2,)
        assert run["interp"].shape == (2, 11)


def test_topics_eval_and_counts(results):
    assert results["topics_eval"] == [1, 2]
    assert results["counts"] == {
        "n_topics": 2,
        "n_topics_all": 2,
        "n_rel_total": 3,
        "n_judged_docs": 3,
        "n_single_rel_topics": 1,
    }


def test_four_treatment_runs(results):
    assert list(results["treatments"]) == list(TREATMENTS)
    for run in results["treatments"].values():
        assert set(run["metrics"]["map"]) == {"mean", "lo", "hi"}
        assert run["per_topic"]["ap"].shape == (2,)


def test_grid_is_five_by_five_and_agrees_with_the_default_run(results):
    assert len(results["grid"]) == 25
    assert set(results["grid"]) == {(k1, b) for k1 in GRID_K1 for b in GRID_B}
    # The default cell is the same computation as the headline BM25 title run,
    # so it must reproduce its MAP bit for bit.
    assert results["grid"][(BM25_K1, BM25_B)] == results["main"]["bm25|title"][
        "metrics"
    ]["map"]["mean"]


def test_eight_comparisons_cover_every_topic(results):
    assert [pair["key"] for pair in results["comparisons"]] == [
        row[0] for row in COMPARISONS
    ]
    for pair in results["comparisons"]:
        assert pair["wins"] + pair["losses"] + pair["ties"] == 2
        assert pair["lo"] <= pair["mean_diff"] <= pair["hi"]
        assert [row["num"] for row in pair["per_topic"]] == [1, 2]
        assert pair["label"]


def test_comparison_diffs_match_the_per_topic_arrays(results):
    pair = results["comparisons"][0]
    assert pair["a"] == "bm25|title" and pair["b"] == "tfidf_raw|title"
    a = results["main"]["bm25|title"]["per_topic"]["ap"]
    b = results["main"]["tfidf_raw|title"]["per_topic"]["ap"]
    assert [row["diff"] for row in pair["per_topic"]] == pytest.approx(list(a - b))


def test_treatment_comparisons_read_the_treatment_runs(results):
    pair = next(p for p in results["comparisons"] if p["key"] == "stem_vs_nostem")
    assert pair["a"] == "treat:stem_stop" and pair["b"] == "treat:nostem_stop"
    a = results["treatments"]["stem_stop"]["per_topic"]["ap"]
    b = results["treatments"]["nostem_stop"]["per_topic"]["ap"]
    assert pair["mean_diff"] == pytest.approx(float((a - b).mean()))


# ---------------------------------------------------------------- writers ---


def test_write_runs(results, tmp_path):
    obj = load(write_runs(results, tmp_path))
    assert obj["schema"] == "runs.v1"
    assert obj["n_topics"] == 2 and obj["n_topics_all"] == 2
    assert obj["n_rel_total"] == 3 and obj["n_judged_docs"] == 3
    assert obj["n_single_rel_topics"] == 1
    assert [r["key"] for r in obj["rankers"]] == list(RANKERS)
    assert [f["key"] for f in obj["fields"]] == list(FIELDS)
    assert [run["key"] for run in obj["runs"]] == RUN_KEYS
    first = obj["runs"][0]
    assert first["ranker"] == "tfidf_raw" and first["field"] == "title"
    assert set(first["metrics"]) == {
        "map",
        "p10",
        "ndcg10",
        "rprec",
        "recall100",
        "judged10",
    }
    assert first["metrics"]["map"]["mean"] == pytest.approx(
        results["main"]["tfidf_raw|title"]["metrics"]["map"]["mean"], abs=5e-7
    )


def test_write_per_topic(results, tmp_path):
    obj = load(write_per_topic(results, tmp_path))
    assert obj["schema"] == "per_topic.v1"
    assert obj["topics"] == [1, 2]
    assert list(obj["ap"]) == RUN_KEYS
    for values in obj["ap"].values():
        assert len(values) == 2


def test_write_comparisons(results, tmp_path):
    obj = load(write_comparisons(results, tmp_path))
    assert obj["schema"] == "comparisons.v1"
    assert len(obj["pairs"]) == 8
    for pair in obj["pairs"]:
        assert set(pair) == {
            "key",
            "a",
            "b",
            "label",
            "mean_diff",
            "lo",
            "hi",
            "wins",
            "losses",
            "ties",
            "per_topic",
        }
        assert pair["wins"] + pair["losses"] + pair["ties"] == 2
        assert [row["num"] for row in pair["per_topic"]] == [1, 2]


def test_write_pr_curves(results, tmp_path):
    obj = load(write_pr_curves(results, tmp_path))
    assert obj["schema"] == "pr_curves.v1"
    assert obj["recall_levels"] == [round(i / 10, 6) for i in range(11)]
    assert list(obj["curves"]) == RUN_KEYS
    for curve in obj["curves"].values():
        assert len(curve) == 11
        assert all(0.0 <= point <= 1.0 for point in curve)


def test_write_topics(results, topics, qrels, tmp_path):
    obj = load(write_topics(results, topics, qrels, tmp_path))
    assert obj["schema"] == "topics.v1"
    assert [row["num"] for row in obj["rows"]] == [1, 2]
    row = obj["rows"][0]
    assert row["title"] == "jet engine"
    assert row["n_rel"] == 2 and row["n_judged"] == 3
    assert list(row["ap"]) == RUN_KEYS
    assert row["best_run"] in RUN_KEYS
    best = max(row["ap"][key] for key in RUN_KEYS)
    assert row["ap"][row["best_run"]] == best
    # Ties go to the first run in run order.
    assert row["best_run"] == next(k for k in RUN_KEYS if row["ap"][k] == best)


def test_write_grid(results, tmp_path):
    obj = load(write_grid(results, tmp_path))
    assert obj["schema"] == "bm25_grid.v1"
    assert obj["k1"] == GRID_K1 and obj["b"] == GRID_B
    assert len(obj["map"]) == 5 and all(len(row) == 5 for row in obj["map"])
    # Rows are k1, columns are b.
    assert obj["default"] == {
        "k1": BM25_K1,
        "b": BM25_B,
        "map": obj["map"][GRID_K1.index(BM25_K1)][GRID_B.index(BM25_B)],
    }
    assert obj["best"]["map"] == max(max(row) for row in obj["map"])
    assert obj["best"]["k1"] in GRID_K1 and obj["best"]["b"] in GRID_B
    assert "same 2 topics" in obj["caveat"]


def test_write_treatments(results, indexes, tmp_path):
    obj = load(write_treatments(results, indexes, tmp_path))
    assert obj["schema"] == "treatments.v1"
    assert obj["ranker"] == "bm25" and obj["field"] == "title"
    assert [row["key"] for row in obj["rows"]] == list(TREATMENTS)
    for row in obj["rows"]:
        assert row["label"]
        assert row["n_types"] == len(indexes[row["key"]].postings)
        assert set(row["map"]) == {"mean", "lo", "hi"}
        assert set(row["p10"]) == {"mean", "lo", "hi"}
