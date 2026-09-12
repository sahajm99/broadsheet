"""The whole evaluation matrix, and the seven JSON files the charts read.

`build_runs` scores everything once and hands back one dict; each `write_*`
function is a pure projection of that dict onto one file in the contract. The
split matters because the same per-topic AP vector feeds three different files
-- `runs.json`'s interval, `per_topic.json`'s array and `comparisons.json`'s
paired difference -- and computing it three times would invite three slightly
different answers.

What gets scored:

- **9 main runs**: three rankers x three query fields, all on the `stem_stop`
  index. One `Scorer` per ranker, reused across the three fields, because the
  cosine rankers pay for a full pass over the postings when they are built.
- **4 treatment runs**: BM25 on the topic title over each of the four
  stem/stopword indexes, which is the only clean way to compare them -- the
  same ranker, the same query, a different vocabulary.
- **25 grid cells**: BM25 on the title over `stem_stop` for each (k1, b). The
  index is reused; only the `Scorer` changes, and BM25 scorers are free to
  build. The (1.2, 0.75) cell is the same computation as the headline BM25
  title run and reproduces its MAP exactly, which is the cheapest available
  check that the grid is wired to the same machinery.
- **8 paired comparisons**, each a bootstrap over the per-topic AP difference.

Every float leaves through `round6` and every numpy scalar is converted before
it reaches `json`, which cannot serialise `np.float64`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.constants import (
    BM25_B,
    BM25_K1,
    GRID_B,
    GRID_K1,
    OUT_DIR,
    TREATMENT_LABELS,
    TREATMENTS,
)
from pipeline.evaluate import (
    RECALL_LEVELS,
    bootstrap_mean,
    bootstrap_paired,
    eval_topics,
    evaluate_run,
)
from pipeline.index import Index
from pipeline.io import now_iso, round6, write_json
from pipeline.rank import RANKERS, Scorer, run_topics
from pipeline.topics import FIELDS, Topic

# Rankers outer, fields inner: this is the run order the contract lists, the
# order `runs.json` writes, and the order `best_run` breaks ties in.
RUN_KEYS = [f"{ranker}|{field}" for ranker in RANKERS for field in FIELDS]

# The per-topic metric name -> the name it is written under. AP is the only
# one that changes: the mean of per-topic AP is MAP.
METRIC_NAMES = {
    "ap": "map",
    "p10": "p10",
    "ndcg10": "ndcg10",
    "rprec": "rprec",
    "recall100": "recall100",
    "judged10": "judged10",
}

# (key, a, b, label) with the difference always a - b. A treatment run is
# addressed as "treat:<treatment>"; everything else is a main run key.
COMPARISONS = [
    (
        "bm25_vs_raw",
        "bm25|title",
        "tfidf_raw|title",
        "BM25 vs course tf-idf (title)",
    ),
    (
        "log_vs_raw",
        "tfidf_log|title",
        "tfidf_raw|title",
        "log tf-idf vs course tf-idf (title)",
    ),
    (
        "bm25_vs_log",
        "bm25|title",
        "tfidf_log|title",
        "BM25 vs log tf-idf (title)",
    ),
    (
        "desc_vs_title",
        "bm25|title_desc",
        "bm25|title",
        "BM25: title + description vs title",
    ),
    (
        "narr_vs_title",
        "bm25|title_desc_narr",
        "bm25|title",
        "BM25: title + description + narrative vs title",
    ),
    (
        "raw_narr_vs_title",
        "tfidf_raw|title_desc_narr",
        "tfidf_raw|title",
        "course tf-idf: title + description + narrative vs title",
    ),
    (
        "stem_vs_nostem",
        "treat:stem_stop",
        "treat:nostem_stop",
        "BM25: stemmed vs unstemmed (stopwords removed)",
    ),
    (
        "stop_vs_nostop",
        "treat:stem_stop",
        "treat:stem_nostop",
        "BM25: stopwords removed vs kept (stemmed)",
    ),
]

GRID_CAVEAT = "tuned and scored on the same {n} topics"


# ------------------------------------------------------------- build_runs ---


def _summarise(run, qrels, topics_eval: list[int]) -> dict:
    """One run's per-topic arrays, its bootstrapped means and its curves."""
    per_topic = evaluate_run(run, qrels, topics_eval)
    interp = per_topic["interp"]
    scalars = {name: per_topic[name] for name in METRIC_NAMES}
    return {
        "metrics": {
            written: bootstrap_mean(scalars[name])
            for name, written in METRIC_NAMES.items()
        },
        "per_topic": scalars,
        "interp": interp,
    }


def _counts(topics, qrels, topics_eval: list[int]) -> dict[str, int]:
    """The five headline counts `runs.json` prints beside the metrics.

    `n_topics_all` is every topic in the topic file, not every judged topic:
    the gap between it and `n_topics` is the cost of evaluating a 250-topic
    judgment set against a one-month slice of the collection (D3, D4).
    """
    judged_docs: set[str] = set()
    n_rel_total = 0
    for judgments in qrels.values():
        judged_docs.update(judgments)
        n_rel_total += sum(1 for value in judgments.values() if value == 1)
    n_single = sum(
        1
        for num in topics_eval
        if sum(1 for value in qrels[num].values() if value == 1) == 1
    )
    return {
        "n_topics": len(topics_eval),
        "n_topics_all": len(topics),
        "n_rel_total": n_rel_total,
        "n_judged_docs": len(judged_docs),
        "n_single_rel_topics": n_single,
    }


def _ap(results: dict, key: str) -> np.ndarray:
    """The per-topic AP vector for a main run key or a `treat:` key."""
    if key.startswith("treat:"):
        return results["treatments"][key[len("treat:") :]]["per_topic"]["ap"]
    return results["main"][key]["per_topic"]["ap"]


def build_runs(
    indexes: dict[str, Index],
    topics: list[Topic],
    qrels: dict[int, dict[str, int]],
    stopwords: frozenset[str],
) -> dict:
    """Score every run, treatment, grid cell and comparison, once.

    The returned dict is the single input to all seven writers; nothing below
    re-scores anything.
    """
    stem_stop = indexes["stem_stop"]
    topics_eval = eval_topics(qrels)

    main: dict[str, dict] = {}
    for ranker in RANKERS:
        # One scorer per ranker: the cosine rankers walk every posting list to
        # build their document norms, which costs about a hundred queries.
        scorer = Scorer(stem_stop, ranker)
        for field in FIELDS:
            run = run_topics(stem_stop, scorer, topics, field, stopwords)
            main[f"{ranker}|{field}"] = _summarise(run, qrels, topics_eval)

    treatments: dict[str, dict] = {}
    for key in TREATMENTS:
        index = indexes[key]
        run = run_topics(index, Scorer(index, "bm25"), topics, "title", stopwords)
        treatments[key] = _summarise(run, qrels, topics_eval)

    grid: dict[tuple[float, float], float] = {}
    for k1 in GRID_K1:
        for b in GRID_B:
            scorer = Scorer(stem_stop, "bm25", k1, b)
            run = run_topics(stem_stop, scorer, topics, "title", stopwords)
            ap = evaluate_run(run, qrels, topics_eval)["ap"]
            grid[(k1, b)] = float(ap.mean()) if ap.size else 0.0

    results = {
        "topics_eval": topics_eval,
        "counts": _counts(topics, qrels, topics_eval),
        "main": main,
        "treatments": treatments,
        "grid": grid,
    }
    results["comparisons"] = [
        _compare(results, key, a_key, b_key, label, topics_eval)
        for key, a_key, b_key, label in COMPARISONS
    ]
    return results


def _compare(
    results: dict,
    key: str,
    a_key: str,
    b_key: str,
    label: str,
    topics_eval: list[int],
) -> dict:
    """One paired comparison row, per-topic differences included."""
    a = _ap(results, a_key)
    b = _ap(results, b_key)
    stats = bootstrap_paired(a, b)
    return {
        "key": key,
        "a": a_key,
        "b": b_key,
        "label": label,
        "mean_diff": stats["mean_diff"],
        "lo": stats["lo"],
        "hi": stats["hi"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "ties": stats["ties"],
        "per_topic": [
            {"num": int(num), "diff": float(diff)}
            for num, diff in zip(topics_eval, a - b)
        ],
    }


# ---------------------------------------------------------------- writers ---


def _interval(stats: dict[str, float]) -> dict[str, float]:
    return {
        "mean": round6(stats["mean"]),
        "lo": round6(stats["lo"]),
        "hi": round6(stats["hi"]),
    }


def write_runs(results: dict, out_dir: Path = OUT_DIR) -> Path:
    """`runs.json`: the nine runs with a bootstrap interval on every metric."""
    obj = {
        "schema": "runs.v1",
        "generated_at": now_iso(),
        **results["counts"],
        "rankers": [{"key": key, "label": label} for key, label in RANKERS.items()],
        "fields": [{"key": key, "label": label} for key, label in FIELDS.items()],
        "runs": [
            {
                "key": key,
                "ranker": key.split("|")[0],
                "field": key.split("|")[1],
                "metrics": {
                    name: _interval(stats)
                    for name, stats in results["main"][key]["metrics"].items()
                },
            }
            for key in RUN_KEYS
        ],
    }
    path = Path(out_dir) / "runs.json"
    write_json(path, obj)
    return path


def write_per_topic(results: dict, out_dir: Path = OUT_DIR) -> Path:
    """`per_topic.json`: every run's AP for every evaluable topic.

    This is what lets the site draw the distribution behind a MAP instead of
    only its interval -- a mean of 0.2 made of forty zeros and a few 0.9s is a
    different engine from one made of seventy 0.2s.
    """
    obj = {
        "schema": "per_topic.v1",
        "generated_at": now_iso(),
        "topics": [int(num) for num in results["topics_eval"]],
        "ap": {
            key: [round6(value) for value in results["main"][key]["per_topic"]["ap"]]
            for key in RUN_KEYS
        },
    }
    path = Path(out_dir) / "per_topic.json"
    write_json(path, obj)
    return path


def write_comparisons(results: dict, out_dir: Path = OUT_DIR) -> Path:
    """`comparisons.json`: the eight paired differences, topic by topic."""
    obj = {
        "schema": "comparisons.v1",
        "generated_at": now_iso(),
        "pairs": [
            {
                "key": pair["key"],
                "a": pair["a"],
                "b": pair["b"],
                "label": pair["label"],
                "mean_diff": round6(pair["mean_diff"]),
                "lo": round6(pair["lo"]),
                "hi": round6(pair["hi"]),
                "wins": int(pair["wins"]),
                "losses": int(pair["losses"]),
                "ties": int(pair["ties"]),
                "per_topic": [
                    {"num": int(row["num"]), "diff": round6(row["diff"])}
                    for row in pair["per_topic"]
                ],
            }
            for pair in results["comparisons"]
        ],
    }
    path = Path(out_dir) / "comparisons.json"
    write_json(path, obj)
    return path


def write_pr_curves(results: dict, out_dir: Path = OUT_DIR) -> Path:
    """`pr_curves.json`: mean interpolated precision at 11 recall levels (S3)."""
    obj = {
        "schema": "pr_curves.v1",
        "generated_at": now_iso(),
        "recall_levels": [round6(level) for level in RECALL_LEVELS],
        "curves": {
            key: [round6(value) for value in results["main"][key]["interp"].mean(axis=0)]
            for key in RUN_KEYS
        },
    }
    path = Path(out_dir) / "pr_curves.json"
    write_json(path, obj)
    return path


def write_topics(
    results: dict,
    topics: list[Topic],
    qrels: dict[int, dict[str, int]],
    out_dir: Path = OUT_DIR,
) -> Path:
    """`topics.json`: one row per evaluable topic, with each run's AP.

    `best_run` is the run with the highest AP on that topic, first in run order
    on a tie -- which is how a table of zeros still names a run rather than
    leaving the column blank.
    """
    titles = {topic.num: topic.title for topic in topics}
    per_run = {key: results["main"][key]["per_topic"]["ap"] for key in RUN_KEYS}
    rows = []
    for position, num in enumerate(results["topics_eval"]):
        judgments = qrels.get(num, {})
        ap = {key: float(per_run[key][position]) for key in RUN_KEYS}
        best_run = RUN_KEYS[0]
        for key in RUN_KEYS:
            if ap[key] > ap[best_run]:
                best_run = key
        rows.append(
            {
                "num": int(num),
                "title": titles.get(num, ""),
                "n_rel": sum(1 for value in judgments.values() if value == 1),
                "n_judged": len(judgments),
                "ap": {key: round6(value) for key, value in ap.items()},
                "best_run": best_run,
            }
        )
    obj = {"schema": "topics.v1", "generated_at": now_iso(), "rows": rows}
    path = Path(out_dir) / "topics.json"
    write_json(path, obj)
    return path


def write_grid(results: dict, out_dir: Path = OUT_DIR) -> Path:
    """`bm25_grid.json`: MAP over the (k1, b) surface, with the caveat (R3).

    The caveat is part of the file, not of the chart's copy, so it cannot be
    dropped by a later redesign: the best cell was chosen on the same 71 topics
    it is scored on and is therefore an upper bound, not a result.
    """
    grid = results["grid"]
    matrix = [[grid[(k1, b)] for b in GRID_B] for k1 in GRID_K1]
    best_k1, best_b = max(
        ((k1, b) for k1 in GRID_K1 for b in GRID_B),
        key=lambda cell: grid[cell],
    )
    obj = {
        "schema": "bm25_grid.v1",
        "generated_at": now_iso(),
        "k1": list(GRID_K1),
        "b": list(GRID_B),
        "map": [[round6(value) for value in row] for row in matrix],
        "best": {
            "k1": best_k1,
            "b": best_b,
            "map": round6(grid[(best_k1, best_b)]),
        },
        "default": {
            "k1": BM25_K1,
            "b": BM25_B,
            "map": round6(grid[(BM25_K1, BM25_B)]),
        },
        "caveat": GRID_CAVEAT.format(n=results["counts"]["n_topics"]),
    }
    path = Path(out_dir) / "bm25_grid.json"
    write_json(path, obj)
    return path


def write_treatments(
    results: dict, indexes: dict[str, Index], out_dir: Path = OUT_DIR
) -> Path:
    """`treatments.json`: what stemming and stopping cost in retrieval terms.

    `n_types` comes from the index rather than from the run, so the row ties
    the vocabulary size on the vocabulary chart to the MAP beside it.
    """
    obj = {
        "schema": "treatments.v1",
        "generated_at": now_iso(),
        "ranker": "bm25",
        "field": "title",
        "rows": [
            {
                "key": key,
                "label": TREATMENT_LABELS[key],
                "n_types": len(indexes[key].postings),
                "map": _interval(results["treatments"][key]["metrics"]["map"]),
                "p10": _interval(results["treatments"][key]["metrics"]["p10"]),
            }
            for key in TREATMENTS
        ],
    }
    path = Path(out_dir) / "treatments.json"
    write_json(path, obj)
    return path
