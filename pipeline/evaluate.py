"""TREC-style metrics and the percentile bootstrap that puts bars on them.

Two conventions from `docs/DECISIONS.md` are baked in here and are worth
naming, because they decide what the numbers mean:

- **D5, unjudged means not relevant.** A document nobody looked at scores
  zero. That is the standard TREC pool convention and it biases every run
  downwards by an unknown amount, which is exactly why `judged@10` is
  reported beside every metric: it says how much of a ranking the judgments
  can even see.
- **S2, AP's denominator is the number of relevant documents in the slice**,
  not the number retrieved. A run that finds one of three relevant documents
  cannot score above 1/3 no matter how it orders them.

Metric functions take a plain `list[str]` of DOCNOs already in rank order and
a `set[str]` of relevant DOCNOs, so they are testable against hand arithmetic
and know nothing about indexes, scorers or qrels files. A topic with no
relevant document in the slice, or a run with no results for a topic, scores
zero for everything rather than raising -- D4 keeps such topics out of the
means, and the zero is what the array holds if one ever gets through.

The bootstrap is a plain percentile bootstrap over *topics* (S1): resample the
71 per-topic values with replacement 10,000 times, take the mean of each
resample, and read the 2.5th and 97.5th percentiles. Every call uses the same
seeded generator, so the interval printed in `runs.json` is the same interval
printed in `comparisons.json` for the same vector of values, and two runs of
the pipeline produce byte-identical files.
"""

from __future__ import annotations

import math

import numpy as np

from pipeline.constants import N_BOOT, SEED

# The 11 standard recall cut-offs. Written as `i / 10` rather than as literals
# so that a recall of `hits / n_rel` which is mathematically equal to a level
# compares equal as a double: IEEE division is correctly rounded, so two
# divisions of equal rationals give the same bits.
RECALL_LEVELS = [i / 10 for i in range(11)]

# Below this, a paired difference is a tie rather than a win or a loss. Two
# runs that agree on a topic can still differ in the last bit or two of a
# float, and calling that a win would be noise.
TIE_EPS = 1e-9

METRICS = ("ap", "p10", "ndcg10", "rprec", "recall100", "judged10")


# ---------------------------------------------------------------- metrics ---


def average_precision(ranked: list[str], rel: set[str]) -> float:
    """Mean of the precision at each rank holding a relevant document.

    Divided by `len(rel)`, so relevant documents the run never retrieved count
    as precision zero (S2).
    """
    if not rel:
        return 0.0
    hits = 0
    total = 0.0
    for rank, docno in enumerate(ranked, start=1):
        if docno in rel:
            hits += 1
            total += hits / rank
    return total / len(rel)


def precision_at(ranked: list[str], rel: set[str], k: int) -> float:
    """Share of the top `k` that is relevant, with `k` as the denominator.

    A ranking shorter than `k` is not given credit for the slots it did not
    fill: P@10 of a five-document run with two hits is 0.2, not 0.4.
    """
    if k <= 0:
        return 0.0
    return sum(1 for docno in ranked[:k] if docno in rel) / k


def recall_at(ranked: list[str], rel: set[str], k: int) -> float:
    """Share of the relevant documents that appear in the top `k`."""
    if not rel or k <= 0:
        return 0.0
    return sum(1 for docno in ranked[:k] if docno in rel) / len(rel)


def r_precision(ranked: list[str], rel: set[str]) -> float:
    """Precision at rank R, where R is the number of relevant documents.

    The one cut-off that adapts to the topic: a topic with two relevant
    documents is scored on its top two, not on a fixed ten mostly made of
    slots it could never fill.
    """
    return precision_at(ranked, rel, len(rel))


def ndcg_at(ranked: list[str], rel: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary gains (S2).

    Gain is 1 for a relevant document and 0 otherwise, discounted by
    `1 / log2(rank + 1)`. The ideal ranking puts `min(k, |rel|)` relevant
    documents first, so a topic with fewer relevant documents than `k` can
    still reach 1.0.
    """
    if not rel or k <= 0:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, docno in enumerate(ranked[:k], start=1)
        if docno in rel
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(rel)) + 1))
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def judged_at(ranked: list[str], judged: set[str], k: int) -> float:
    """Share of the top `k` that any assessor looked at, relevant or not (D5).

    This is a diagnostic, not a quality measure: it says how much of the
    ranking the pool covers, which bounds how much the other metrics can
    know.
    """
    if k <= 0:
        return 0.0
    return sum(1 for docno in ranked[:k] if docno in judged) / k


def interpolated_precision(ranked: list[str], rel: set[str]) -> list[float]:
    """The 11-point interpolated precision curve for one topic (S3).

    At each recall level, the highest precision observed at that recall or
    beyond. Levels the run never reaches are 0.0, which is what makes the
    right-hand end of a mean curve drop: a run that finds half the relevant
    documents contributes nothing above recall 0.5.
    """
    n_rel = len(rel)
    if n_rel == 0 or not ranked:
        return [0.0] * 11
    points: list[tuple[float, float]] = []
    hits = 0
    for rank, docno in enumerate(ranked, start=1):
        if docno in rel:
            hits += 1
            points.append((hits / n_rel, hits / rank))
    if not points:
        return [0.0] * 11
    curve = []
    for level in RECALL_LEVELS:
        best = 0.0
        for recall, precision in points:
            if recall >= level and precision > best:
                best = precision
        curve.append(best)
    return curve


# -------------------------------------------------------------- bootstrap ---


def bootstrap_mean(
    values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, float]:
    """`{"mean", "lo", "hi"}`: the sample mean and a 95% percentile interval.

    `mean` is the real sample mean, not the mean of the resample means, so the
    number on the page is the number the metric actually is; the resamples only
    supply the bounds.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def bootstrap_paired(
    a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, float]:
    """Bootstrap of `a - b` plus the per-topic win/loss/tie counts.

    Paired on topics, so the topic-difficulty variance that dominates a
    per-run interval cancels: two runs can have heavily overlapping MAP
    intervals and still differ on nearly every topic in the same direction.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors differ in shape: {a.shape} vs {b.shape}")
    diff = a - b
    ties = np.abs(diff) < TIE_EPS
    stats = bootstrap_mean(diff, n_boot=n_boot, seed=seed)
    return {
        "mean_diff": stats["mean"],
        "lo": stats["lo"],
        "hi": stats["hi"],
        "wins": int(np.count_nonzero((diff > 0) & ~ties)),
        "losses": int(np.count_nonzero((diff < 0) & ~ties)),
        "ties": int(np.count_nonzero(ties)),
    }


# ------------------------------------------------------------- run level ---


def eval_topics(qrels: dict[int, dict[str, int]]) -> list[int]:
    """Topic numbers with at least one relevant document in the slice (D4).

    Ascending, and the single source of the topic order every per-topic array
    and every JSON file in this project uses.
    """
    return sorted(
        num
        for num, judgments in qrels.items()
        if any(value == 1 for value in judgments.values())
    )


def evaluate_run(
    run: dict[int, list[tuple[str, float]]],
    qrels: dict[int, dict[str, int]],
    topics_eval: list[int],
) -> dict[str, np.ndarray]:
    """Every metric for one run, as arrays aligned to `topics_eval`.

    Returns the six scalar metrics as 1-D arrays plus `interp`, the
    `(topics x 11)` matrix of interpolated precision whose column means are the
    run's PR curve.
    """
    rows: dict[str, list[float]] = {metric: [] for metric in METRICS}
    curves: list[list[float]] = []
    for num in topics_eval:
        judgments = qrels.get(num, {})
        rel = {docno for docno, value in judgments.items() if value == 1}
        judged = set(judgments)
        ranked = [docno for docno, _ in run.get(num, [])]
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"topic {num}: ranking contains duplicate document numbers")
        rows["ap"].append(average_precision(ranked, rel))
        rows["p10"].append(precision_at(ranked, rel, 10))
        rows["ndcg10"].append(ndcg_at(ranked, rel, 10))
        rows["rprec"].append(r_precision(ranked, rel))
        rows["recall100"].append(recall_at(ranked, rel, 100))
        rows["judged10"].append(judged_at(ranked, judged, 10))
        curves.append(interpolated_precision(ranked, rel))
    out: dict[str, np.ndarray] = {
        metric: np.array(values, dtype=float) for metric, values in rows.items()
    }
    out["interp"] = np.array(curves, dtype=float).reshape(len(topics_eval), 11)
    return out
