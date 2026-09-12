"""Metrics, the percentile bootstrap and per-run evaluation.

Every expected number in the hand cases is computed on paper in the task brief
and written here as its closed form *and* as the brief's six-decimal figure, so
a test cannot be quietly fitted to whatever the implementation happens to
return.

The hand case throughout is `ranked = [a, b, c, d, e]` with `rel = {a, c, e}`:
relevant documents at ranks 1, 3 and 5.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pipeline.constants import N_BOOT, SEED
from pipeline.evaluate import (
    RECALL_LEVELS,
    average_precision,
    bootstrap_mean,
    bootstrap_paired,
    eval_topics,
    evaluate_run,
    interpolated_precision,
    judged_at,
    ndcg_at,
    precision_at,
    r_precision,
    recall_at,
)
from pipeline.topics import load_qrels

FIXTURE = Path(__file__).parent / "fixtures" / "mini"

RANKED = ["a", "b", "c", "d", "e"]
REL = {"a", "c", "e"}
JUDGED = {"a", "b", "z"}


# ---------------------------------------------------------------- metrics ---


def test_average_precision_hand_case():
    # (1/1 + 2/3 + 3/5) / 3
    assert average_precision(RANKED, REL) == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    assert average_precision(RANKED, REL) == pytest.approx(0.755556, abs=5e-7)


def test_average_precision_divides_by_relevant_in_the_slice():
    # Only one of the three relevant documents is retrieved: AP is (1/1) / 3,
    # not 1.0 -- the denominator is |rel|, not the number of hits (S2).
    assert average_precision(["a"], REL) == pytest.approx(1 / 3)


def test_precision_at_k():
    assert precision_at(RANKED, REL, 2) == pytest.approx(0.5)
    assert precision_at(RANKED, REL, 5) == pytest.approx(0.6)
    # k is the denominator even when the ranking is shorter than k.
    assert precision_at(["a"], REL, 10) == pytest.approx(0.1)
    assert precision_at(RANKED, REL, 0) == 0.0


def test_recall_at_k():
    assert recall_at(RANKED, REL, 2) == pytest.approx(1 / 3)
    assert recall_at(RANKED, REL, 5) == pytest.approx(1.0)
    assert recall_at(RANKED, REL, 100) == pytest.approx(1.0)


def test_r_precision_is_precision_at_the_number_relevant():
    # |rel| = 3, two of the first three are relevant.
    assert r_precision(RANKED, REL) == pytest.approx(2 / 3)


def test_ndcg_hand_case():
    dcg = 1 + 1 / math.log2(4) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert dcg == pytest.approx(1.886853, abs=5e-7)
    assert idcg == pytest.approx(2.130930, abs=5e-7)
    assert ndcg_at(RANKED, REL, 5) == pytest.approx(dcg / idcg)
    # The brief quotes 0.885458; the exact ratio is 0.8854599, which rounds to
    # 0.885460. The brief's figure is its rounded intermediates divided, so the
    # closed form above is the assertion that matters and this one is loose.
    assert ndcg_at(RANKED, REL, 5) == pytest.approx(0.885458, abs=5e-6)


def test_ndcg_ideal_is_capped_at_k():
    # Only rank 1 is inside k=1 and the ideal is a single one, so a relevant
    # document at rank 1 is a perfect nDCG@1 even with |rel| = 3.
    assert ndcg_at(RANKED, REL, 1) == pytest.approx(1.0)


def test_judged_at_counts_any_judgment_not_just_relevant():
    # {a, b} of the top five were judged; z was judged but never retrieved.
    assert judged_at(RANKED, JUDGED, 5) == pytest.approx(0.4)
    assert judged_at(RANKED, JUDGED, 2) == pytest.approx(1.0)


def test_interpolated_precision_hand_case():
    expected = [1.0, 1.0, 1.0, 1.0, 2 / 3, 2 / 3, 2 / 3, 0.6, 0.6, 0.6, 0.6]
    got = interpolated_precision(RANKED, REL)
    assert len(got) == 11
    assert got == pytest.approx(expected)


def test_interpolated_precision_is_zero_past_the_recall_reached():
    # Only the first relevant document is retrieved, so recall never exceeds
    # 1/3 and every level above 0.4 has no point to take a maximum over.
    got = interpolated_precision(["a", "x", "y"], REL)
    assert got[:4] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert got[4:] == pytest.approx([0.0] * 7)


def test_recall_levels_are_the_eleven_contract_points():
    assert RECALL_LEVELS == pytest.approx([i / 10 for i in range(11)])


def test_no_relevant_documents_scores_zero_everywhere():
    empty: set[str] = set()
    assert average_precision(RANKED, empty) == 0.0
    assert ndcg_at(RANKED, empty, 10) == 0.0
    assert r_precision(RANKED, empty) == 0.0
    assert recall_at(RANKED, empty, 10) == 0.0
    assert interpolated_precision(RANKED, empty) == [0.0] * 11


def test_empty_ranking_scores_zero_everywhere():
    assert average_precision([], REL) == 0.0
    assert precision_at([], REL, 10) == 0.0
    assert recall_at([], REL, 10) == 0.0
    assert r_precision([], REL) == 0.0
    assert ndcg_at([], REL, 10) == 0.0
    assert judged_at([], JUDGED, 10) == 0.0
    assert interpolated_precision([], REL) == [0.0] * 11


# -------------------------------------------------------------- bootstrap ---


def test_bootstrap_mean_brackets_the_sample_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    out = bootstrap_mean(values)
    assert out["mean"] == pytest.approx(2.5)
    assert out["lo"] >= 1.0
    assert out["hi"] <= 4.0
    assert out["lo"] <= out["mean"] <= out["hi"]


def test_bootstrap_mean_is_deterministic():
    values = np.random.default_rng(1).random(71)
    assert bootstrap_mean(values) == bootstrap_mean(values)
    # A different seed moves the interval, which is what makes pinning the
    # seed worth doing. (With only a handful of coarse values the percentiles
    # can coincide across seeds, so this uses a 71-topic-shaped vector.)
    assert bootstrap_mean(values, seed=SEED + 1) != bootstrap_mean(values)


def test_bootstrap_mean_uses_n_boot_resamples_of_size_n():
    values = np.array([0.0, 1.0])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 2, size=(N_BOOT, 2))
    expected = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    out = bootstrap_mean(values)
    assert out["lo"] == pytest.approx(expected[0])
    assert out["hi"] == pytest.approx(expected[1])


def test_bootstrap_mean_of_nothing_is_zero():
    assert bootstrap_mean(np.array([])) == {"mean": 0.0, "lo": 0.0, "hi": 0.0}


def test_bootstrap_paired_identical_runs_are_all_ties():
    a = np.array([0.2, 0.4, 0.6, 0.8])
    out = bootstrap_paired(a, a.copy())
    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["lo"] == pytest.approx(0.0)
    assert out["hi"] == pytest.approx(0.0)
    assert (out["wins"], out["losses"], out["ties"]) == (0, 0, 4)


def test_bootstrap_paired_counts_wins_losses_and_near_ties():
    a = np.array([0.5, 0.1, 0.30000000000001, 0.9])
    b = np.array([0.2, 0.4, 0.30000000000000, 0.4])
    out = bootstrap_paired(a, b)
    # The third pair differs by 1e-14, under the 1e-9 tie threshold.
    assert (out["wins"], out["losses"], out["ties"]) == (2, 1, 1)
    assert out["wins"] + out["losses"] + out["ties"] == 4
    assert out["mean_diff"] == pytest.approx((a - b).mean())


def test_bootstrap_paired_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bootstrap_paired(np.array([1.0, 2.0]), np.array([1.0]))


# ------------------------------------------------------------- run level ---


def test_eval_topics_keeps_only_topics_with_a_relevant_document():
    qrels = {
        7: {"d1": 0, "d2": 0},
        3: {"d1": 1},
        5: {"d9": 1, "d8": 0},
    }
    assert eval_topics(qrels) == [3, 5]


def test_eval_topics_on_the_fixture_qrels():
    assert eval_topics(load_qrels(FIXTURE / "qrels.txt")) == [1, 2]


def test_evaluate_run_shapes_and_values():
    run = {
        1: [("a", 3.0), ("b", 2.0), ("c", 1.0), ("d", 0.5), ("e", 0.25)],
        2: [],
    }
    qrels = {
        1: {"a": 1, "b": 0, "c": 1, "e": 1},
        2: {"z": 1},
    }
    out = evaluate_run(run, qrels, [1, 2])
    for key in ("ap", "p10", "ndcg10", "rprec", "recall100", "judged10"):
        assert out[key].shape == (2,)
    assert out["interp"].shape == (2, 11)
    assert out["ap"][0] == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    # judged@10: four of topic 1's judgments land in the ten slots.
    assert out["judged10"][0] == pytest.approx(0.4)
    assert out["recall100"][0] == pytest.approx(1.0)
    # An empty ranking is zero for every metric and an all-zero curve.
    assert out["ap"][1] == 0.0
    assert out["judged10"][1] == 0.0
    assert out["interp"][1].tolist() == [0.0] * 11


def test_evaluate_run_tolerates_a_topic_missing_from_the_run():
    out = evaluate_run({}, {4: {"a": 1}}, [4])
    assert out["ap"].tolist() == [0.0]
    assert out["interp"].shape == (1, 11)


def test_evaluate_run_rejects_duplicate_docnos():
    run = {1: [("a", 2.0), ("a", 1.0)]}
    qrels = {1: {"a": 1}}
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_run(run, qrels, [1])
