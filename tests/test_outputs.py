"""Structural checks on the committed output of a real pipeline run.

The corpus is licensed and cannot be in CI (D1), so CI can never rebuild these
files; what it can do is read the ones in the repository and insist they are
internally consistent (P1). That is what this module is: no corpus, no
rebuild, only the committed JSON, the committed gzip and the committed golden
fixtures.

Two kinds of assertion. **Structural** ones -- schema strings, array lengths
that must agree with each other, bootstrap bounds that must bracket their
mean, PR curves that must not rise, wins + losses + ties that must equal the
topic count -- would catch a writer wired to the wrong array. **Anchors** --
5,368 documents, 71 evaluable topics, 186 relevant judgments -- would catch a
corpus or qrels file swapped for a different one. Nothing here asserts that
one ranker beat another: that is a result, and a test that demanded it would
be a test that forbids the answer from changing.

When the outputs have not been generated yet the whole module skips with a
message saying how to make them.
"""

import gzip
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "site/public/data"
GOLDEN = ROOT / "tests/golden"
SEARCH_INDEX = DATA / "search/index.json.gz"
STOPWORDS_JSON = ROOT / "site/src/search/stopwords.json"

SIZE_LIMIT = 200_000
INDEX_GZ_LIMIT = 2_500_000
N_STOPWORDS = 523
PARITY_TOPIC = "352"

# file stem -> schema string
CONTRACT = {
    "corpus": "corpus.v1",
    "vocab": "vocab.v1",
    "postings": "postings.v1",
    "runs": "runs.v1",
    "per_topic": "per_topic.v1",
    "comparisons": "comparisons.v1",
    "pr_curves": "pr_curves.v1",
    "topics": "topics.v1",
    "bm25_grid": "bm25_grid.v1",
    "treatments": "treatments.v1",
    "claims": "claims.v1",
    "meta": "meta.v1",
}

RANKERS = ("tfidf_raw", "tfidf_log", "bm25")
FIELDS = ("title", "title_desc", "title_desc_narr")
RUN_KEYS = [f"{ranker}|{field}" for ranker in RANKERS for field in FIELDS]
METRICS = ("map", "p10", "ndcg10", "rprec", "recall100", "judged10")

CLAIM_KEYS = """
n_docs n_files date_min date_max total_tokens_alpha n_types_raw n_types_alpha
n_types_nostop n_types_stem stem_reduction_pct stopword_token_share_pct
df1_share_pct n_postings median_doc_len mean_doc_len max_posting_term
max_posting_df n_topics_all n_topics_eval n_rel_total n_judged_docs
n_single_rel_topics heaps_beta zipf_slope map_tfidf_raw_title
map_tfidf_raw_title_lo map_tfidf_raw_title_hi map_tfidf_log_title
map_bm25_title map_bm25_title_lo map_bm25_title_hi p10_bm25_title
ndcg10_bm25_title judged10_bm25_title recall100_bm25_title diff_bm25_raw
diff_bm25_raw_lo diff_bm25_raw_hi wins_bm25_raw losses_bm25_raw ties_bm25_raw
map_bm25_title_desc map_bm25_title_desc_narr diff_desc_title diff_desc_title_lo
diff_desc_title_hi diff_narr_title diff_narr_title_lo diff_narr_title_hi
map_nostem_stop diff_stem_nostem diff_stem_nostem_lo diff_stem_nostem_hi
map_stem_nostop diff_stop_nostop diff_stop_nostop_lo diff_stop_nostop_hi
grid_best_map grid_best_k1 grid_best_b grid_default_map index_gz_mb
index_gz_bytes n_terms_index pr_p_at_r0_bm25 pr_p_at_r0_raw
""".split()

# The anchors: what a correct run over this corpus and these judgments gives.
N_DOCS = 5368
N_FILES = 15
N_TOPICS_EVAL = 71
N_TOPICS_ALL = 250
N_REL_TOTAL = 186
N_JUDGED_DOCS = 1844
N_SINGLE_REL_TOPICS = 36

pytestmark = pytest.mark.skipif(
    not (DATA / "meta.json").exists(),
    reason=(
        "no generated output: run `uv run python -m pipeline` with the corpus "
        "in data/raw/ft911 (see data/raw/SOURCE.md). CI does not have the "
        "corpus and reads the committed files instead."
    ),
)


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def load_golden(name: str) -> dict:
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs():
    return load("runs")


@pytest.fixture(scope="module")
def claims():
    return load("claims")["values"]


@pytest.fixture(scope="module")
def n_topics(runs):
    return runs["n_topics"]


# ------------------------------------------------------------- every file ---


@pytest.mark.parametrize("name,schema", sorted(CONTRACT.items()))
def test_every_contract_file_exists_and_carries_its_schema(name, schema):
    path = DATA / f"{name}.json"
    assert path.exists(), f"{path} was not written"
    obj = load(name)
    assert obj["schema"] == schema
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", obj["generated_at"])


@pytest.mark.parametrize("name", sorted(CONTRACT))
def test_every_contract_file_is_small_enough_to_fetch(name):
    size = (DATA / f"{name}.json").stat().st_size
    assert size < SIZE_LIMIT, f"{name}.json is {size:,} bytes"


@pytest.mark.parametrize("name", sorted(CONTRACT))
def test_files_are_utf8_with_lf_newlines(name):
    text = (DATA / f"{name}.json").read_text(encoding="utf-8")
    assert "\r" not in text
    assert text.endswith("\n")


# ------------------------------------------------------------------ corpus ---


def test_corpus_counts_and_span():
    corpus = load("corpus")
    assert corpus["n_docs"] == N_DOCS
    assert corpus["n_files"] == N_FILES
    assert corpus["date_min"] <= corpus["date_max"]
    assert corpus["date_min"].startswith("1991-")
    assert corpus["date_max"].startswith("1991-")
    assert sum(corpus["doc_len"]["counts"]) == N_DOCS
    assert len(corpus["doc_len"]["bins"]) == len(corpus["doc_len"]["counts"])
    summary = corpus["doc_len_summary"]
    assert 0 < summary["p10"] <= summary["median"] <= summary["p90"] <= summary["max"]


def test_vocab_funnel_only_shrinks():
    vocab = load("vocab")
    steps = [row["step"] for row in vocab["funnel"]]
    assert steps == ["raw_alpha", "minus_digits", "minus_stopwords", "stemmed"]
    types = [row["n_types"] for row in vocab["funnel"]]
    tokens = [row["n_tokens"] for row in vocab["funnel"]]
    assert types == sorted(types, reverse=True)
    assert tokens == sorted(tokens, reverse=True)
    assert vocab["heaps_fit"]["beta"] > 0
    assert vocab["zipf_fit"]["slope"] < 0
    assert len(vocab["top_terms"]) == 25
    assert len(vocab["stem_groups"]) == 12


def test_postings_summary_agrees_with_its_histogram():
    postings = load("postings")
    summary = postings["summary"]
    assert sum(postings["hist"]["counts"]) == summary["n_terms"]
    assert summary["n_postings"] > summary["n_terms"]
    assert 0 < summary["df1_share"] < 1
    assert summary["max_len"] == postings["longest"][0]["df"]
    sizes = postings["size_bytes"]
    assert sizes["index_gz"] == SEARCH_INDEX.stat().st_size
    assert sizes["index_json"] > sizes["index_gz"]


# -------------------------------------------------------------- the runs ---


def test_runs_has_nine_runs_with_bracketed_intervals(runs):
    assert [run["key"] for run in runs["runs"]] == RUN_KEYS
    assert runs["n_topics"] == N_TOPICS_EVAL
    assert runs["n_topics_all"] == N_TOPICS_ALL
    assert runs["n_rel_total"] == N_REL_TOTAL
    assert runs["n_judged_docs"] == N_JUDGED_DOCS
    assert runs["n_single_rel_topics"] == N_SINGLE_REL_TOPICS
    for run in runs["runs"]:
        assert run["ranker"] in RANKERS
        assert run["field"] in FIELDS
        assert run["key"] == f"{run['ranker']}|{run['field']}"
        assert tuple(run["metrics"]) == METRICS
        for name, stats in run["metrics"].items():
            lo, mean, hi = stats["lo"], stats["mean"], stats["hi"]
            assert 0 <= lo <= mean <= hi <= 1, f"{run['key']} {name}: {stats}"
            assert mean > 0, f"{run['key']} {name} is zero"


def test_per_topic_arrays_line_up_with_the_topic_list(n_topics):
    per_topic = load("per_topic")
    assert len(per_topic["topics"]) == n_topics
    assert per_topic["topics"] == sorted(per_topic["topics"])
    assert sorted(per_topic["ap"]) == sorted(RUN_KEYS)
    for key, values in per_topic["ap"].items():
        assert len(values) == n_topics, key
        assert all(0 <= value <= 1 for value in values), key


def test_comparisons_account_for_every_topic(n_topics):
    pairs = load("comparisons")["pairs"]
    assert len(pairs) == 8
    for pair in pairs:
        assert pair["wins"] + pair["losses"] + pair["ties"] == n_topics, pair["key"]
        assert pair["lo"] <= pair["mean_diff"] <= pair["hi"], pair["key"]
        assert len(pair["per_topic"]) == n_topics, pair["key"]
        mean = sum(row["diff"] for row in pair["per_topic"]) / n_topics
        assert mean == pytest.approx(pair["mean_diff"], abs=1e-6), pair["key"]


def test_pr_curves_never_rise_with_recall():
    curves = load("pr_curves")
    assert curves["recall_levels"] == [round(i / 10, 6) for i in range(11)]
    assert sorted(curves["curves"]) == sorted(RUN_KEYS)
    for key, values in curves["curves"].items():
        assert len(values) == 11, key
        assert all(0 <= value <= 1 for value in values), key
        assert values == sorted(values, reverse=True), key


def test_topics_rows_are_ascending_and_judged(n_topics):
    rows = load("topics")["rows"]
    assert len(rows) == n_topics
    nums = [row["num"] for row in rows]
    assert nums == sorted(nums)
    assert sum(row["n_rel"] for row in rows) == N_REL_TOTAL
    for row in rows:
        assert row["title"]
        assert 1 <= row["n_rel"] <= row["n_judged"]
        assert sorted(row["ap"]) == sorted(RUN_KEYS)
        assert row["best_run"] in RUN_KEYS


def test_grid_is_five_by_five_and_the_best_cell_is_the_best():
    grid = load("bm25_grid")
    assert grid["k1"] == [0.6, 0.9, 1.2, 1.5, 2.0]
    assert grid["b"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(grid["map"]) == 5
    assert all(len(row) == 5 for row in grid["map"])
    flat = [value for row in grid["map"] for value in row]
    assert grid["best"]["map"] == max(flat)
    assert grid["best"]["map"] >= grid["default"]["map"]
    assert grid["default"]["k1"] == 1.2 and grid["default"]["b"] == 0.75
    assert grid["map"][grid["k1"].index(1.2)][grid["b"].index(0.75)] == pytest.approx(
        grid["default"]["map"]
    )
    assert "same" in grid["caveat"]


def test_treatments_are_the_four_index_variants():
    treatments = load("treatments")
    assert treatments["ranker"] == "bm25" and treatments["field"] == "title"
    rows = treatments["rows"]
    assert [row["key"] for row in rows] == [
        "stem_stop",
        "nostem_stop",
        "stem_nostop",
        "nostem_nostop",
    ]
    for row in rows:
        assert row["n_types"] > 0
        for metric in ("map", "p10"):
            stats = row[metric]
            assert 0 < stats["lo"] <= stats["mean"] <= stats["hi"] < 1
    # Stemming conflates surface forms, so a stemmed index is always smaller.
    by_key = {row["key"]: row for row in rows}
    assert by_key["stem_stop"]["n_types"] < by_key["nostem_stop"]["n_types"]


# ------------------------------------------------------------ claims/meta ---


def test_claims_has_every_key_and_no_null(claims):
    assert sorted(claims) == sorted(CLAIM_KEYS)
    for key, value in claims.items():
        assert value is not None, key
        assert isinstance(value, (int, float, str)), key
        assert not isinstance(value, bool), key


def test_claims_agree_with_the_files_they_quote(claims, runs):
    corpus = load("corpus")
    assert claims["n_docs"] == corpus["n_docs"] == N_DOCS
    assert claims["n_topics_eval"] == N_TOPICS_EVAL
    assert claims["n_rel_total"] == N_REL_TOTAL
    assert claims["n_judged_docs"] == N_JUDGED_DOCS
    assert claims["n_single_rel_topics"] == N_SINGLE_REL_TOPICS
    bm25 = next(run for run in runs["runs"] if run["key"] == "bm25|title")
    assert claims["map_bm25_title"] == bm25["metrics"]["map"]["mean"]
    assert claims["map_bm25_title_lo"] == bm25["metrics"]["map"]["lo"]
    assert claims["map_bm25_title_hi"] == bm25["metrics"]["map"]["hi"]
    assert claims["n_terms_index"] == load("postings")["summary"]["n_terms"]
    assert claims["index_gz_bytes"] == SEARCH_INDEX.stat().st_size
    assert claims["index_gz_mb"] == round(claims["index_gz_bytes"] / 1e6, 2)
    assert claims["pr_p_at_r0_bm25"] == load("pr_curves")["curves"]["bm25|title"][0]


def test_claims_that_are_metrics_are_proper_fractions(claims):
    for key, value in claims.items():
        if key.startswith(("map_", "p10_", "ndcg10_", "judged10_", "recall100_")):
            assert 0 < value < 1, key
        if key.endswith("_pct"):
            assert 0 <= value <= 100, key
            assert value == round(value, 1), key


def test_meta_records_the_corpus_it_was_built_from():
    meta = load("meta")
    files = meta["corpus_files"]
    assert len(files) == N_FILES
    for row in files:
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), row["name"]
        assert row["name"].startswith("ft911_")
    assert len({row["sha256"] for row in files}) == N_FILES
    assert meta["n_docs"] == N_DOCS
    assert meta["seed"] == 20260912
    assert meta["n_boot"] == 10000
    assert meta["python"].startswith("3.")
    assert meta["numpy"]
    assert meta["index_gz_bytes"] == SEARCH_INDEX.stat().st_size


# ---------------------------------------------------------- search index ---


@pytest.fixture(scope="module")
def search_index():
    return json.loads(gzip.decompress(SEARCH_INDEX.read_bytes()).decode("utf-8"))


def test_search_index_is_small_enough_to_download():
    size = SEARCH_INDEX.stat().st_size
    assert size < INDEX_GZ_LIMIT, f"index.json.gz is {size:,} bytes"


def test_search_index_shape(search_index):
    assert search_index["schema"] == "search_index.v1"
    assert search_index["n_docs"] == N_DOCS
    assert len(search_index["docs"]) == N_DOCS
    assert len(search_index["doc_len"]) == N_DOCS
    assert search_index["k1"] == 1.2 and search_index["b"] == 0.75
    assert search_index["avgdl"] == pytest.approx(
        sum(search_index["doc_len"]) / N_DOCS, abs=1e-6
    )
    assert len(search_index["terms"]) == load("postings")["summary"]["n_terms"]


def test_search_index_ships_citations_not_articles(search_index):
    assert "text" not in search_index
    for row in search_index["docs"]:
        assert set(row) == {"no", "d", "h"}
        assert row["no"].startswith("FT911-")
        assert re.fullmatch(r"1991-\d{2}-\d{2}", row["d"])
        assert len(row["h"]) <= 200
    # Every value in the file is a number, a stem, a DOCNO, a date or a
    # headline. The only strings are the `docs` rows and the term keys, so
    # there is no field anywhere whose value is prose. ("text" itself is a
    # perfectly good stem and appears as a key under `terms`.)
    strings = {
        value
        for row in search_index["docs"]
        for value in row.values()
    }
    assert all(len(value) <= 200 for value in strings)
    assert all(
        isinstance(flat, list) and all(isinstance(n, int) for n in flat)
        for flat in search_index["terms"].values()
    )


def test_search_index_gaps_are_well_formed(search_index):
    n_postings = 0
    for term, flat in list(search_index["terms"].items())[:2000]:
        assert len(flat) % 2 == 0 and flat, term
        doc_idx = 0
        for i in range(0, len(flat), 2):
            doc_idx += flat[i]
            assert 0 <= doc_idx < N_DOCS, term
            assert flat[i + 1] > 0, term
            assert flat[i] >= 0, term
        n_postings += len(flat) // 2
    assert n_postings > 0


def test_stopwords_json_is_the_course_list():
    words = json.loads(STOPWORDS_JSON.read_text(encoding="utf-8"))
    assert len(words) == N_STOPWORDS
    assert words == sorted(words)
    assert "the" in words and "able" in words
    assert all(word and word == word.strip().lower() for word in words)


# --------------------------------------------------------- golden fixtures ---


def test_topic_titles_cover_all_250_topics():
    titles = load_golden("topic_titles.json")
    assert len(titles) == N_TOPICS_ALL
    nums = [int(key) for key in titles]
    assert nums == sorted(nums)
    assert all(title.strip() for title in titles.values())


@pytest.mark.parametrize(
    "name", ["top10_bm25_title.json", "top10_tfidf_raw_title.json"]
)
def test_top10_fixtures_are_ranked_lists_of_citations(name):
    top10 = load_golden(name)
    titles = load_golden("topic_titles.json")
    assert set(top10) <= set(titles)
    assert len(top10) > 200
    for num, rows in top10.items():
        assert 1 <= len(rows) <= 10, num
        scores = [row["score"] for row in rows]
        assert scores == sorted(scores, reverse=True), num
        for row in rows:
            assert set(row) == {"no", "score"}
            assert row["no"].startswith("FT911-")
            assert row["score"] > 0


@pytest.mark.parametrize(
    "name", ["top10_bm25_title.json", "top10_tfidf_raw_title.json"]
)
def test_the_parity_topic_has_a_full_top_ten(name):
    rows = load_golden(name)[PARITY_TOPIC]
    assert len(rows) == 10
    assert len({row["no"] for row in rows}) == 10


def test_golden_fixtures_are_utf8_with_lf_newlines():
    for name in (
        "topic_titles.json",
        "top10_bm25_title.json",
        "top10_tfidf_raw_title.json",
    ):
        text = (GOLDEN / name).read_text(encoding="utf-8")
        assert "\r" not in text
        assert text.endswith("\n")
