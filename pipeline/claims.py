"""`claims.json` and `meta.json`: every number the prose is allowed to say.

No numeral describing the data is hand-typed into the site (W7). The page
writes `<span data-stat="map_bm25_title|3">` and the filler reads this file, so
a sentence can never drift from the chart beside it.

`build_claims` therefore reads the contract files that have already been
written rather than recomputing anything from the corpus. That is the whole
point: a claim is by construction the same number the chart draws, because it
is literally read back out of the file the chart reads. If the two ever
disagree, the pipeline is broken in a way no amount of careful duplication
here would catch.

`build_meta` is the provenance record. The corpus itself cannot be committed
(D1), so what is committed is the SHA-256 of every file it was built from,
re-verified against the files on disk at every run -- a rebuild from a
different copy of the FT collection fails loudly instead of silently
publishing different numbers under the same hashes.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from pipeline.constants import N_BOOT, OUT_DIR, RAW_DIR, ROOT, SEED
from pipeline.io import now_iso, round6

SHA256SUMS = ROOT / "data/raw/SHA256SUMS"
HASH_CHUNK = 1 << 20

# Files `build_claims` reads back. Every one is written before it.
CONTRACT = (
    "corpus",
    "vocab",
    "postings",
    "runs",
    "comparisons",
    "pr_curves",
    "topics",
    "bm25_grid",
    "treatments",
)


def _load(out_dir: Path, name: str) -> dict:
    return json.loads((Path(out_dir) / f"{name}.json").read_text(encoding="utf-8"))


def _pct(part: float, whole: float) -> float:
    """A share as a percentage to one decimal place, 0.0 when the base is 0."""
    return round(100.0 * part / whole, 1) if whole else 0.0


def _mb(size_bytes: int) -> float:
    """Megabytes as a reader counts them: bytes / 1e6, two decimals."""
    return round(size_bytes / 1e6, 2)


def build_claims(out_dir: Path = OUT_DIR) -> dict:
    """The `claims.v1` object, projected from the files already written."""
    data = {name: _load(out_dir, name) for name in CONTRACT}
    corpus, vocab, postings = data["corpus"], data["vocab"], data["postings"]
    runs, grid = data["runs"], data["bm25_grid"]

    funnel = {row["step"]: row for row in vocab["funnel"]}
    raw, alpha = funnel["raw_alpha"], funnel["minus_digits"]
    nostop, stemmed = funnel["minus_stopwords"], funnel["stemmed"]
    summary = postings["summary"]
    widest = postings["longest"][0]
    gz_bytes = postings["size_bytes"]["index_gz"]

    metrics = {run["key"]: run["metrics"] for run in runs["runs"]}
    pairs = {pair["key"]: pair for pair in data["comparisons"]["pairs"]}
    treatments = {row["key"]: row for row in data["treatments"]["rows"]}
    curves = data["pr_curves"]["curves"]

    def interval(run_key: str, metric: str, prefix: str) -> dict:
        stats = metrics[run_key][metric]
        return {
            prefix: stats["mean"],
            f"{prefix}_lo": stats["lo"],
            f"{prefix}_hi": stats["hi"],
        }

    def difference(pair_key: str, prefix: str) -> dict:
        pair = pairs[pair_key]
        return {
            prefix: pair["mean_diff"],
            f"{prefix}_lo": pair["lo"],
            f"{prefix}_hi": pair["hi"],
        }

    values = {
        # -- the corpus
        "n_docs": corpus["n_docs"],
        "n_files": corpus["n_files"],
        "date_min": corpus["date_min"],
        "date_max": corpus["date_max"],
        "total_tokens_alpha": corpus["total_tokens_alpha"],
        # -- the vocabulary funnel
        "n_types_raw": raw["n_types"],
        "n_types_alpha": alpha["n_types"],
        "n_types_nostop": nostop["n_types"],
        "n_types_stem": stemmed["n_types"],
        "stem_reduction_pct": _pct(
            nostop["n_types"] - stemmed["n_types"], nostop["n_types"]
        ),
        "stopword_token_share_pct": _pct(
            alpha["n_tokens"] - nostop["n_tokens"], alpha["n_tokens"]
        ),
        # -- the index
        "df1_share_pct": round(100.0 * summary["df1_share"], 1),
        "n_postings": summary["n_postings"],
        "median_doc_len": corpus["doc_len_summary"]["median"],
        "mean_doc_len": corpus["doc_len_summary"]["mean"],
        "max_posting_term": widest["term"],
        "max_posting_df": widest["df"],
        "n_terms_index": summary["n_terms"],
        "index_gz_bytes": gz_bytes,
        "index_gz_mb": _mb(gz_bytes),
        "heaps_beta": round6(vocab["heaps_fit"]["beta"]),
        "zipf_slope": round6(vocab["zipf_fit"]["slope"]),
        # -- what the judgments cover
        "n_topics_all": runs["n_topics_all"],
        "n_topics_eval": runs["n_topics"],
        "n_rel_total": runs["n_rel_total"],
        "n_judged_docs": runs["n_judged_docs"],
        "n_single_rel_topics": runs["n_single_rel_topics"],
        # -- the rankers on the title
        **interval("tfidf_raw|title", "map", "map_tfidf_raw_title"),
        "map_tfidf_log_title": metrics["tfidf_log|title"]["map"]["mean"],
        **interval("bm25|title", "map", "map_bm25_title"),
        "p10_bm25_title": metrics["bm25|title"]["p10"]["mean"],
        "ndcg10_bm25_title": metrics["bm25|title"]["ndcg10"]["mean"],
        "judged10_bm25_title": metrics["bm25|title"]["judged10"]["mean"],
        "recall100_bm25_title": metrics["bm25|title"]["recall100"]["mean"],
        **difference("bm25_vs_raw", "diff_bm25_raw"),
        "wins_bm25_raw": pairs["bm25_vs_raw"]["wins"],
        "losses_bm25_raw": pairs["bm25_vs_raw"]["losses"],
        "ties_bm25_raw": pairs["bm25_vs_raw"]["ties"],
        # -- longer queries
        "map_bm25_title_desc": metrics["bm25|title_desc"]["map"]["mean"],
        "map_bm25_title_desc_narr": metrics["bm25|title_desc_narr"]["map"]["mean"],
        **difference("desc_vs_title", "diff_desc_title"),
        **difference("narr_vs_title", "diff_narr_title"),
        # -- stemming and stopwording
        "map_nostem_stop": treatments["nostem_stop"]["map"]["mean"],
        **difference("stem_vs_nostem", "diff_stem_nostem"),
        "map_stem_nostop": treatments["stem_nostop"]["map"]["mean"],
        **difference("stop_vs_nostop", "diff_stop_nostop"),
        # -- tuning
        "grid_best_map": grid["best"]["map"],
        "grid_best_k1": grid["best"]["k1"],
        "grid_best_b": grid["best"]["b"],
        "grid_default_map": grid["default"]["map"],
        # -- the shape of the PR curves at recall 0 (S3)
        "pr_p_at_r0_bm25": curves["bm25|title"][0],
        "pr_p_at_r0_raw": curves["tfidf_raw|title"][0],
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"claims are missing values for {missing}")
    return {"schema": "claims.v1", "generated_at": now_iso(), "values": values}


# ------------------------------------------------------------ provenance ---


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_files(
    raw_dir: Path = RAW_DIR, sums_path: Path = SHA256SUMS
) -> list[dict]:
    """`[{name, sha256}]` from `SHA256SUMS`, verified against the files.

    Raises `ValueError` naming every file whose content does not match, so a
    run against a different copy of the FT collection cannot publish numbers
    under these hashes (D1).
    """
    rows: list[dict] = []
    for line in Path(sums_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, name = line.partition(" ")
        rows.append({"name": Path(name.strip().lstrip("*")).name, "sha256": expected})

    bad = []
    for row in rows:
        path = Path(raw_dir) / row["name"]
        if not path.is_file():
            bad.append(f"{row['name']}: missing from {raw_dir}")
            continue
        actual = sha256_file(path)
        if actual != row["sha256"]:
            bad.append(f"{row['name']}: {actual} != {row['sha256']}")
    if bad:
        raise ValueError(
            "the corpus on disk does not match data/raw/SHA256SUMS:\n  "
            + "\n  ".join(bad)
        )
    return rows


def build_meta(
    n_docs: int,
    sizes: dict[str, int],
    raw_dir: Path = RAW_DIR,
    sums_path: Path = SHA256SUMS,
) -> dict:
    """The `meta.v1` object: what was built, from what, with what."""
    gz_bytes = sizes["index_gz"]
    return {
        "schema": "meta.v1",
        "generated_at": now_iso(),
        "corpus_files": corpus_files(raw_dir, sums_path),
        "n_docs": n_docs,
        "index_gz_bytes": gz_bytes,
        "index_gz_mb": _mb(gz_bytes),
        "seed": SEED,
        "n_boot": N_BOOT,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
