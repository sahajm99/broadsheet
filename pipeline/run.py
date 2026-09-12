"""`python -m pipeline`: the corpus in, every file the site reads out.

One pass. The corpus is parsed once, the four indexes are built once, the
evaluation matrix is scored once, and the writers are pure projections of what
is already in memory -- so two runs on the same corpus produce byte-identical
files apart from `generated_at`, and a diff under `site/public/data` always
means a real change.

Order matters in two places. The output directory is emptied only after every
number has been computed, so a crash halfway through leaves the last good set
of files in place rather than a half-written one. And `claims.json` is written
last, because it is read back out of the other files (see `claims.py`): it
cannot be produced before the numbers it quotes exist on disk.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from pipeline.claims import build_claims, build_meta
from pipeline.constants import (
    GOLDEN_DIR,
    OUT_DIR,
    QRELS_PATH,
    RAW_DIR,
    ROOT,
    STOPWORDS_JSON,
    TOPICS_PATH,
)
from pipeline.index import build_all
from pipeline.io import compact_json, gzip_json, write_bytes, write_json
from pipeline.parse import load_corpus
from pipeline.runs import (
    build_runs,
    write_comparisons,
    write_grid,
    write_per_topic,
    write_pr_curves,
    write_runs,
    write_topics,
    write_treatments,
)
from pipeline.search_export import (
    build_search_index,
    export_golden,
    export_stopwords,
)
from pipeline.tokenize import load_stopwords
from pipeline.topics import load_qrels, load_topics
from pipeline.vocab_stats import write_corpus, write_postings, write_vocab

SEARCH_DIR_NAME = "search"
SEARCH_INDEX_NAME = "index.json.gz"


def _label(path: Path) -> str:
    """A path as the report should print it: repo-relative, forward slashes."""
    path = Path(path)
    try:
        path = path.relative_to(ROOT)
    except ValueError:
        pass
    return path.as_posix()


def _clean(out_dir: Path) -> None:
    """Remove the previous run's output so a renamed file cannot linger."""
    for path in sorted(out_dir.glob("*.json")):
        path.unlink()
    for path in sorted((out_dir / SEARCH_DIR_NAME).glob("*.gz")):
        path.unlink()


def _print_table(rows: list[tuple[str, int]]) -> None:
    width = max((len(name) for name, _ in rows), default=0)
    total = sum(size for _, size in rows)
    print()
    print(f"{'file'.ljust(width)}  {'bytes':>11}")
    print(f"{'-' * width}  {'-' * 11}")
    for name, size in rows:
        print(f"{name.ljust(width)}  {size:>11,}")
    print(f"{'-' * width}  {'-' * 11}")
    print(f"{'total'.ljust(width)}  {total:>11,}")


def run(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OUT_DIR,
    golden_dir: Path = GOLDEN_DIR,
    stopwords_json: Path = STOPWORDS_JSON,
) -> list[tuple[str, int]]:
    """Build everything and write it; return `[(file name, bytes)]`."""
    started = time.perf_counter()

    print(f"reading {_label(raw_dir)}")
    docs = load_corpus(raw_dir)
    n_files = sum(1 for path in Path(raw_dir).iterdir() if path.is_file())
    stopwords = load_stopwords()
    print(f"  {len(docs):,} documents in {n_files} files")

    print("building four indexes")
    indexes = build_all(docs, stopwords)
    stem_stop = indexes["stem_stop"]
    print(f"  stem_stop: {len(stem_stop.postings):,} terms")

    print(f"reading {TOPICS_PATH.name} and {QRELS_PATH.name}")
    topics = load_topics()
    qrels = load_qrels(docnos=set(stem_stop.docnos))

    print("scoring 9 runs, 4 treatments, 25 grid cells, 8 comparisons")
    results = build_runs(indexes, topics, qrels, stopwords)
    print(f"  {results['counts']['n_topics']} evaluable topics")

    # The search payload is compressed here rather than at the point it is
    # written because `postings.json`, written first, reports its size.
    print("compressing the browser index")
    search_obj = build_search_index(stem_stop, docs)
    payload = gzip_json(search_obj)
    sizes = {"index_gz": len(payload), "index_json": len(compact_json(search_obj))}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean(out_dir)

    rows: list[tuple[str, int]] = []

    def written(path: Path, size: int | None = None) -> None:
        rows.append((_label(path), size if size is not None else path.stat().st_size))

    written(write_corpus(docs, stem_stop, n_files, out_dir))
    written(write_vocab(docs, stopwords, indexes, out_dir))
    written(write_postings(stem_stop, sizes, out_dir))
    written(write_runs(results, out_dir))
    written(write_per_topic(results, out_dir))
    written(write_comparisons(results, out_dir))
    written(write_pr_curves(results, out_dir))
    written(write_topics(results, topics, qrels, out_dir))
    written(write_grid(results, out_dir))
    written(write_treatments(results, indexes, out_dir))

    search_path = out_dir / SEARCH_DIR_NAME / SEARCH_INDEX_NAME
    written(search_path, write_bytes(search_path, payload))

    for name, size in export_golden(
        stem_stop, docs, topics, stopwords, golden_dir
    ).items():
        written(Path(golden_dir) / name, size)
    written(Path(stopwords_json), export_stopwords(stopwords, stopwords_json))

    claims_path = out_dir / "claims.json"
    written(claims_path, write_json(claims_path, build_claims(out_dir)))
    meta_path = out_dir / "meta.json"
    written(meta_path, write_json(meta_path, build_meta(len(docs), sizes, raw_dir)))

    _print_table(rows)
    print(f"\ndone in {time.perf_counter() - started:.1f}s")
    return rows


def main() -> int:
    """Entry point: 0 on success, 1 with a traceback on any failure."""
    try:
        run()
    except Exception:  # noqa: BLE001 - the exit code is the contract
        traceback.print_exc()
        print("\npipeline failed; see the traceback above", file=sys.stderr)
        return 1
    return 0
