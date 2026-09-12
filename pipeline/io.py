"""Deterministic JSON and gzip writers for everything the site reads.

Two rules the site depends on: every file is small enough to fetch on a cold
page load (`SIZE_LIMIT`), and two runs of the pipeline on the same corpus
produce byte-identical output, so a diff in `site/public/data` always means a
real change. Hence the fixed separators, the sorted-nothing/insertion-order
dicts, `allow_nan=False` (a NaN would be invalid JSON) and `mtime=0` in gzip.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.constants import SIZE_LIMIT


def now_iso() -> str:
    """UTC timestamp to the second, e.g. `"2026-09-12T14:03:01Z"`."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def round6(x: float) -> float:
    """Every float in the contract is written to 6 decimal places."""
    return round(float(x), 6)


def write_json(path: Path, obj: dict, limit: int = SIZE_LIMIT) -> int:
    """Write `obj` as UTF-8 JSON with LF newlines; return the byte size.

    Raises `ValueError` naming the file when the result is over `limit`, so a
    chart that quietly grew past what a browser should download fails the
    pipeline instead of the page.
    """
    path = Path(path)
    text = json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False) + "\n"
    data = text.encode("utf-8")
    if len(data) > limit:
        raise ValueError(
            f"{path.name} is {len(data):,} bytes, over the {limit:,} byte limit"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return len(data)


def write_gz(path: Path, obj: dict) -> int:
    """Write `obj` as gzipped compact JSON; return the compressed byte size."""
    path = Path(path)
    data = json.dumps(obj, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw:
        # filename="" keeps the path out of the gzip header: with `fileobj`
        # alone, gzip stores the output file's name and two identical payloads
        # written to different paths would differ byte-for-byte.
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
        ) as gz:
            gz.write(data.encode("utf-8"))
    return path.stat().st_size
