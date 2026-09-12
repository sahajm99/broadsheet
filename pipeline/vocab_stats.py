"""Vocabulary statistics: the funnel, Heaps, Zipf, and the two distributions.

These are the numbers behind the first four charts on the site. Everything is
computed over the TEXT field of the corpus, which is what `build_index` also
indexes, so a reader can tie the funnel's last row to the `stem_stop`
vocabulary size without a footnote.

Ordering note carried over from the tokenizer: stopwords are removed *before*
stemming, so the "stemmed" funnel row is the stem of an already-stopworded
stream. That is the course pipeline, and it is why inflected stopwords such as
"used" survive while "use" (from "uses") does not.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pipeline.constants import OUT_DIR, TREATMENTS
from pipeline.index import Index
from pipeline.io import now_iso, round6, write_json
from pipeline.parse import Doc
from pipeline.porter import stem as porter_stem
# The funnel's first two rows must use the tokenizer's own rules, not a copy of
# them, or the chart could drift away from the index it is meant to explain.
from pipeline.tokenize import _HAS_DIGIT_RE, _TOKEN_RE, analyze, tokenize

# Closed-open left edges; the last bin catches everything at or above its edge.
DOC_LEN_BINS = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]
POSTING_BINS = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097]

FUNNEL_LABELS = {
    "raw_alpha": "Alphanumeric tokens",
    "minus_digits": "Tokens with digits dropped",
    "minus_stopwords": "Stopwords removed",
    "stemmed": "Porter stemmed",
}

HEAPS_POINTS = 40
HEAPS_START = 1000
ZIPF_HEAD = 10
ZIPF_TAIL_POINTS = 50
ZIPF_FIT_RANKS = 1000
N_STEM_GROUPS = 12
N_STEM_WORDS = 8


# ------------------------------------------------------------------ funnel ---


def funnel(docs: list[Doc], stopwords: frozenset[str]) -> list[dict]:
    """What each tokenizer rule costs, as four shrinking steps.

    `raw_alpha` is every `[a-z0-9]+` run of the lowercased text, before the
    digit rule; the three rules that follow are applied in pipeline order.
    """
    raw = Counter()
    nodigit = Counter()
    nostop = Counter()
    stemmed = Counter()

    for doc in docs:
        for token in _TOKEN_RE.findall(doc.text.lower()):
            raw[token] += 1
            if _HAS_DIGIT_RE.search(token):
                continue
            nodigit[token] += 1
            if token in stopwords:
                continue
            nostop[token] += 1
            stemmed[porter_stem(token)] += 1

    counters = {
        "raw_alpha": raw,
        "minus_digits": nodigit,
        "minus_stopwords": nostop,
        "stemmed": stemmed,
    }
    return [
        {
            "step": step,
            "label": FUNNEL_LABELS[step],
            "n_types": len(counter),
            "n_tokens": sum(counter.values()),
        }
        for step, counter in counters.items()
    ]


# ------------------------------------------------------------------- fits ---


def _loglog_fit(xs, ys) -> tuple[float, float]:
    """Ordinary least squares of log10(ys) on log10(xs): (slope, intercept)."""
    logx = np.log10(np.asarray(xs, dtype=float))
    logy = np.log10(np.asarray(ys, dtype=float))
    slope, intercept = np.polyfit(logx, logy, 1)
    return float(slope), float(intercept)


def heaps_points(docs: list[Doc], stopwords: frozenset[str]) -> list[dict]:
    """Vocabulary growth over the `nostem_stop` token stream in corpus order.

    Unstemmed is the honest stream for Heaps' law: stemming is a vocabulary
    *reduction*, and folding it in would measure the stemmer, not the corpus.
    """
    stream = [t for doc in docs for t in analyze(doc.text, stopwords, stem=False)]
    total = len(stream)
    if total == 0:
        return []

    start = min(HEAPS_START, total)
    if total > start:
        raw = np.geomspace(start, total, HEAPS_POINTS)
    else:
        raw = np.array([total], dtype=float)
    checkpoints = sorted({int(round(v)) for v in raw} | {total})
    checkpoints = [n for n in checkpoints if 0 < n <= total]

    points: list[dict] = []
    seen: set[str] = set()
    wanted = iter(checkpoints)
    target = next(wanted, None)
    for n, token in enumerate(stream, start=1):
        seen.add(token)
        while target is not None and n == target:
            points.append({"n_tokens": n, "n_types": len(seen)})
            target = next(wanted, None)
    return points


def heaps_fit(points: list[dict]) -> dict:
    """Heaps' `V = k * N ** beta`, fitted by least squares in log-log space."""
    usable = [p for p in points if p["n_tokens"] > 0 and p["n_types"] > 0]
    if len(usable) < 2:
        return {"k": 0.0, "beta": 0.0}
    slope, intercept = _loglog_fit(
        [p["n_tokens"] for p in usable], [p["n_types"] for p in usable]
    )
    return {"k": round6(10.0**intercept), "beta": round6(slope)}


def _ranked_terms(index: Index) -> list[tuple[str, int]]:
    """(term, collection frequency) by frequency desc, then term ascending."""
    cfs = [(term, sum(tf for _, tf in plist)) for term, plist in index.postings.items()]
    cfs.sort(key=lambda pair: (-pair[1], pair[0]))
    return cfs


def zipf_points(index: Index) -> list[dict]:
    """Rank-frequency samples: ranks 1..10, then log-spaced to the vocabulary."""
    ranked = _ranked_terms(index)
    vocab = len(ranked)
    if vocab == 0:
        return []
    head = list(range(1, min(ZIPF_HEAD, vocab) + 1))
    tail: list[int] = []
    if vocab > ZIPF_HEAD + 1:
        spaced = np.geomspace(ZIPF_HEAD + 1, vocab, ZIPF_TAIL_POINTS)
        tail = [int(round(v)) for v in spaced]
    ranks = sorted({r for r in head + tail if 1 <= r <= vocab})
    return [
        {"rank": rank, "freq": ranked[rank - 1][1], "term": ranked[rank - 1][0]}
        for rank in ranks
    ]


def zipf_fit(index: Index) -> dict:
    """Least squares of log10(freq) on log10(rank) over the first 1000 ranks."""
    ranked = _ranked_terms(index)[:ZIPF_FIT_RANKS]
    if len(ranked) < 2:
        return {"slope": 0.0, "intercept": 0.0}
    ranks = range(1, len(ranked) + 1)
    slope, intercept = _loglog_fit(list(ranks), [cf for _, cf in ranked])
    return {"slope": round6(slope), "intercept": round6(intercept)}


# -------------------------------------------------------------- term lists ---


def top_terms(index: Index, n: int = 25) -> list[dict]:
    """The `n` most frequent terms with their document and collection counts."""
    return [
        {"term": term, "df": index.df(term), "cf": cf}
        for term, cf in _ranked_terms(index)[:n]
    ]


def stem_groups(
    docs: list[Doc],
    stopwords: frozenset[str],
    index_stem: Index,
    n_groups: int = N_STEM_GROUPS,
    n_words: int = N_STEM_WORDS,
) -> list[dict]:
    """The surface forms the stemmer collapses, largest conflation first.

    Words are the unstemmed stopworded vocabulary; `cf` per group comes from
    the stemmed index, so the group total matches the `stem_stop` term counts
    exactly.
    """
    words = Counter()
    for doc in docs:
        words.update(analyze(doc.text, stopwords, stem=False))

    groups: dict[str, set[str]] = defaultdict(set)
    for word in words:
        groups[porter_stem(word)].add(word)

    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), -index_stem.cf(item[0]), item[0]),
    )
    rows = []
    for stem_form, members in ordered[:n_groups]:
        best = sorted(members, key=lambda w: (-words[w], w))[:n_words]
        rows.append(
            {
                "stem": stem_form,
                "n_words": len(members),
                "words": [{"word": w, "cf": words[w]} for w in best],
            }
        )
    return rows


def longest(index: Index, n: int = 20) -> list[dict]:
    """The terms with the widest posting lists."""
    dfs = [(term, len(plist)) for term, plist in index.postings.items()]
    dfs.sort(key=lambda pair: (-pair[1], pair[0]))
    return [{"term": term, "df": df} for term, df in dfs[:n]]


# ------------------------------------------------------------- histograms ---


def _histogram(values, bins: list[int]) -> list[int]:
    """Counts per closed-open bin; the last bin catches the whole tail."""
    counts = [0] * len(bins)
    edges = np.asarray(bins)
    for value in values:
        slot = int(np.searchsorted(edges, value, side="right")) - 1
        counts[max(slot, 0)] += 1
    return counts


def posting_hist(index: Index) -> dict:
    """How many terms have a posting list of each length."""
    lengths = [len(plist) for plist in index.postings.values()]
    return {"bins": list(POSTING_BINS), "counts": _histogram(lengths, POSTING_BINS)}


def posting_summary(index: Index) -> dict:
    """Posting-list totals, plus the share of terms seen in exactly one doc."""
    lengths = np.array([len(p) for p in index.postings.values()], dtype=float)
    n_terms = int(lengths.size)
    if n_terms == 0:
        return {
            "n_terms": 0,
            "n_postings": 0,
            "mean_len": 0.0,
            "median_len": 0.0,
            "max_len": 0,
            "df1_share": 0.0,
        }
    return {
        "n_terms": n_terms,
        "n_postings": int(lengths.sum()),
        "mean_len": round6(lengths.mean()),
        "median_len": round6(np.median(lengths)),
        "max_len": int(lengths.max()),
        "df1_share": round6(float((lengths == 1).sum()) / n_terms),
    }


def doc_len_hist(index: Index) -> dict:
    """How many documents fall in each length band, in indexed terms."""
    return {
        "bins": list(DOC_LEN_BINS),
        "counts": _histogram(index.doc_len, DOC_LEN_BINS),
    }


def doc_len_summary(index: Index) -> dict:
    """Mean, median, deciles and maximum document length, in indexed terms."""
    lengths = np.array(index.doc_len, dtype=float)
    if lengths.size == 0:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "max": 0}
    return {
        "mean": round6(lengths.mean()),
        "median": round6(np.median(lengths)),
        "p10": round6(np.percentile(lengths, 10)),
        "p90": round6(np.percentile(lengths, 90)),
        "max": int(lengths.max()),
    }


# ---------------------------------------------------------------- writers ---


def write_corpus(
    docs: list[Doc],
    index_stem_stop: Index,
    n_files: int,
    out_dir: Path = OUT_DIR,
) -> Path:
    """`corpus.json`: corpus size, date span and the length distribution."""
    dates = [doc.date for doc in docs]
    obj = {
        "schema": "corpus.v1",
        "generated_at": now_iso(),
        "n_docs": len(docs),
        "n_files": n_files,
        "date_min": min(dates) if dates else "",
        "date_max": max(dates) if dates else "",
        "total_tokens_alpha": sum(len(tokenize(doc.text)) for doc in docs),
        "doc_len": doc_len_hist(index_stem_stop),
        "doc_len_summary": doc_len_summary(index_stem_stop),
    }
    path = Path(out_dir) / "corpus.json"
    write_json(path, obj)
    return path


def write_vocab(
    docs: list[Doc],
    stopwords: frozenset[str],
    indexes: dict[str, Index],
    out_dir: Path = OUT_DIR,
) -> Path:
    """`vocab.json`: the funnel, the four treatments, Heaps and Zipf."""
    stem_stop = indexes["stem_stop"]
    points = heaps_points(docs, stopwords)
    obj = {
        "schema": "vocab.v1",
        "generated_at": now_iso(),
        "funnel": funnel(docs, stopwords),
        "treatments": {
            key: {
                "n_types": len(indexes[key].postings),
                "n_tokens": indexes[key].total_tokens,
            }
            for key in TREATMENTS
        },
        "top_terms": top_terms(stem_stop),
        "stem_groups": stem_groups(docs, stopwords, stem_stop),
        "heaps": points,
        "heaps_fit": heaps_fit(points),
        "zipf": zipf_points(stem_stop),
        "zipf_fit": zipf_fit(stem_stop),
    }
    path = Path(out_dir) / "vocab.json"
    write_json(path, obj)
    return path


def write_postings(
    index_stem_stop: Index,
    size_bytes: dict,
    out_dir: Path = OUT_DIR,
) -> Path:
    """`postings.json`: the posting-length distribution and the index sizes."""
    obj = {
        "schema": "postings.v1",
        "generated_at": now_iso(),
        "hist": posting_hist(index_stem_stop),
        "summary": posting_summary(index_stem_stop),
        "longest": longest(index_stem_stop),
        "size_bytes": dict(size_bytes),
    }
    path = Path(out_dir) / "postings.json"
    write_json(path, obj)
    return path
