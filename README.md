# Broadsheet

![ci](https://github.com/sahajm99/broadsheet/actions/workflows/ci.yml/badge.svg)

A search engine built from scratch over 5,368 Financial Times articles from
April and May 1991, and an honest measurement of how well it works. The
pipeline is a Python 3.12 program with no retrieval library in it: its own
tokenizer, its own Porter stemmer, its own inverted index, and three rankers —
the course's tf-idf cosine, a log-tf variant and BM25. It scores all three over
the 250 TREC Robust 2004 topics, evaluates them against NIST's relevance
judgments with percentile-bootstrap intervals on every mean and every paired
difference, and writes every number it found to JSON. The site reads those JSON
files; no numeral on the page is typed by hand. The same index is shipped to
the browser as one gzip, so the search box on the page is the engine described
by the charts below it, not a demonstration of a different one.

## Run it

```bash
uv sync                                   # Python 3.12 + numpy + pytest
# place the 15 corpus files in data/raw/ft911/ — see data/raw/SOURCE.md
uv run python -m pipeline                 # writes site/public/data/*.json (~20s)
uv run pytest                             # the pipeline's tests

cd site && npm ci && npm run dev          # the page, at http://localhost:5173
```

`uv run python -m pipeline` is the only way any published number is produced.
It parses the corpus, builds four index treatments (stemmed or not, stopwords
removed or not), runs the evaluation matrix, and writes `site/public/data/` plus
`site/public/data/search/index.json.gz`, the index the browser downloads. Two
runs over the same corpus produce byte-identical files apart from the
`generated_at` timestamp, so a diff under `site/public/data/` always means a
real change.

The corpus is not in this repository (see below), so `python -m pipeline` cannot
run in CI. `uv run pytest` works either way: the unit tests use a seven-document
fixture, and `tests/test_outputs.py` validates the committed output files
structurally — schemas, array lengths that must agree, bootstrap bounds that
must bracket their mean, PR curves that must not rise — without needing the
corpus to rebuild them.

## Data and licensing

The code in this repository is MIT licensed (see `LICENSE`). The data is not
mine to license, and three rules follow from that.

- **The article text is never committed and never served** (decision D1).
  `data/raw/ft911/` is git-ignored. The Financial Times files are TREC disk 4
  content licensed to the course, not to the public. What fixes provenance
  instead is `data/raw/SOURCE.md` and `data/raw/SHA256SUMS`; the pipeline
  re-verifies every hash on every run and repeats them in `meta.json`, so a
  rebuild from a different copy of the collection fails loudly rather than
  publishing different numbers under the same hashes.
- **Derived artefacts are shipped** (D2): stems with their posting lists and
  term frequencies, plus a headline, a date and a document number per article.
  A stemmed bag of words cannot reconstruct an article, and a headline with a
  date is a citation — which is what a search result is.
- **Evaluation uses the public NIST judgments** (D3): the TREC Robust 2004
  qrels filtered to `FT911-` documents, not only the four topics the course
  supplied. Same judges, same documents, 71 evaluable topics instead of 16.
  Topics with no relevant document inside this one-month slice are excluded
  from every mean and the count is printed (D4). Topics, qrels and their
  provenance are in `data/trec/`.

Every taste decision in the project has one line of reasoning in
`docs/DECISIONS.md`, and the design is in `docs/DESIGN.md`.

## Layout

| Path | What is in it |
|---|---|
| `pipeline/` | The whole engine: parser, tokenizer, Porter stemmer, index, rankers, metrics, writers. `python -m pipeline` runs `run.py`. |
| `tests/` | pytest suite; `tests/fixtures/mini/` is a seven-document corpus, `tests/golden/` holds Porter's official vocabulary and the parity fixtures for the browser port. |
| `data/raw/` | Where the corpus goes, plus its `SOURCE.md` and `SHA256SUMS`. Git-ignored contents. |
| `data/trec/` | The 250 TREC topics and the filtered Robust 2004 qrels. |
| `site/` | The Vite + TypeScript page. `site/public/data/` is pipeline output; `site/src/search/` is the browser's own index reader and ranker. |
| `docs/` | `BRIEF.md`, `DESIGN.md`, `DECISIONS.md`, `PROGRESS.md`. |
| `course/` | The original course notebooks, unchanged, for reference (P3). |
| `notebooks/` | The executed analysis notebook. |
