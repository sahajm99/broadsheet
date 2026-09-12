# Broadsheet

[![ci](https://github.com/sahajm99/broadsheet/actions/workflows/ci.yml/badge.svg)](https://github.com/sahajm99/broadsheet/actions/workflows/ci.yml)

**Live: [sahajm99.github.io/broadsheet](https://sahajm99.github.io/broadsheet/)**

A search engine built from scratch over 5,368 Financial Times articles from
April and May 1991, and an honest measurement of how well it works. The
pipeline is a Python 3.12 program with no retrieval library in it: its own
tokenizer, its own Porter stemmer, its own inverted index, and three rankers,
the course's tf-idf cosine, a log-tf variant and BM25. It runs all three over
the 250 TREC Robust 2004 topics, scores them against NIST's relevance
judgments with percentile-bootstrap intervals on every mean and every paired
difference, and writes every number it found to JSON. The site reads those JSON
files, so no numeral in the page's prose is typed by hand. The same index ships
to the browser as one gzip, which means the search box at the top of the page
is the engine the charts below it describe, not a demonstration of a different
one.

## What it shows

- **BM25 beats the weighting the course submitted, by a margin worth stating
  with an interval.** Mean average precision over title queries is 0.370 for
  BM25 against 0.291 for raw tf-idf cosine, a paired difference of +0.078 with
  a 95% bootstrap interval of [+0.017, +0.141]. Topic by topic, BM25 wins 40,
  loses 17 and ties 14.
- **The evaluation rests on a small, uneven base, and says so.** Filtering the
  Robust 2004 judgments to this one-month slice leaves 71 evaluable topics, 186
  relevant article-topic pairs and 1,844 judged articles, and 36 of those topics
  have exactly one relevant article, so their average precision is one divided
  by a single rank.
- **Roughly half of every ranking is scored against a blank.** judged@10 for
  BM25 is 0.47, meaning fewer than half the articles in a typical top ten were
  ever looked at by an assessor. Unjudged counts as not relevant, so every mean
  reported here is a lower bound of unknown tightness.
- **Text processing is most of the index.** The course rules drop 49.9% of all
  tokens as stopwords, and the Porter stemmer folds the remaining 47,506
  distinct word forms into 32,645 stems.
- **Most of the vocabulary is almost never useful.** 44.1% of stems occur in
  exactly one article, which is a large, cheap tail of terms that no query will
  ever reach, sitting beside a handful of stems that appear in thousands of
  articles and discriminate nothing.
- **Neither fitted exponent is the textbook one.** Vocabulary grows as
  tokens^0.63, faster than the square-root rule of thumb, and collection
  frequency falls as rank^-0.70, flatter than the classic one over rank. Both
  are straight fits through curves that bend, and the page says so rather than
  quoting the law.
- **Tuning BM25 on this slice overfits it, visibly.** A k1 by b grid reaches
  MAP 0.417 at b = 0, against 0.370 at the published defaults. The best cell
  switches length normalisation off entirely, and the gain is larger than
  BM25's whole advantage over the course weighting, on the same 71 topics it
  was tuned on.
- **The whole engine fits in a 1.25 MB download.** Stems, posting lists with
  term frequencies, headlines and dates ship as one gzip that the page fetches
  on the first query and ranks locally, with an explain panel showing the tf,
  df, idf and score share behind every result.

## Run it

```bash
uv sync                                   # Python 3.12 + numpy + pytest
# place the 15 corpus files in data/raw/ft911/ - see data/raw/SOURCE.md
uv run python -m pipeline                 # writes site/public/data/*.json (~20s)
uv run pytest                             # the pipeline's tests

cd site && npm ci && npm run dev          # the page, at http://localhost:5173
npm test                                  # stemmer and browser-ranker parity
```

`uv run python -m pipeline` is the only way any published number is produced.
It parses the corpus, builds four index treatments (stemmed or not, stopwords
removed or not), runs the evaluation matrix, and writes `site/public/data/` plus
`site/public/data/search/index.json.gz`, the index the browser downloads. Two
runs over the same corpus produce byte-identical files apart from the
`generated_at` timestamp, so a diff under `site/public/data/` always means a
real change.

The corpus is not in this repository (see below), so `python -m pipeline` cannot
run in CI. `uv run pytest` works either way: the unit tests use a small
hand-checked fixture corpus, and `tests/test_outputs.py` validates the committed
output files structurally without needing the corpus to rebuild them.

## How it is checked

- **One stemmer, two languages, one golden file.** The Python stemmer and the
  TypeScript one are both run over Martin Porter's official vocabulary file,
  23,531 words, and every output is compared to his published stems. A
  divergence in either language fails its own suite.
- **The browser ranks what the pipeline ranked.** `site/tests/parity.test.ts`
  replays the BM25 top ten for the 243 answerable TREC topic titles against a
  fixture the Python pipeline wrote, comparing document numbers and scores, and
  asserts that the remaining topics return nothing in both implementations.
- **Every committed JSON is checked structurally.** `tests/test_outputs.py`
  reads each file the pipeline writes and asserts its schema, the array lengths
  that must agree with one another, bootstrap bounds that must bracket their own
  mean, and precision-recall curves that must not rise.
- **The notebook cannot leak the corpus.** `tests/test_notebook.py` reads the
  executed notebook and fails if any cell output contains article text, so a
  re-execution that accidentally prints a document is caught before it is
  committed.
- CI runs pytest, the site typecheck, the site build and vitest on every push,
  and deploys the page from `main`.

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
  date is a citation, which is what a search result is.
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
| `tests/` | pytest suite; `tests/fixtures/mini/` is a hand-checked miniature corpus, `tests/golden/` holds Porter's official vocabulary and the parity fixtures for the browser port. |
| `data/raw/` | Where the corpus goes, plus its `SOURCE.md` and `SHA256SUMS`. Git-ignored contents. |
| `data/trec/` | The 250 TREC topics and the filtered Robust 2004 qrels. |
| `site/` | The Vite + TypeScript page. `site/public/data/` is pipeline output; `site/src/search/` is the browser's own index reader and ranker; `site/tests/` holds the parity suites. |
| `docs/` | `BRIEF.md`, `DESIGN.md`, `DECISIONS.md`, `PROGRESS.md`. |
| `course/` | The original course notebooks, unchanged, for reference (P3). |
| `notebooks/` | The executed analysis notebook. |

## Notebook

`notebooks/analysis.ipynb` is the long-form version of the argument the site
makes in six sections. It walks the same pipeline in order, from parsing the
SGML records through tokenising, stemming and indexing to the ranking and the
evaluation, and it plots each stage with matplotlib from the same committed
JSON the site reads, so the notebook and the page cannot disagree. It is
committed already executed, so it reads without a corpus and without a Python
environment, and it is where to look for the intermediate steps and the
arithmetic that the site's charts summarise. Re-executing it needs the corpus in
place and `uv run jupyter nbconvert --execute --to notebook --inplace
notebooks/analysis.ipynb`.
