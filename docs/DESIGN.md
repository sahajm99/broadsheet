# Broadsheet: design

A search engine built from scratch over 5,368 Financial Times articles from 1991,
evaluated the way the TREC community evaluates search, and published as one page
that lets the visitor run queries and see exactly why each result ranked where it did.

Rebuilt from the CSCE 5200 Information Retrieval and Web Search project (UNT, Fall
2024, solo): a tokenizer with Porter stemming (phase 1), forward and inverted
indexes (phase 2), a vector-space query processor (phase 3).

## The problem

The course submission worked as a pipeline but was never measured. Its output file
carried internal ids (`Doc_5292`) instead of document numbers, so it could not be
scored against the relevance judgments, and it used raw term counts as the term
weight. The reports describe precision and recall in the abstract. A hiring manager
who opens that repo sees three notebooks and no number.

The rebuild answers three questions with numbers and intervals:

1. What does the text look like once a machine has tokenised, stopworded and stemmed
   it? (vocabulary, Zipf, Heaps, posting lists)
2. Does the engine find the right documents, and which weighting finds more of them?
   (tf-idf cosine as submitted, log-tf cosine, BM25; title vs longer queries)
3. What does an evaluation on 2.5 percent of a collection actually let you conclude?
   (the sparse-judgment problem, stated rather than hidden)

## Audience and job

A hiring manager or engineer with two minutes. The page must let them type a query,
see results with the score broken down per term, and scroll into an evaluation that
reads as an argument with stated uncertainty. Nothing on the page asks them to trust
a number they cannot trace to a JSON file the pipeline wrote.

## Data

- Corpus: `ft911_1` to `ft911_15`, 5,368 `<DOC>` records (`DOCNO`, `DATE`,
  `HEADLINE`, `TEXT`), 14.5 MB. This is TREC disk 4 material licensed to the course;
  the article text is **not redistributed**. The raw files live in `data/raw/ft911/`,
  which is git-ignored; `data/raw/SOURCE.md` and `SHA256SUMS` fix the provenance.
- Judgments: NIST Robust 2004 qrels (public), filtered to `FT911-` document numbers:
  3,019 judgment lines, 1,844 judged documents, 186 relevant (document, topic)
  pairs, 71 topics with at least one relevant document in this slice, 36 of them
  with exactly one. Topics: NIST files 301-450 and 601-700 (250 topics, title,
  description, narrative). Both committed under `data/trec/` with `SOURCE.md`.
- Stopword list: the 523-word list issued by the course, committed as
  `data/stopwords.txt`.
- Porter test vectors: Martin Porter's `voc.txt` and `output.txt` (23,531 words),
  committed under `tests/golden/porter/` so the Python and TypeScript stemmers are
  both checked against the canonical output.

What the browser downloads: JSON aggregates under 200 KB each, plus one on-demand
compressed search index (`search/index.json.gz`, target under 2 MB) that carries
stems, posting lists with term frequencies, and document headlines, dates and
numbers. No article text ships.

## Pipeline

`python -m pipeline` (Python 3.12, uv, no pandas needed; numpy for the bootstrap):

1. `parse.py`: SGML records to `Doc(docno, date, headline, text)`.
2. `tokenize.py`: lowercase, split on non-alphanumerics, drop any token containing a
   digit (course rule), stopword filter, Porter stem.
3. `porter.py`: the 1980 algorithm as implemented on tartarus.org, own code.
4. `index.py`: forward and inverted indexes for four treatments (stem/nostem x
   stop/nostop); statistics for the vocabulary story.
5. `rank.py`: `tfidf_raw` (course weighting: raw tf x log10(N/df), cosine),
   `tfidf_log` ((1 + log10 tf) x idf, cosine), `bm25` (k1 = 1.2, b = 0.75) plus a
   k1 x b grid.
6. `topics.py`: parse the 250 topics; `qrels` filtered to the corpus.
7. `evaluate.py`: AP, P@10, R-precision, nDCG@10, recall@100, judged@10,
   interpolated 11-point precision; percentile bootstrap over topics (10,000
   resamples, seed 20260912) for means and paired differences.
8. Writers for every JSON file in `docs/superpowers/plans/2026-09-12-broadsheet.md`
   (the contract), then `claims.json` and `meta.json` last.

## Site

Vite + TypeScript + `plotly.js-cartesian-dist-min`, no framework, no CDN, fonts
self-hosted. Same chrome as CardioLens (theme tokens, lazy figures, `data-stat`
fills, `role="img"` plus a `<details>` table per chart), adapted rather than copied
where the subject differs.

Sections, in reading order:

1. **Search** (hero): a query box over the 5,368 articles, BM25 by default with a
   switch to the course's tf-idf cosine, ten results with headline, date and
   document number, and an "explain" panel per result listing each matched stem
   with tf, df, idf and its share of the score. Example query chips. Runs entirely
   in the browser after one index download whose size is printed from `meta.json`.
2. **From text to terms**: vocabulary funnel (raw types, alphabetic, minus
   stopwords, stemmed), Heaps' law curve, Zipf plot, stem collapse examples.
3. **The index**: posting-list length distribution, document length distribution,
   the heaviest stems, index size.
4. **Does it find the right articles?**: how evaluation works here (judgments,
   slice, the 71 topics), MAP / P@10 / nDCG@10 per ranker with bootstrap
   intervals, per-topic AP distribution, paired BM25 minus tf-idf differences,
   interpolated precision-recall curves, query-field effect (title, +description,
   +narrative), BM25 k1 x b grid with the "tuned on the test set" caveat.
5. **What changed since the course version**: internal ids fixed, log-tf and BM25
   added, the evaluation actually run, stemmer parity tested in two languages.
6. **What this cannot tell you**: the limits list.

The memorable element is the explain panel: the visitor sees the arithmetic of a
ranking, which is the thing every search engine hides.

## Verification

- `uv run pytest`: Porter golden (23,531 words), tokenizer rules, parser and index
  on a hand-written 12-document fixture with hand-computed df and tf, BM25 and
  cosine values on a 3-document example computed by hand, metric unit tests on
  known cases, bootstrap determinism, structural checks over every committed JSON.
- Site: `vitest` runs the TypeScript Porter against the same golden file and checks
  the browser BM25 top-10 for every topic title against a fixture the Python
  pipeline wrote (`tests/golden/top10_bm25_title.json`).
- CI on every push: pytest, site typecheck and build, vitest, Pages deploy from
  main. The pipeline itself cannot run in CI because the corpus is not
  redistributable; `meta.json` records the corpus SHA-256s so a local rerun is
  checkable.
- Live QA with headless Chrome: every figure renders in light, dark and at 400 px;
  a query returns results with an explain panel; no console errors.
