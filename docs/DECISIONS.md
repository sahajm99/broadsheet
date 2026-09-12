# Decisions

One line of why for every taste decision. The user's standing instructions from the
CardioLens build apply: commits authored by `sahajm99` with no co-author trailer,
skip thoroughness where it does not change the result, finish.

## Data and licensing

| # | Decision | Why |
|---|----------|-----|
| D1 | The FT article text is never committed or served. `data/raw/ft911/` is git-ignored; provenance is fixed by `data/raw/SOURCE.md` and `SHA256SUMS`, and `meta.json` repeats the hashes. | The Financial Times files are TREC disk 4 content licensed to the course, not to the public. |
| D2 | Derived artefacts are shipped: stems with posting lists and term frequencies, plus headline, date and document number per article. | A stemmed bag of words cannot reconstruct an article; a headline with a date is a citation, which is what a search result is. |
| D3 | Evaluate against the NIST Robust 2004 qrels filtered to `FT911-`, not only the four course topics. | The course file held four topics with judgments; the public file covers 250 topics over the same collection and gives 71 evaluable topics instead of 16. Same judges, same documents. |
| D4 | Topics with no relevant document inside the slice are excluded from every mean and the count is printed. | An AP of zero for a topic that could not be answered measures the slice, not the engine. |
| D5 | Unjudged documents count as not relevant, and `judged@10` (share of the top ten that any judge looked at) is reported beside every metric. | Standard TREC convention; the second number shows how much of each ranking the judgments can even see. |
| D6 | The course stopword list (523 words) is used unchanged and committed. | The rebuild measures the course's pipeline; changing the list would change the subject. |

## Text processing

| # | Decision | Why |
|---|----------|-----|
| X1 | Tokeniser follows the course rules: lowercase, split on non-alphanumerics, discard any token containing a digit. | Assignment 1 specified them; the vocabulary funnel shows what each rule costs. |
| X2 | Own Porter implementation in Python and TypeScript, both tested against Martin Porter's 23,531-word vector file. | One algorithm in two languages must agree or the browser demo would rank differently from the published numbers; NLTK's default mode adds extensions that no JavaScript port matches. |
| X3 | Four index treatments: stem x stopwords on or off. | The vocabulary and evaluation sections both ask what stemming and stopwording buy. |

## Ranking

| # | Decision | Why |
|---|----------|-----|
| R1 | `tfidf_raw` reproduces the course weighting exactly: raw tf x log10(N/df) on both document and query, cosine. | It is the baseline the story compares against. |
| R2 | `tfidf_log` uses (1 + log10 tf) x idf. `bm25` uses k1 = 1.2, b = 0.75 as the published default. | Standard textbook variants; the defaults are named so the grid can show what tuning changes. |
| R3 | The BM25 grid (k1 in {0.6, 0.9, 1.2, 1.5, 2.0}, b in {0.0, 0.25, 0.5, 0.75, 1.0}) is reported with the sentence that it was tuned on the same 71 topics it is scored on. | Showing the surface is useful; pretending the best cell is a generalising number is not. |
| R4 | Queries are the topic `title` by default; `title+desc` and `title+desc+narr` are extra runs. | The course claimed longer queries help; the page tests it. |
| R5 | No phrase, proximity, pseudo-relevance feedback or learned ranking. | The subject is the course's engine, measured honestly, not a better engine. |

## Statistics

| # | Decision | Why |
|---|----------|-----|
| S1 | Percentile bootstrap over topics, 10,000 resamples, seed 20260912, for every mean and every paired difference. | 71 topics with skewed per-topic AP; a t interval would be wrong-shaped and a p-value would invite the wrong reading. |
| S2 | nDCG uses binary gains and log2 discount; AP uses the number of relevant documents in the slice as denominator. | Conventional trec_eval definitions so the numbers are comparable to published ones. |
| S3 | Interpolated 11-point precision is averaged across topics per run. | The standard PR curve for a batch evaluation. |

## Site and design

| # | Decision | Why |
|---|----------|-----|
| W1 | The hero is the search box, with results and an explain panel that lists every matched stem with tf, df, idf and its share of the score. | The most characteristic thing about a search engine is searching; the arithmetic is what every engine hides. |
| W2 | The index is one gzip JSON file fetched on the first query and inflated with `DecompressionStream`; its size is printed from `meta.json`. | GitHub Pages does not compress unknown types; the browser API is in every current browser; the visitor is told the cost before paying it. |
| W3 | Browser rankers are BM25 (default) and the course tf-idf cosine; a vitest parity test compares browser top-10s to the pipeline's for every topic title. | Two implementations that disagree would undermine every number on the page. |
| W4 | Visual identity: FT salmon paper as the page ground in light mode, near-black ink, one deep teal accent; IBM Plex Sans for everything, IBM Plex Mono only for stems, document numbers and scores. | The subject is the pink paper; mono is reserved for the machine's view of text, which is what the page is about. Distinct from CardioLens (white paper, serif essay). |
| W5 | Chart palette is the validated dataviz reference palette (blue, orange, aqua, yellow; dark variants) re-validated against the salmon and dark surfaces; sequential teal ramp for the grid heat map. | The validator, not taste, decides. |
| W6 | Same chrome contract as CardioLens: CSS tokens in both dark scopes, Plotly template from CSS variables, `Plotly.react` on theme toggle, lazy figures, `data-stat` fills from `claims.json`, `role="img"` and a `<details>` table per chart. | Proven on the last project; nothing here needs a different mechanism. |
| W7 | No numeral describing the data is hand-typed in the site; every one goes through `data-stat` or a title template. | Numbers that drift from the pipeline would break the honesty thesis. |

## Process

| # | Decision | Why |
|---|----------|-----|
| P1 | CI runs pytest, vitest, typecheck, build and deploy; it does not run the pipeline or execute the notebook. | The corpus cannot be in CI (D1). Committed JSON is validated structurally and by the parity fixtures instead. |
| P2 | Task reviews are dispatched only for the four pipeline tasks whose defects would poison every number (tokeniser and stemmer, index, rankers, evaluation); site tasks get one controller QA pass. | The user asked to skip thoroughness that does not change the result. |
| P3 | The course notebooks are kept under `course/` unchanged for reference, and the page says what changed since them. | The rebuild's claim is that it fixed and measured the course engine; the reader can check. |
| P4 | Portfolio: add a `broadsheet` entry with category `data-engineering`, status live. | It is an index-and-evaluate pipeline; the existing category fits and needs no union change. |
