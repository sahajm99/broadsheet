# Broadsheet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A from-scratch search engine over 5,368 Financial Times 1991 articles, evaluated against NIST Robust 2004 judgments, published as an interactive page at `https://sahajm99.github.io/broadsheet/` with a browser-side search demo and twelve charts, every number traceable to JSON written by `python -m pipeline`.

**Architecture:** A Python 3.12 pipeline (`pipeline/`) parses the corpus, builds four index treatments, runs three rankers over 250 topics, evaluates against qrels with bootstrap intervals, and writes JSON aggregates to `site/public/data/` plus one gzip search index. A Vite + TypeScript site renders the story with Plotly and runs BM25 and tf-idf cosine in the browser using its own Porter port, checked for parity against pipeline fixtures.

**Tech Stack:** Python 3.12, numpy, pytest, uv; Vite 8, TypeScript 6 strict, `plotly.js-cartesian-dist-min`, vitest, `@fontsource-variable/ibm-plex-sans`, `@fontsource/ibm-plex-mono`; GitHub Actions + Pages.

**Spec:** `docs/DESIGN.md` and `docs/DECISIONS.md` (binding). Repo root: `C:\Users\sahaj\OneDrive\Desktop\Experiments\projects\active\broadsheet`.

## Global Constraints

- Every commit is authored as `sahajm99 <64627746+sahajm99@users.noreply.github.com>` (repo-local git config already set). **No `Co-Authored-By` trailer of any kind.** Do not change the git identity.
- Python: run with `uv run` (bare `python` is not on PATH). Tests: `uv run pytest`. Write files with LF line endings and UTF-8.
- `data/raw/ft911/` is git-ignored and must never be committed or copied into `site/`. No article `TEXT` content may appear in any committed or served file. Headlines, dates and document numbers may.
- Every JSON under `site/public/data/` except `search/index.json.gz` is under 200,000 bytes, written with `indent=1`, `allow_nan=False`, sorted keys off, and carries a top-level `"schema"` string. Floats rounded to 6 decimals.
- Determinism: bootstrap seed `20260912`, `10000` resamples, `numpy.random.default_rng(seed)`. Running the pipeline twice yields byte-identical output except `generated_at` lines.
- Ties in ranking break by ascending document index (corpus order, `FT911-1` first). Sorting is by `(-score, doc_idx)`.
- Site: `base: "/broadsheet/"`; every fetch uses `import.meta.env.BASE_URL`; no CDN, no framework; `strict: true` TypeScript; no hand-typed numeral describing the data in `site/index.html` or chart titles (use `data-stat` and title templates).
- Course notebooks under `course/` are read-only reference.
- The source folder `C:\Users\sahaj\OneDrive\Desktop\UNT_Course_Work\` is read-only; never modify it.

---

## Repository layout

```
broadsheet/
  data/raw/ft911/            git-ignored corpus (15 files), SOURCE.md, SHA256SUMS committed
  data/trec/                 topics.301-450.601-700.txt, qrels.ft911.txt, SOURCE.md
  data/stopwords.txt         523 course stopwords
  course/                    the three course notebooks, unchanged
  pipeline/                  constants.py io.py parse.py tokenize.py porter.py index.py
                             vocab_stats.py topics.py rank.py evaluate.py runs.py
                             search_export.py claims.py run.py __main__.py
  tests/                     test_*.py, fixtures/mini/ft_mini_1, golden/porter/{voc,output}.txt,
                             golden/top10_bm25_title.json, golden/top10_tfidf_raw_title.json,
                             golden/topic_titles.json
  site/                      Vite app; public/data/*.json; public/data/search/index.json.gz
  notebooks/analysis.ipynb
  .github/workflows/ci.yml
  docs/DESIGN.md DECISIONS.md PROGRESS.md superpowers/plans/
```

## JSON contract (the site reads exactly these)

All files in `site/public/data/`. Floats to 6 dp. `lo`/`hi` are 95% percentile-bootstrap bounds over topics.

| File | Shape |
|---|---|
| `corpus.json` | `{schema:"corpus.v1", generated_at, n_docs, n_files, date_min:"1991-05-14", date_max, total_tokens_alpha, doc_len:{bins:[int edges], counts:[int]}, doc_len_summary:{mean, median, p10, p90, max}}` where doc_len is in stemmed, stopworded terms; bins are `[0,50,100,150,200,300,400,500,750,1000,1500,2000,3000,5000]` closed-open with the last bin catching the rest. |
| `vocab.json` | `{schema:"vocab.v1", generated_at, funnel:[{step, label, n_types, n_tokens}] (4 rows in order raw_alpha, minus_digits, minus_stopwords, stemmed), treatments:{stem_stop:{n_types,n_tokens}, stem_nostop, nostem_stop, nostem_nostop}, top_terms:[{term, df, cf}] (25, by cf desc, stem_stop), stem_groups:[{stem, n_words, words:[{word, cf}] (up to 8 by cf desc)}] (12 largest groups by n_words, ties by cf), heaps:[{n_tokens, n_types}] (about 40 log-spaced checkpoints over the nostem_stop stream plus the final point), heaps_fit:{k, beta} (least squares on log-log), zipf:[{rank, freq, term}] (ranks 1..10, then log-spaced to vocab size, about 60 points, stem_stop, cf), zipf_fit:{slope, intercept} (least squares on log10 rank vs log10 freq over ranks 1..1000)}` |
| `postings.json` | `{schema:"postings.v1", generated_at, hist:{bins:[1,2,3,5,9,17,33,65,129,257,513,1025,2049,4097], counts:[int]}, summary:{n_terms, n_postings, mean_len, median_len, max_len, df1_share}, longest:[{term, df}] (20 by df desc), size_bytes:{index_gz, index_json}}` for stem_stop. |
| `runs.json` | `{schema:"runs.v1", generated_at, n_topics, n_topics_all, n_rel_total, n_judged_docs, n_single_rel_topics, rankers:[{key, label}], fields:[{key, label}], runs:[{key:"bm25|title", ranker:"bm25", field:"title", metrics:{map:{mean,lo,hi}, p10:{...}, ndcg10:{...}, rprec:{...}, recall100:{...}, judged10:{...}}}]}` 9 runs: rankers `tfidf_raw`, `tfidf_log`, `bm25` x fields `title`, `title_desc`, `title_desc_narr`, all on stem_stop. |
| `per_topic.json` | `{schema:"per_topic.v1", generated_at, topics:[int] (71 ascending), ap:{run_key:[float]}}` |
| `comparisons.json` | `{schema:"comparisons.v1", generated_at, pairs:[{key, a, b, label, mean_diff, lo, hi, wins, losses, ties, per_topic:[{num, diff}]}]}` pairs (a minus b): `bm25|title` vs `tfidf_raw|title`; `tfidf_log|title` vs `tfidf_raw|title`; `bm25|title` vs `tfidf_log|title`; `bm25|title_desc` vs `bm25|title`; `bm25|title_desc_narr` vs `bm25|title`; `tfidf_raw|title_desc_narr` vs `tfidf_raw|title`; `treat:stem_stop` vs `treat:nostem_stop`; `treat:stem_stop` vs `treat:stem_nostop` (treatment runs are bm25 title). Ties: `abs(diff) < 1e-9`. |
| `pr_curves.json` | `{schema:"pr_curves.v1", generated_at, recall_levels:[0,0.1,...,1.0], curves:{run_key:[11 floats]}}` mean interpolated precision over the 71 topics. |
| `topics.json` | `{schema:"topics.v1", generated_at, rows:[{num, title, n_rel, n_judged, ap:{run_key:float} (9 keys), best_run}]}` 71 rows ascending by num. |
| `bm25_grid.json` | `{schema:"bm25_grid.v1", generated_at, k1:[0.6,0.9,1.2,1.5,2.0], b:[0.0,0.25,0.5,0.75,1.0], map:[[float]] (rows k1, cols b), best:{k1,b,map}, default:{k1:1.2,b:0.75,map}, caveat:"tuned and scored on the same 71 topics"}` |
| `treatments.json` | `{schema:"treatments.v1", generated_at, ranker:"bm25", field:"title", rows:[{key, label, n_types, map:{mean,lo,hi}, p10:{mean,lo,hi}}]}` 4 rows in order stem_stop, nostem_stop, stem_nostop, nostem_nostop. |
| `claims.json` | `{schema:"claims.v1", generated_at, values:{...}}` keys listed in Task 5. |
| `meta.json` | `{schema:"meta.v1", generated_at, corpus_files:[{name, sha256}], n_docs, index_gz_bytes, index_gz_mb, seed, n_boot, python, numpy}` |
| `search/index.json.gz` | gzip (level 9) of `{schema:"search_index.v1", n_docs, avgdl, k1:1.2, b:0.75, docs:[{no, d, h}], doc_len:[int], terms:{stem:[gap,tf,gap,tf,...]}}` stem_stop; gaps are doc-index deltas starting from 0; `d` is `YYYY-MM-DD`; `h` is the headline with the `FT  14 MAY 91 / ` prefix removed and whitespace collapsed, max 200 chars. |
| `stopwords.json` | written to `site/src/search/stopwords.json` as `[string]` sorted. |

Golden fixtures written by the pipeline into `tests/golden/`: `topic_titles.json` (`{"301":"International Organized Crime", ...}` 250 entries), `top10_bm25_title.json` and `top10_tfidf_raw_title.json` (`{"301":[{no, score}, ...10]}` for every topic that has at least one result; scores 6 dp).

---

### Task 1: Parser, tokenizer, Porter stemmer

**Files:**
- Create: `pyproject.toml`, `.python-version` (`3.12`), `.gitignore`, `pipeline/__init__.py`, `pipeline/constants.py`, `pipeline/parse.py`, `pipeline/tokenize.py`, `pipeline/porter.py`, `tests/fixtures/mini/ft_mini_1`, `tests/test_parse.py`, `tests/test_tokenize.py`, `tests/test_porter.py`, `data/raw/SOURCE.md`, `data/trec/SOURCE.md`
- Existing: `data/raw/ft911/*` (corpus), `data/stopwords.txt`, `tests/golden/porter/voc.txt`, `tests/golden/porter/output.txt`

**Interfaces (produces):**
```python
# pipeline/constants.py
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw/ft911"; STOPWORDS_PATH = ROOT / "data/stopwords.txt"
TOPICS_PATH = ROOT / "data/trec/topics.301-450.601-700.txt"; QRELS_PATH = ROOT / "data/trec/qrels.ft911.txt"
OUT_DIR = ROOT / "site/public/data"; GOLDEN_DIR = ROOT / "tests/golden"; STOPWORDS_JSON = ROOT / "site/src/search/stopwords.json"
SEED = 20260912; N_BOOT = 10_000; BM25_K1 = 1.2; BM25_B = 0.75
GRID_K1 = [0.6, 0.9, 1.2, 1.5, 2.0]; GRID_B = [0.0, 0.25, 0.5, 0.75, 1.0]
SIZE_LIMIT = 200_000; TOP_N = 1000
TREATMENTS = {"stem_stop": (True, True), "nostem_stop": (False, True), "stem_nostop": (True, False), "nostem_nostop": (False, False)}  # (stem, stop)
TREATMENT_LABELS = {"stem_stop": "Stemmed, stopwords removed", "nostem_stop": "Unstemmed, stopwords removed", "stem_nostop": "Stemmed, stopwords kept", "nostem_nostop": "Unstemmed, stopwords kept"}

# pipeline/parse.py
@dataclass(frozen=True) class Doc: docno: str; date: str; headline: str; text: str   # date "YYYY-MM-DD", headline cleaned
def parse_file(path: Path) -> list[Doc]
def load_corpus(raw_dir: Path = RAW_DIR) -> list[Doc]   # sorted by int(docno.split("-")[1]); raises FileNotFoundError with a message naming data/raw/SOURCE.md if the dir is missing or empty
def clean_headline(raw: str) -> str   # strip r"^\s*FT\s+\d{1,2}\s+[A-Z]{3}\s+\d{2}\s*/\s*", collapse whitespace, strip, cut to 200 chars
def iso_date(yymmdd: str) -> str     # "910514" -> "1991-05-14"; two-digit years 90-99 -> 19xx

# pipeline/tokenize.py
def load_stopwords(path: Path = STOPWORDS_PATH) -> frozenset[str]   # strip, lowercase, skip blanks
def tokenize(text: str) -> list[str]          # re.findall(r"[a-z0-9]+", text.lower()) then drop tokens containing any digit
def analyze(text: str, stopwords: frozenset[str] | None, stem: bool) -> list[str]   # tokenize -> stop filter (if stopwords) -> porter (if stem)

# pipeline/porter.py
def stem(word: str) -> str   # tartarus.org Porter (with its documented departures); words of length <= 2 returned unchanged; input assumed lowercase ascii letters
```

- [ ] **Step 1: pyproject and gitignore.** `pyproject.toml` like CardioLens but `name = "broadsheet"`, dependencies `numpy>=1.26,<3` only; dev group `pytest>=8`, `jupyter>=1.0`, `nbconvert>=7`, `ipykernel>=6`, `matplotlib>=3.8`; `[tool.pytest.ini_options] testpaths=["tests"] addopts="-q" pythonpath=["."]`; `[tool.uv] package=false`. `.gitignore`: `data/raw/ft911/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `site/node_modules/`, `site/dist/`, `.superpowers/`, `.ipynb_checkpoints/`, `*.pyc`. Run `uv sync` and commit `uv.lock`.
- [ ] **Step 2: SOURCE.md files.** `data/raw/SOURCE.md`: the corpus is the `ft911` subset of the Financial Times 1991 collection (TREC disk 4), 15 files `ft911_1`..`ft911_15`, 5,368 documents, provided for CSCE 5200 (UNT, Fall 2024); licensed content, not redistributed; place the files here to run the pipeline; hashes in `SHA256SUMS`. `data/trec/SOURCE.md`: topics from `https://trec.nist.gov/data/topics_eng/` (301-350, 351-400, 401-450) and `https://trec.nist.gov/data/robust/04.testset.gz` (the committed file is `04.testset`, 250 topics); qrels from `https://trec.nist.gov/data/robust/qrels.robust2004.txt` filtered with `awk '$3 ~ /^FT911-/'` on 2026-09-12 (3,019 lines); NIST data, public.
- [ ] **Step 3: Mini corpus fixture.** `tests/fixtures/mini/ft_mini_1`: six `<DOC>` records in the real SGML shape (`<DOCNO>FT911-1</DOCNO>` .. `FT911-6`, `<DATE>910514\n</DATE>`, `<HEADLINE>\nFT  14 MAY 91 / Jets over Cranwell\n</HEADLINE>`, `<TEXT>...</TEXT>`, `<PUB>`, `<PAGE>`). Texts, hand-written so that counts are checkable: doc1 "The jet flew. Jets fly fast, and the jet's engine roared 3 times." doc2 "Banks lend money; the bank lent 100 pounds to a banker." doc3 "Antarctica exploration: explorers explore the Antarctic ice." doc4 "The chunnel links Britain and France. British trains, French trains." doc5 "Mutual funds: fund managers predict fund returns." doc6 "A short one." Also `ft_mini_2` with one record `FT911-10` (headline "FT  15 MAY 91 / Tenth", text "Tenth document about jets.") to test multi-file ordering (10 after 6).
- [ ] **Step 4: Failing tests, then implement.** `tests/test_parse.py`: `load_corpus(FIXTURE)` returns 7 docs in order `FT911-1..6, FT911-10`; doc1 headline == "Jets over Cranwell"; date == "1991-05-14"; text contains "engine roared"; `clean_headline("FT  14 MAY 91 / (CORRECTED) Jubilee\nof a jet")` == "(CORRECTED) Jubilee of a jet"; `iso_date("910514")`=="1991-05-14"; missing dir raises FileNotFoundError mentioning `SOURCE.md`. `tests/test_tokenize.py`: `tokenize("The jet's engine roared 3 times, B2 bombers!")` == `["the","jet","s","engine","roared","times","bombers"]`; `analyze(..., stopwords=frozenset({"the","s"}), stem=True)` == `["jet","engin","roar","time","bomber"]`; `load_stopwords()` has 523 entries and contains "the" and "able" but not "". `tests/test_porter.py`: parametrise over `zip(voc, output)` from the golden files (23,531 cases; use one test that iterates and collects mismatches, asserting `mismatches == []`), plus explicit cases `caresses->caress`, `ponies->poni`, `relational->relat`, `sky->sky`, `agreed->agre`, `generalization->gener`, `oscillators->oscil`, `hopping->hop`, `feed->feed`, `at->at`.
- [ ] **Step 5: Implement `porter.py`** as a faithful port of Martin Porter's tartarus.org version (steps 1a, 1b, 1c, 2, 3, 4, 5a, 5b with the `m()` measure, `cons`, `vowelinstem`, `doublec`, `cvc`; the departures: step 1c uses `y -> i` only when a vowel precedes; step 2 includes `logi -> log`, `bli -> ble`; step 4 `ion` requires preceding `s` or `t`). The golden file settles every doubt: the test must pass with zero mismatches.
- [ ] **Step 6: Run `uv run pytest -q`.** Expected: all pass; Porter golden under 5 s.
- [ ] **Step 7: Commit** `feat(pipeline): parser, tokenizer and Porter stemmer with golden tests`.

### Task 2: Index and vocabulary statistics

**Files:**
- Create: `pipeline/index.py`, `pipeline/vocab_stats.py`, `pipeline/io.py`, `tests/test_index.py`, `tests/test_vocab_stats.py`

**Interfaces:**
```python
# pipeline/io.py
def write_json(path: Path, obj: dict, limit: int = SIZE_LIMIT) -> int   # json.dumps(indent=1, allow_nan=False, ensure_ascii=False) + "\n"; mkdir parents; raise ValueError if bytes > limit; returns bytes
def write_gz(path: Path, obj: dict) -> int   # gzip level 9, mtime=0 for determinism, compact separators; returns compressed bytes
def now_iso() -> str
def round6(x: float) -> float
# pipeline/index.py
@dataclass class Index:
    docnos: list[str]; doc_len: list[int]; postings: dict[str, list[tuple[int, int]]]  # term -> [(doc_idx, tf)] ascending doc_idx
    n_docs: int; avgdl: float; total_tokens: int; stem: bool; stop: bool
    def df(self, term) -> int
def build_index(docs: list[Doc], stopwords: frozenset[str] | None, stem: bool) -> Index   # text field only, analyze() per doc; doc_len = number of terms kept
def build_all(docs, stopwords) -> dict[str, Index]   # keyed by TREATMENTS
# pipeline/vocab_stats.py
def funnel(docs, stopwords) -> list[dict]   # four rows: raw_alpha (all [a-z0-9]+ tokens, types), minus_digits, minus_stopwords, stemmed; n_types and n_tokens per step
def heaps_points(docs, stopwords) -> list[dict]; def heaps_fit(points) -> dict
def zipf_points(index) -> list[dict]; def zipf_fit(index) -> dict
def stem_groups(docs, stopwords, index_stem) -> list[dict]   # map each unstemmed stopworded token to its stem; groups sorted by n_words desc then cf desc; 12 groups, 8 words each
def top_terms(index, n=25) -> list[dict]
def posting_hist(index) -> dict; def posting_summary(index) -> dict; def longest(index, n=20)
def doc_len_hist(index) -> dict; def doc_len_summary(index) -> dict
def write_corpus(docs, index_stem_stop, n_files) -> Path; def write_vocab(docs, stopwords, indexes) -> Path; def write_postings(index_stem_stop, size_bytes: dict) -> Path
```

- [ ] **Step 1: Failing tests on the mini fixture.** In `test_index.py` (fixture: `load_corpus(FIXTURE)`, stopwords `frozenset({"the","and","a","to","s"})`): `build_index(docs, sw, stem=True)`: `n_docs == 7`; `df("jet") == 2` (doc1 and doc10; "jets"->"jet"); postings for "jet" == `[(0, 3), (6, 1)]` (doc1: jet, jets, jet's->jet s -> jet ×3; doc10: jets); `df("bank") == 2`? No: doc2 "Banks lend money; the bank lent 100 pounds to a banker" -> banks->bank, bank->bank, banker->banker: `postings["bank"] == [(1, 2)]` and `postings["banker"] == [(1, 1)]`; `doc_len[5] == 2` for doc6 "A short one." -> ["short","one"]; `avgdl == total_tokens / 7`; `build_index(docs, None, stem=False).df("the") == 4` (docs 1,2,3? count precisely when writing the fixture and hard-code); `build_all` returns the four keys. In `test_vocab_stats.py`: `funnel` rows have non-increasing `n_types` and the first row's `n_tokens` equals the count of `[a-z0-9]+` matches; `heaps_fit` on synthetic points `n_types = 5 * n_tokens ** 0.6` recovers `beta` within 1e-6; `zipf_fit` on exact `freq = 1000 / rank` gives slope -1 within 1e-6; `posting_hist` counts sum to `n_terms`; `doc_len_hist` counts sum to `n_docs`.
- [ ] **Step 2: Implement.** Build postings with `dict[str, dict[int,int]]` then freeze to sorted lists. Heaps checkpoints: log-spaced token counts from 1000 to the total (about 40, `numpy.geomspace`, unique ints) plus the final total. Zipf points: ranks 1..10 then `numpy.geomspace(11, V, 50)` unique ints, cf by rank from the sorted term list.
- [ ] **Step 3: Run on the real corpus in a REPL** (`uv run python -c ...`) and note in the report: vocab sizes per treatment, funnel rows, total tokens, top 5 terms, index build time. Nothing hard-coded from this into tests.
- [ ] **Step 4: `uv run pytest -q`** green. **Commit** `feat(pipeline): index builder and vocabulary statistics`.

### Task 3: Topics, qrels and rankers

**Files:**
- Create: `pipeline/topics.py`, `pipeline/rank.py`, `tests/test_topics.py`, `tests/test_rank.py`, `tests/fixtures/mini/topics.txt`, `tests/fixtures/mini/qrels.txt`

**Interfaces:**
```python
# pipeline/topics.py
@dataclass(frozen=True) class Topic: num: int; title: str; desc: str; narr: str
def load_topics(path: Path = TOPICS_PATH) -> list[Topic]      # 250 topics, ascending num; title/desc/narr whitespace-collapsed; desc without the "Description:" label, narr without "Narrative:"
def query_text(topic: Topic, field: str) -> str               # "title" | "title_desc" | "title_desc_narr"
def load_qrels(path: Path = QRELS_PATH, docnos: set[str] | None = None) -> dict[int, dict[str, int]]   # topic -> docno -> 1 if rel > 0 else 0; filtered to docnos when given
FIELDS = {"title": "Title only", "title_desc": "Title + description", "title_desc_narr": "Title + description + narrative"}
# pipeline/rank.py
RANKERS = {"tfidf_raw": "tf-idf cosine (course weighting)", "tfidf_log": "log tf-idf cosine", "bm25": "BM25"}
class Scorer:  # built once per (index, ranker[, k1, b]); caches idf per term and per-document vector norms for the cosine rankers
    def __init__(self, index: Index, ranker: str, k1: float = BM25_K1, b: float = BM25_B)
    def idf(self, term) -> float
    def score(self, query_terms: list[str]) -> dict[int, float]   # doc_idx -> score, only docs with > 0
    def rank(self, query_terms: list[str], top_n: int = TOP_N) -> list[tuple[int, float]]   # sorted by (-score, doc_idx)
    def explain(self, query_terms, doc_idx) -> list[dict]        # [{term, qtf, tf, df, idf, contribution}]
def run_topics(index, scorer, topics, field, top_n=TOP_N) -> dict[int, list[tuple[str, float]]]   # topic num -> [(docno, score)]; query terms = analyze(query_text, stopwords if index.stop else None, index.stem)
```
Formulas (must match the TypeScript port bit-for-bit in operation order):
- `tfidf_raw`: `idf = log10(N / df)`; `w_d = tf * idf`; `w_q = qtf * idf`; `score = sum(w_q * w_d) / (norm_d * norm_q)` where `norm_d = sqrt(sum over all terms in doc of w_d^2)` and `norm_q = sqrt(sum of w_q^2 over query terms present in the vocabulary)`; terms absent from the vocabulary are skipped. Query terms repeated count via qtf.
- `tfidf_log`: identical with `w = (1 + log10(tf)) * idf` for both document and query.
- `bm25`: `idf = ln((N - df + 0.5) / (df + 0.5) + 1)`; `score = sum over query terms of qtf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`.

- [ ] **Step 1: Failing tests.** `test_topics.py`: `load_topics()` has 250 topics, first num 301 title "International Organized Crime", topic 352 title "British Chunnel impact", topic 700 exists; `query_text(t352, "title_desc")` starts with the title and contains "Chunnel had on the British economy"; `load_qrels()` has 3019 judgments in total (sum of len), topic 353 has 10 relevant; with `docnos` given as a 3-element set it filters. Fixture `topics.txt` with two topics (1: title "jet engine", desc "Documents about jets.", narr "Any jet."; 2: title "bank money") and `qrels.txt` (`1 0 FT911-1 1`, `1 0 FT911-10 1`, `1 0 FT911-2 0`, `2 0 FT911-2 1`). `test_rank.py` on a 3-document hand example built with `build_index` from Doc objects (stopwords None, stem False): d0 "apple apple banana", d1 "apple cherry", d2 "banana banana banana cherry cherry". Expected by hand for query `["apple","banana"]`: `tfidf_raw`: N=3, df(apple)=2 -> idf=log10(1.5)=0.176091, df(banana)=2 -> 0.176091, df(cherry)=2 -> 0.176091; d0 weights apple 0.352183, banana 0.176091, norm 0.393756; query weights 0.176091 each, norm 0.249030; dot = 0.352183*0.176091 + 0.176091*0.176091 = 0.093024; score d0 = 0.093024 / (0.393756*0.249030) = 0.948683 (= 3/sqrt(10)); d1: dot 0.176091^2 = 0.031008, norm_d = sqrt(2)*0.176091 = 0.249030, score = 0.5; d2: banana w = 3*0.176091 = 0.528273, cherry 0.352183, norm 0.634919, dot = 0.176091*0.528273 = 0.093024, score = 0.093024/(0.634919*0.249030) = 0.588348 (= sqrt(9/26)). Assert each to 1e-6 and `rank` order `[(0, ..), (2, ..), (1, ..)]`. `bm25` k1=1.2 b=0.75: avgdl = 10/3; idf(apple)=ln((3-2+0.5)/(2+0.5)+1)=ln(1.6)=0.470004; d0 dl=3: tf_apple=2 -> 2*2.2/(2+1.2*(0.25+0.75*0.9)) = 4.4/(2+1.2*0.925) = 4.4/3.11 = 1.414791 -> *0.470004 = 0.664958; banana tf=1 -> 2.2/(1+1.11) = 1.042654 -> 0.490052; total 1.155010. Assert 1e-5. Tie-break test: two identical docs rank by index. `explain` for d0 returns two rows whose contributions sum to the score (bm25) and lists `df`, `idf`.
- [ ] **Step 2: Implement** with precomputed per-doc norms via one pass over postings (`norm_sq[doc] += w*w`). `run_topics` maps doc_idx to docno.
- [ ] **Step 3: Smoke on the real corpus** in a REPL: bm25 title for topic 352 returns 1000 docs; note the top 3 headlines in the report.
- [ ] **Step 4: pytest green; commit** `feat(pipeline): topics, qrels and three rankers`.

### Task 4: Evaluation metrics, bootstrap and run writers

**Files:**
- Create: `pipeline/evaluate.py`, `pipeline/runs.py`, `tests/test_evaluate.py`, `tests/test_runs.py`

**Interfaces:**
```python
# pipeline/evaluate.py
def average_precision(ranked: list[str], rel: set[str]) -> float        # 0.0 if rel empty
def precision_at(ranked, rel, k) -> float; def recall_at(ranked, rel, k) -> float
def r_precision(ranked, rel) -> float; def ndcg_at(ranked, rel, k) -> float   # binary gain, discount 1/log2(i+1), ideal = min(k, |rel|) ones
def judged_at(ranked, judged: set[str], k) -> float
def interpolated_precision(ranked, rel) -> list[float]                     # 11 points at recall 0.0..1.0: max precision at any recall >= level
def bootstrap_mean(values: np.ndarray, n_boot=N_BOOT, seed=SEED) -> dict   # {"mean","lo","hi"} percentile 2.5/97.5, rng = np.random.default_rng(seed), resample indices with rng.integers
def bootstrap_paired(a: np.ndarray, b: np.ndarray, n_boot=N_BOOT, seed=SEED) -> dict   # {"mean_diff","lo","hi","wins","losses","ties"} for a - b
def evaluate_run(run: dict[int, list[tuple[str,float]]], qrels, topics_eval: list[int]) -> dict[str, np.ndarray]  # keys ap, p10, ndcg10, rprec, recall100, judged10, and "interp": 2-D (topics x 11)
def eval_topics(qrels) -> list[int]   # topics with >= 1 relevant docno, ascending
# pipeline/runs.py
def build_runs(indexes, topics, qrels) -> dict   # runs the 9 main runs, 4 treatment runs, grid; returns everything write_* needs
def write_runs(...), write_per_topic(...), write_comparisons(...), write_pr_curves(...), write_topics(...), write_grid(...), write_treatments(...) -> Path each, exactly per the contract
```

- [ ] **Step 1: Failing tests on hand cases.** ranked `[a,b,c,d,e]`, rel `{a,c,e}`: AP = (1 + 2/3 + 3/5)/3 = 0.755556; P@2 = 0.5; recall@2 = 1/3; R-prec (k=3) = 2/3; nDCG@5: DCG = 1 + 1/log2(4) + 1/log2(6) = 1 + 0.5 + 0.386853 = 1.886853, IDCG = 1 + 1/log2(3) + 1/log2(4) = 1 + 0.630930 + 0.5 = 2.130930, nDCG = 0.885458. judged@5 with judged {a,b,z} = 0.4. interpolated: recall levels reached at ranks 1 (1/3, p=1), 3 (2/3, p=2/3), 5 (1, p=3/5): points = [1,1,1,1, 2/3,2/3,2/3, 0.6,0.6,0.6,0.6]. Empty rel -> AP 0, nDCG 0, interp all 0. `bootstrap_mean(np.array([1,2,3,4]))` mean 2.5, lo >= 1, hi <= 4, deterministic across two calls; `bootstrap_paired` with a == b gives mean_diff 0, ties 4. `eval_topics` on the fixture qrels returns `[1, 2]`.
- [ ] **Step 2: Implement.** `build_runs`: stem_stop index for the 9 runs (3 scorers, 3 fields); treatment runs = bm25 title on each of the 4 indexes; grid = bm25 title on stem_stop for each (k1, b) (25 scorers; reuse the index; about 25 x 250 queries, fine). Per-run metrics with `bootstrap_mean` over the 71 topics for each metric. Comparisons per the contract list. PR curves = column mean of `interp`. `topics.json` rows with n_rel/n_judged from qrels and `best_run` = argmax AP (first on ties in run order).
- [ ] **Step 3: `test_runs.py`** builds the mini index and the fixture topics/qrels through `build_runs` and checks: runs length 9, metric dicts have mean/lo/hi, `per_topic` topics == `[1, 2]`, comparisons has 8 pairs with `wins + losses + ties == 2`, grid map shape 5 x 5, each written file parses and has the schema string.
- [ ] **Step 4: pytest green; commit** `feat(pipeline): evaluation metrics, bootstrap intervals and run writers`.

### Task 5: Search export, claims, meta, entry point, output tests

**Files:**
- Create: `pipeline/search_export.py`, `pipeline/claims.py`, `pipeline/run.py`, `pipeline/__main__.py`, `tests/test_outputs.py`, `tests/test_search_export.py`, `README.md` (first version), `LICENSE` (MIT, "Copyright (c) 2026 Sahaj Mekala")
- Generates (commit them): `site/public/data/*.json`, `site/public/data/search/index.json.gz`, `site/src/search/stopwords.json`, `tests/golden/topic_titles.json`, `tests/golden/top10_bm25_title.json`, `tests/golden/top10_tfidf_raw_title.json`

**Interfaces:**
```python
# pipeline/search_export.py
def export_search_index(index: Index, docs: list[Doc], out: Path) -> dict   # returns {"index_gz": bytes, "index_json": bytes}
def export_golden(index, docs, topics, stopwords, golden_dir: Path) -> None   # topic_titles + top10 for bm25 and tfidf_raw, title field
def export_stopwords(stopwords, path: Path) -> None
# pipeline/claims.py
def build_claims(...) -> dict; def build_meta(...) -> dict
# pipeline/run.py
def run() -> None   # deletes site/public/data/*.json and search/*.gz first; writes all files; claims and meta last; prints a table of file sizes
```
`claims.json` `values` keys (all numbers): `n_docs, n_files, date_min, date_max, total_tokens_alpha, n_types_raw, n_types_alpha, n_types_nostop, n_types_stem, stem_reduction_pct` (types removed by stemming as % of nostop types), `stopword_token_share_pct` (tokens removed by stopwording as % of alpha tokens), `df1_share_pct, n_postings, median_doc_len, mean_doc_len, max_posting_term, max_posting_df, n_topics_all, n_topics_eval, n_rel_total, n_judged_docs, n_single_rel_topics, heaps_beta, zipf_slope, map_tfidf_raw_title, map_tfidf_raw_title_lo, map_tfidf_raw_title_hi, map_tfidf_log_title, map_bm25_title, map_bm25_title_lo, map_bm25_title_hi, p10_bm25_title, ndcg10_bm25_title, judged10_bm25_title, recall100_bm25_title, diff_bm25_raw, diff_bm25_raw_lo, diff_bm25_raw_hi, wins_bm25_raw, losses_bm25_raw, ties_bm25_raw, map_bm25_title_desc, map_bm25_title_desc_narr, diff_desc_title, diff_desc_title_lo, diff_desc_title_hi, diff_narr_title, diff_narr_title_lo, diff_narr_title_hi, map_nostem_stop, diff_stem_nostem, diff_stem_nostem_lo, diff_stem_nostem_hi, map_stem_nostop, diff_stop_nostop, diff_stop_nostop_lo, diff_stop_nostop_hi, grid_best_map, grid_best_k1, grid_best_b, grid_default_map, index_gz_mb, index_gz_bytes, n_terms_index, pr_p_at_r0_bm25, pr_p_at_r0_raw` (interpolated precision at recall 0.0).

- [ ] **Step 1: Failing tests.** `test_search_export.py` on the mini index: gz round-trips (`gzip.decompress` -> json) to `schema == "search_index.v1"`, `n_docs == 7`, `docs[0] == {"no":"FT911-1","d":"1991-05-14","h":"Jets over Cranwell"}`, `terms["jet"] == [0, 3, 6, 1]` (gaps), no key `"text"` anywhere, `doc_len` length 7; golden files written with 2 topics. `test_outputs.py` (runs against the committed real outputs, skipped with a clear message if `site/public/data/meta.json` is missing): every contract file exists, parses, is under 200,000 bytes, has its schema string; `runs.json` has 9 runs and every metric has `lo <= mean <= hi` within [0, 1]; `per_topic.json` arrays have `n_topics` entries; `comparisons.json` has 8 pairs and `wins + losses + ties == n_topics`; `pr_curves.json` curves are non-increasing across recall levels; `bm25_grid.json` map is 5 x 5 and `best.map >= default.map`; `topics.json` has `n_topics` rows ascending; `claims.json` has every key listed above and no null; `meta.json` has 15 corpus files with 64-hex sha256; anchors: `n_docs == 5368`, `n_topics_eval == 71`, `n_rel_total == 186`, `n_judged_docs == 1844`, `n_single_rel_topics == 36`, `map_bm25_title > map_tfidf_raw_title` is NOT asserted (unknown until run) but `0 < map_* < 1`; `index.json.gz` under 2,500,000 bytes and no `"text"` key; `stopwords.json` has 523 entries; golden top10 files have an entry for topic 352 with 10 results.
- [ ] **Step 2: Implement and run `uv run python -m pipeline`.** Record timings and sizes in the report. Verify idempotence: run twice, `git diff --stat` shows only `generated_at` changes (use `git diff -I generated_at --stat` -> empty).
- [ ] **Step 3: README first version**: title, one-paragraph pitch, badge placeholder `![ci](https://github.com/sahajm99/broadsheet/actions/workflows/ci.yml/badge.svg)`, "Run it" (uv sync, place corpus per `data/raw/SOURCE.md`, `uv run python -m pipeline`, `uv run pytest`, `cd site && npm ci && npm run dev`), "Data and licensing" (D1, D2, D3), layout table.
- [ ] **Step 4: pytest green (all tests, including outputs); commit** `feat(pipeline): search index export, claims, meta and entry point; generated data`.

### Task 6: Site skeleton, theme, search hero shell, CI, first deploy

**Files:**
- Create: `site/package.json`, `site/tsconfig.json` (strict, `resolveJsonModule`, `types: ["vite/client"]`), `site/vite.config.ts` (base `/broadsheet/`, manualChunks plotly, `assetsInclude` not needed), `site/index.html`, `site/public/.nojekyll`, `site/src/main.ts`, `site/src/theme.ts`, `site/src/plotly.ts` + `plotly.d.ts`, `site/src/data.ts`, `site/src/fmt.ts`, `site/src/lazy.ts`, `site/src/figure.ts`, `site/src/stats-fill.ts`, `site/src/types.ts`, `site/src/charts/theme.ts`, `site/src/styles/{tokens,base,layout,figure,panel,search}.css`, `.github/workflows/ci.yml`
- Reference implementation to adapt (read, do not copy blindly): `C:\Users\sahaj\OneDrive\Desktop\Experiments\projects\active\cardiolens\site\` (same file names; `figure.ts` exports `mountFigure`, `updateFigure`, `showError`, `el`; `charts/theme.ts` exports `layoutTemplate`, `series`, `CONFIG`, `hexToRgba`, `seqColorscale`, `errorBars`, `reversed`; `data.ts` exports `loadJson<T>(name)` with BASE_URL and a cache-bypassing retry; `stats-fill.ts` fills `<span data-stat="key|fmt">` from `claims.json`; `lazy.ts` mounts on IntersectionObserver with `rootMargin: 200px`; `theme.ts` toggles `data-theme` and re-renders mounted charts).

**Design tokens (W4, W5):** light: `--surface: #F7E9E1` (FT salmon paper, muted), `--surface-2: #FFF7F2`, `--ink: #1B1A19`, `--ink-2: #4A4643`, `--muted: #7A736E`, `--accent: #0E6B6B` (deep teal), `--accent-ink: #FFFFFF`, `--rule: #E2CFC4`, series `--s1: #2a78d6 --s2: #eb6834 --s3: #1baf7a --s4: #eda100`, seq ramp teal `--seq-1: #E3F1F1 --seq-2: #A9D6D6 --seq-3: #5FB3B3 --seq-4: #1F8A8A --seq-5: #0B4F4F`. Dark: `--surface: #171514`, `--surface-2: #221F1D`, `--ink: #F2ECE7`, `--ink-2: #CFC6BF`, `--muted: #9A908A`, `--accent: #4FB3B3`, `--rule: #3A3431`, series `#3987e5 #d95926 #199e70 #c98500`, seq `#173A3A #1F5C5C #2E8686 #5FB3B3 #A9D6D6`. Validate the four series colours against both surfaces with `node <dataviz skill dir>/scripts/validate_palette.js "<hex,hex,hex,hex>" --mode light --surface <hex>` (the skill's base directory is `C:\Users\sahaj\AppData\Local\Temp\claude\bundled-skills\2.1.269\5185a28eb3331fc4a6d3dfc9795f0279\dataviz`; if the `--surface` flag is not supported, run the default and record the result); if a check fails, snap the failing series to the nearest passing step and record it in the report. Fonts: `@fontsource-variable/ibm-plex-sans` (body, headings, chart text) and `@fontsource/ibm-plex-mono` (400, 500) for stems, docnos and scores only. Prose width 70ch, figures up to 880px, left aligned; sticky top nav with the six section links, horizontal scroller under 640px; theme toggle button with `aria-pressed`.

**index.html sections (ids):** `#search` (hero: h1 "Broadsheet", one-line deck, the query form: `<form id="q-form"><input id="q" type="search" aria-label="Search 5,368 Financial Times articles from 1991">`, a ranker `<select id="ranker">` with `bm25` / `tfidf_raw`, example chips `<button class="chip" data-q="...">` for "Chunnel", "Antarctica exploration", "journalist risks", "mutual fund", "Gorbachev", "junk bonds"; a `<p class="index-note">` with `data-stat="index_gz_mb|1"` MB; `<ol id="results">` and `<p id="q-status" role="status">`), `#terms` "From text to terms", `#index` "The index", `#eval` "Does it find the right articles?", `#changes` "What changed since the course version", `#limits` "What this cannot tell you". Each section: heading, one placeholder paragraph with at least one `data-stat` span, figure containers `<div class="fig" id="fig-<name>" data-chart="<name>">` for: funnel, heaps, zipf, lengths, topterms, rankers, apstrip, paired, prcurves, fields, grid, treatments. Footer: author, source (TREC disk 4 FT 1991 via CSCE 5200; NIST Robust 2004 judgments), stack, MIT.

- [ ] **Step 1: Scaffold** `npm create vite@latest . -- --template vanilla-ts` inside `site/` (or hand-write the files; either way pin `vite ^8`, `typescript ~6`, `plotly.js-cartesian-dist-min ^4`, `@types/plotly.js`, `vitest ^3`). `npm i`. Add scripts `dev`, `build` (`tsc --noEmit && vite build`), `typecheck`, `test` (`vitest run`), `preview`.
- [ ] **Step 2: Port the chrome** from CardioLens, adapting tokens, fonts and copy. `main.ts`: load `claims.json`, fill stats, register charts lazily from a `Record<string, (c: HTMLElement) => Promise<void>>` map (empty entries render a skeleton with the text "Chart arrives in a later task"), init theme, init nav highlight. The search hero renders its shell (form, chips, disabled state "index loads on first search") but performs no search yet (Task 7).
- [ ] **Step 3: CI** `.github/workflows/ci.yml`: checkout; setup-uv python 3.12; `uv sync --frozen`; `uv run pytest`; setup-node 22 with npm cache on `site/package-lock.json`; `npm ci`; `npm run typecheck`; `npm test`; `npm run build`; upload-pages-artifact `site/dist`; deploy job on main. No pipeline run, no notebook execution (P1).
- [ ] **Step 4: Verify locally**: `npm run typecheck`, `npm run build`, `npm run preview` and a headless screenshot at 1280 and 400 px in light and dark (use the gstack browse tool `$B` if available: `$B goto http://localhost:4173/broadsheet/`, `$B screenshot <path under the repo .superpowers dir>`; otherwise describe what `curl` returns). Gzip bundle sizes in the report.
- [ ] **Step 5: Commit** `feat(site): skeleton, theme, search hero shell and CI`. The controller creates the GitHub repo, pushes and enables Pages.

### Task 7: Browser search engine with explain panel and parity tests

**Files:**
- Create: `site/src/search/porter.ts`, `site/src/search/tokenize.ts`, `site/src/search/index.ts`, `site/src/search/rank.ts`, `site/src/search/ui.ts`, `site/tests/porter.test.ts`, `site/tests/parity.test.ts`, `site/tests/tokenize.test.ts`, `site/vitest.config.ts`
- Modify: `site/src/main.ts` (wire the hero), `site/src/styles/search.css`

**Interfaces:**
```ts
// porter.ts
export function stem(word: string): string            // identical algorithm to pipeline/porter.py
// tokenize.ts
export const STOPWORDS: ReadonlySet<string>            // from ./stopwords.json
export function tokenize(text: string): string[]      // /[a-z0-9]+/g on lowercased text, drop tokens containing a digit
export function analyze(text: string, useStop: boolean, useStem: boolean): string[]
// index.ts
export interface SearchIndex { nDocs: number; avgdl: number; k1: number; b: number; docs: {no: string; d: string; h: string}[]; docLen: Int32Array; terms: Map<string, {docs: Int32Array; tf: Int32Array}> }
export async function loadIndex(fetchImpl?: typeof fetch): Promise<SearchIndex>   // fetch BASE_URL + "data/search/index.json.gz", pipe through DecompressionStream("gzip"), JSON.parse, decode gaps; throws a readable Error if DecompressionStream is undefined
export function decodeIndex(raw: RawIndex): SearchIndex   // pure, testable in node
// rank.ts
export type Ranker = "bm25" | "tfidf_raw"
export interface ExplainRow { term: string; qtf: number; tf: number; df: number; idf: number; contribution: number }
export interface Hit { docIdx: number; no: string; date: string; headline: string; score: number; explain: ExplainRow[] }
export function search(index: SearchIndex, query: string, ranker: Ranker, k = 10): { hits: Hit[]; terms: string[]; unknown: string[] }
```
Formulas and tie-break exactly as Task 3, same operation order (`Math.log10`, `Math.log`, `Math.sqrt`); cosine norms computed once per index per ranker on first use (Float64Array over docs) and cached on the index object. For `tfidf_raw` the explain `contribution` is `w_q * w_d / (norm_d * norm_q)`.

- [ ] **Step 1: Tests first.** `porter.test.ts` reads `../../tests/golden/porter/voc.txt` and `output.txt` with `node:fs` and asserts zero mismatches. `tokenize.test.ts` mirrors the Python cases. `parity.test.ts`: reads `site/public/data/search/index.json.gz` with `node:zlib` `gunzipSync`, `decodeIndex`, then for every topic in `tests/golden/topic_titles.json` runs `search(index, title, "bm25", 10)` and asserts the docno list equals `tests/golden/top10_bm25_title.json[num]` and each score matches within 1e-6; same for `tfidf_raw`. Topics with no results must be absent from the golden file and return zero hits.
- [ ] **Step 2: Implement** `porter.ts` (port the Python line by line), `tokenize.ts`, `index.ts`, `rank.ts`.
- [ ] **Step 3: UI (`ui.ts`).** On first submit: status "Loading the index (<size> MB)…", load once, then search; render up to 10 `<li>` with headline (strong), date and docno in mono, score in mono, and a `<details>` "Why this ranked here" containing a table of explain rows (term, tf, df, idf, contribution; `tf` column header "in doc") plus a footer line "Query terms after stemming: …" and, if any, "Not in the index: …". Empty result: "No article contains any of these stems." Ranker select re-runs the current query. Chips set the input and submit. Errors show a retry button and the message. Keyboard: form submit on Enter, chips are buttons.
- [ ] **Step 4: Verify** `npm test`, `npm run typecheck`, `npm run build`, headless run of a query (Chunnel) with a screenshot of the results and an open explain panel, at 1280 and 400 px. No console errors.
- [ ] **Step 5: Commit** `feat(site): browser search with BM25, tf-idf cosine, explain panel and parity tests`.

### Task 8: Charts 1 to 5 (text and index)

**Files:** create `site/src/charts/funnel.ts`, `heaps.ts`, `zipf.ts`, `lengths.ts`, `topterms.ts`; register in `main.ts`.

Each module exports `render(container: HTMLElement): Promise<void>` and uses `mountFigure` (claim title, subtitle with n, `role="img"`, `<details>` table). Titles are templates from the data (no hand-typed numerals):
1. **funnel** (`vocab.json.funnel`): horizontal bars of `n_types` per step, direct labels with thousands separators, title "Stemming folds {n_types_nostop} word forms into {n_types_stem} stems" (format from data), subtitle with tokens per step.
2. **heaps** (`vocab.json.heaps`, `heaps_fit`): log-log line of types vs tokens with the fitted power law as a dashed line; title "New words keep arriving: vocabulary grows as tokens^{beta}" with beta to 2 dp.
3. **zipf** (`vocab.json.zipf`, `zipf_fit`): log-log scatter+line of frequency vs rank, top 10 terms direct-labelled, fitted line dashed; title "Frequency falls as roughly 1/rank^{-slope}".
4. **lengths** (`postings.json.hist` and `corpus.json.doc_len`): two small multiples side by side (subplots), bar histograms on log-x bins; title "{df1_share_pct}% of stems occur in a single article"; a toggle is not needed.
5. **topterms** (`vocab.json.top_terms`): horizontal bars of cf for the 25 most frequent stems, df as a hollow marker on the same axis? No: single axis rule; show cf bars and put df in the hover and the table. Title "'{term}' is the most frequent stem after stopwording".
- [ ] Verify each renders in light, dark and 400 px; typecheck; build; commit `feat(site): vocabulary and index charts`.

### Task 9: Charts 6 to 12 (evaluation)

**Files:** create `site/src/charts/rankers.ts`, `apstrip.ts`, `paired.ts`, `prcurves.ts`, `fields.ts`, `grid.ts`, `treatments.ts`; register in `main.ts`.

6. **rankers** (`runs.json`, field `title`): dot plot, one row per ranker, mean with 95% error bars, a `<select>` for metric (map, p10, ndcg10, rprec, recall100, judged10); series colours fixed per ranker (tfidf_raw s1, tfidf_log s2, bm25 s3) and identical in every chart. Title template: if the bm25 vs tfidf_raw interval excludes zero: "BM25 beats the course weighting on MAP: {map_bm25} vs {map_raw}" else "BM25 and the course weighting are within the interval on MAP: {..} vs {..}".
7. **apstrip** (`per_topic.json`): strip plot of per-topic AP per ranker (title field), jittered by deterministic hash of topic num, with the mean as a bar marker; title "{n_zero}% of topics score zero under {ranker}" for the ranker with most zeros.
8. **paired** (`comparisons.json`): picker over pairs; sorted per-topic difference bars (positive teal accent, negative orange), a line for the mean and a shaded band for the interval; title "BM25 wins {wins}, loses {losses}, ties {ties} against the course weighting" (template per pair label).
9. **prcurves** (`pr_curves.json`): interpolated PR curves for the three rankers on title; title "At recall 0 BM25 reaches {p0_bm25} precision, the course weighting {p0_raw}".
10. **fields** (`runs.json`): grouped dot plot, rankers x fields, MAP with intervals; title from the desc comparison: "Adding the description changes BM25 MAP by {diff} ({lo} to {hi})".
11. **grid** (`bm25_grid.json`): heat map k1 x b with seq ramp, cell text MAP to 3 dp, default cell outlined, best cell marked; title "Best grid cell {best_map} at k1={k1}, b={b}; default {default_map}"; subtitle carries the caveat verbatim.
12. **treatments** (`treatments.json`): four rows, MAP with intervals and n_types as a right-aligned label; title "Stemming changes MAP by {diff_stem_nostem} ({lo} to {hi})".
- [ ] Verify all render in light, dark, 400 px; typecheck; build; commit `feat(site): evaluation charts`.

### Task 10: Narrative, limits, notebook, README, QA polish

**Files:** modify `site/index.html` (prose for all six sections, every number via `data-stat`), `README.md` (final: insights list, notebook section), create `notebooks/build_notebook.py` and the executed `notebooks/analysis.ipynb` (sections: corpus, tokenisation, stemmer parity, index statistics, rankers, evaluation with intervals, PR curves, grid, limits; uses matplotlib; executed locally with `uv run jupyter nbconvert --execute --to notebook --inplace`), `docs/PROGRESS.md` entries.

Prose rules: one to three paragraphs before each section's first figure and a bridge after; the `#changes` section is a list of four items (document numbers instead of internal ids; log-tf and BM25 beside the course weighting; the evaluation actually run against 71 topics with intervals; one stemmer in two languages checked against 23,531 words and the browser top-10 checked against the pipeline for 250 topic titles); the `#limits` list has at least eight items (2.5% slice of the FT collection; unjudged = not relevant; 36 topics with one relevant document; grid tuned on the test set; no phrase or proximity; stopword list from the course; Porter over- and under-stemming; 1991 English financial news; no significance claims, only intervals; the evaluation compares weightings, not the engine to a modern one). Check with `grep -nE "[0-9]{2,}" site/index.html` that only `data-stat` spans, years in labels, and CSS lengths remain.
- [ ] Verify `npm run build`, `uv run pytest`, `npm test`; commit `feat(site): narrative, limits, notebook and README`.

### Task 11: Portfolio entry

**Files (portfolio repo `C:\Users\sahaj\OneDrive\Desktop\Experiments\projects\active\portfolio\portfolio-next`):** modify `src/data/projects.ts` only: add a `broadsheet` entry after `cardiolens` with `category: "data-engineering"`, `status: "live"`, `liveUrl: "https://sahajm99.github.io/broadsheet/"`, `github: "https://github.com/sahajm99/broadsheet"`, title "Broadsheet — A Search Engine, Measured", description from the DESIGN pitch, tech `["Python", "NumPy", "TypeScript", "Vite", "Plotly.js", "BM25", "TREC evaluation"]`, following the exact shape of the `cardiolens` entry (read it first). Do not stage or commit any other file in that repo (there are unrelated uncommitted changes). Commit only `src/data/projects.ts` as sahajm99, no trailer; the controller pushes.
