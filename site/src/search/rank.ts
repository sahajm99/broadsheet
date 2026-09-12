/**
 * The two rankers the page offers, ported from `pipeline/rank.py` (W3).
 *
 * Scoring is postings-driven: a query touches only the posting lists of its
 * own terms and accumulates into a `Float64Array` keyed by document index, so
 * cost is the length of those lists and not the size of the collection.
 *
 * The operation order below is load-bearing. A score is accumulated per
 * document as the sum, over the *distinct* query terms in order of first
 * appearance, of that term's contribution; for the cosine ranker the sum is
 * divided once, at the end, by `norm_d * norm_q`. Cosine weights use
 * `Math.log10`, BM25's idf uses `Math.log`. Floating-point addition is not
 * associative, and `tests/parity.test.ts` compares top tens document by
 * document against the Python run, so none of this is free to be tidied.
 */

import type { SearchIndex } from "./index.ts";
import { analyze } from "./tokenize.ts";

export type Ranker = "bm25" | "tfidf_raw";

export const RANKER_LABELS: Record<Ranker, string> = {
  bm25: "BM25",
  tfidf_raw: "tf-idf cosine (course weighting)",
};

export function isRanker(value: string): value is Ranker {
  return value === "bm25" || value === "tfidf_raw";
}

/** One row of the arithmetic behind a single result. */
export interface ExplainRow {
  term: string;
  qtf: number;
  tf: number;
  df: number;
  idf: number;
  contribution: number;
}

export interface Hit {
  docIdx: number;
  no: string;
  date: string;
  headline: string;
  score: number;
  explain: ExplainRow[];
}

export interface SearchResult {
  hits: Hit[];
  /** Distinct query stems, in order of first appearance. */
  terms: string[];
  /** The subset of `terms` with df 0: stems this corpus has never seen. */
  unknown: string[];
  /** How many articles scored above zero, of which `hits` is the top slice. */
  total: number;
}

/**
 * Document norms are an index-wide property of one ranker, so they are built
 * once and reused. A `WeakMap` keeps them out of the `SearchIndex` shape and
 * lets the index be collected with them.
 */
const NORM_CACHE = new WeakMap<SearchIndex, Map<Ranker, Float64Array>>();

/**
 * Inverse document frequency; 0 for a term outside the vocabulary.
 *
 * `log10(N / df)` for the cosine ranker, the BM25-plus-one variant
 * `ln((N - df + 0.5) / (df + 0.5) + 1)` for BM25, which is never negative and
 * so cannot subtract score for a very common term. Note that the cosine idf
 * is exactly 0 when a term is in every document, which drops that term from
 * the query and from every document vector.
 */
function idfOf(index: SearchIndex, df: number, ranker: Ranker): number {
  if (df === 0) return 0;
  if (ranker === "bm25") {
    return Math.log((index.nDocs - df + 0.5) / (df + 0.5) + 1);
  }
  return Math.log10(index.nDocs / df);
}

function idfOfTerm(index: SearchIndex, term: string, ranker: Ranker): number {
  const postings = index.terms.get(term);
  return idfOf(index, postings ? postings.docs.length : 0, ranker);
}

/** Euclidean norm of every document vector, in one pass over the postings. */
function docNorms(index: SearchIndex, ranker: Ranker): Float64Array {
  let perRanker = NORM_CACHE.get(index);
  if (!perRanker) {
    perRanker = new Map();
    NORM_CACHE.set(index, perRanker);
  }
  const cached = perRanker.get(ranker);
  if (cached) return cached;

  const normSq = new Float64Array(index.nDocs);
  for (const postings of index.terms.values()) {
    const idf = idfOf(index, postings.docs.length, ranker);
    if (idf === 0) continue;
    for (let i = 0; i < postings.docs.length; i += 1) {
      const w = postings.tf[i] * idf;
      normSq[postings.docs[i]] += w * w;
    }
  }
  for (let d = 0; d < normSq.length; d += 1) normSq[d] = Math.sqrt(normSq[d]);
  perRanker.set(ranker, normSq);
  return normSq;
}

/**
 * Distinct query terms that are in the vocabulary, in query order, mapped to
 * their query term frequency. `Map` keeps insertion order, which fixes the
 * accumulation order the Python dict fixes on the other side.
 */
function queryTf(index: SearchIndex, stems: string[]): Map<string, number> {
  const qtf = new Map<string, number>();
  for (const term of stems) {
    const seen = qtf.get(term);
    if (seen !== undefined) qtf.set(term, seen + 1);
    else if (index.terms.has(term)) qtf.set(term, 1);
  }
  return qtf;
}

/** Term frequency of `term` in one document; 0 when it does not occur. */
function tfIn(index: SearchIndex, term: string, docIdx: number): number {
  const postings = index.terms.get(term);
  if (!postings) return 0;
  let lo = 0;
  let hi = postings.docs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const d = postings.docs[mid];
    if (d === docIdx) return postings.tf[mid];
    if (d < docIdx) lo = mid + 1;
    else hi = mid - 1;
  }
  return 0;
}

interface Accumulator {
  score: Float64Array;
  touched: number[];
}

function scoreBm25(
  index: SearchIndex,
  qtf: Map<string, number>,
): Accumulator | null {
  const { k1, b, avgdl, docLen } = index;
  const score = new Float64Array(index.nDocs);
  const seen = new Uint8Array(index.nDocs);
  const touched: number[] = [];
  for (const [term, q] of qtf) {
    const postings = index.terms.get(term);
    if (!postings) continue;
    const idf = idfOf(index, postings.docs.length, "bm25");
    if (idf === 0) continue;
    for (let i = 0; i < postings.docs.length; i += 1) {
      const d = postings.docs[i];
      const tf = postings.tf[i];
      const dl = docLen[d];
      score[d] += (q * idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / avgdl));
      if (seen[d] === 0) {
        seen[d] = 1;
        touched.push(d);
      }
    }
  }
  return { score, touched };
}

function scoreCosine(
  index: SearchIndex,
  qtf: Map<string, number>,
): Accumulator | null {
  const dot = new Float64Array(index.nDocs);
  const seen = new Uint8Array(index.nDocs);
  const touched: number[] = [];
  let normQSq = 0;
  for (const [term, q] of qtf) {
    const postings = index.terms.get(term);
    if (!postings) continue;
    const idf = idfOf(index, postings.docs.length, "tfidf_raw");
    const wq = q * idf;
    normQSq += wq * wq;
    if (wq === 0) continue;
    for (let i = 0; i < postings.docs.length; i += 1) {
      const d = postings.docs[i];
      dot[d] += wq * (postings.tf[i] * idf);
      if (seen[d] === 0) {
        seen[d] = 1;
        touched.push(d);
      }
    }
  }
  const normQ = Math.sqrt(normQSq);
  if (normQ === 0) return null;

  const norm = docNorms(index, "tfidf_raw");
  const score = dot;
  const kept: number[] = [];
  for (const d of touched) {
    const normD = norm[d];
    if (normD === 0 || score[d] <= 0) {
      score[d] = 0;
      continue;
    }
    score[d] = score[d] / (normD * normQ);
    kept.push(d);
  }
  return { score, touched: kept };
}

/** The query's own norm, in the same term order the score used. */
function queryNorm(
  index: SearchIndex,
  qtf: Map<string, number>,
  ranker: Ranker,
): number {
  let sum = 0;
  for (const [term, q] of qtf) {
    const w = q * idfOfTerm(index, term, ranker);
    sum += w * w;
  }
  return Math.sqrt(sum);
}

/**
 * One row per distinct query term the document actually has, in query order,
 * whose contributions sum to the document's score. Terms outside the
 * vocabulary, and terms absent from this document, are left out.
 */
function explainHit(
  index: SearchIndex,
  qtf: Map<string, number>,
  ranker: Ranker,
  docIdx: number,
): ExplainRow[] {
  const rows: ExplainRow[] = [];
  if (ranker === "bm25") {
    const { k1, b, avgdl } = index;
    const dl = index.docLen[docIdx];
    for (const [term, q] of qtf) {
      const tf = tfIn(index, term, docIdx);
      if (tf === 0) continue;
      const df = index.terms.get(term)?.docs.length ?? 0;
      const idf = idfOf(index, df, "bm25");
      rows.push({
        term,
        qtf: q,
        tf,
        df,
        idf,
        contribution:
          (q * idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / avgdl)),
      });
    }
    return rows;
  }

  const denom = docNorms(index, ranker)[docIdx] * queryNorm(index, qtf, ranker);
  // Every query term is in every document, so nothing scores at all and there
  // is no arithmetic to show.
  if (denom === 0) return rows;
  for (const [term, q] of qtf) {
    const tf = tfIn(index, term, docIdx);
    if (tf === 0) continue;
    const df = index.terms.get(term)?.docs.length ?? 0;
    const idf = idfOf(index, df, ranker);
    const wq = q * idf;
    const wd = tf * idf;
    rows.push({ term, qtf: q, tf, df, idf, contribution: (wq * wd) / denom });
  }
  return rows;
}

/**
 * Rank the collection for one query string.
 *
 * The query is analysed exactly as the index was built -- stopwords removed,
 * then stemmed -- because the index it is searching was built that way.
 * Ties are broken by ascending document index, after descending score, which
 * is what `sorted(..., key=lambda item: (-item[1], item[0]))` does in Python.
 */
export function search(
  index: SearchIndex,
  query: string,
  ranker: Ranker,
  k = 10,
): SearchResult {
  const stems = analyze(query, true, true);
  const terms: string[] = [];
  const seen = new Set<string>();
  for (const term of stems) {
    if (seen.has(term)) continue;
    seen.add(term);
    terms.push(term);
  }
  const unknown = terms.filter((term) => !index.terms.has(term));

  const qtf = queryTf(index, stems);
  if (qtf.size === 0) return { hits: [], terms, unknown, total: 0 };

  const acc =
    ranker === "bm25" ? scoreBm25(index, qtf) : scoreCosine(index, qtf);
  if (!acc) return { hits: [], terms, unknown, total: 0 };

  const scored: number[] = [];
  for (const d of acc.touched) if (acc.score[d] > 0) scored.push(d);
  scored.sort((a, bIdx) => acc.score[bIdx] - acc.score[a] || a - bIdx);

  const hits = scored.slice(0, k).map((docIdx) => {
    const doc = index.docs[docIdx];
    return {
      docIdx,
      no: doc.no,
      date: doc.d,
      headline: doc.h,
      score: acc.score[docIdx],
      explain: explainHit(index, qtf, ranker, docIdx),
    };
  });

  return { hits, terms, unknown, total: scored.length };
}
