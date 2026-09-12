/**
 * Shapes of the JSON aggregates written by `python -m pipeline` into
 * `site/public/data/`. One interface per contract file, so a chart task can
 * name its file and get the fields checked.
 */

/** A bootstrap mean with its percentile interval (S1). */
export interface Interval {
  mean: number;
  lo: number;
  hi: number;
}

/** `claims.v1`: a flat bag of named numbers quoted in the prose. */
export interface Claims {
  schema: "claims.v1";
  generated_at: string;
  values: Record<string, number | string>;
}

/** `meta.v1`: provenance of the corpus and the size of the search index. */
export interface Meta {
  schema: "meta.v1";
  generated_at: string;
  corpus_files: { name: string; sha256: string }[];
  n_docs: number;
  index_gz_bytes: number;
  index_gz_mb: number;
  seed: number;
  n_boot: number;
  python: string;
  numpy: string;
}

/** `corpus.v1`: what the collection is, and how long its documents are. */
export interface Corpus {
  schema: "corpus.v1";
  generated_at: string;
  n_docs: number;
  n_files: number;
  date_min: string;
  date_max: string;
  total_tokens_alpha: number;
  /** Document length in stemmed, stopworded terms; bins are closed-open. */
  doc_len: { bins: number[]; counts: number[] };
  doc_len_summary: {
    mean: number;
    median: number;
    p10: number;
    p90: number;
    max: number;
  };
}

export interface FunnelStep {
  step: string;
  label: string;
  n_types: number;
  n_tokens: number;
}

export interface TreatmentSize {
  n_types: number;
  n_tokens: number;
}

/** One stem and the surface forms that fold into it. */
export interface StemGroup {
  stem: string;
  n_words: number;
  words: { word: string; cf: number }[];
}

/** `vocab.v1`: the funnel from raw text to terms, plus Zipf and Heaps. */
export interface Vocab {
  schema: "vocab.v1";
  generated_at: string;
  funnel: FunnelStep[];
  treatments: {
    stem_stop: TreatmentSize;
    stem_nostop: TreatmentSize;
    nostem_stop: TreatmentSize;
    nostem_nostop: TreatmentSize;
  };
  top_terms: { term: string; df: number; cf: number }[];
  stem_groups: StemGroup[];
  heaps: { n_tokens: number; n_types: number }[];
  heaps_fit: { k: number; beta: number };
  zipf: { rank: number; freq: number; term: string }[];
  zipf_fit: { slope: number; intercept: number };
}

/** `postings.v1`: the shape of the inverted index. */
export interface Postings {
  schema: "postings.v1";
  generated_at: string;
  hist: { bins: number[]; counts: number[] };
  summary: {
    n_terms: number;
    n_postings: number;
    mean_len: number;
    median_len: number;
    max_len: number;
    df1_share: number;
  };
  longest: { term: string; df: number }[];
  size_bytes: { index_gz: number; index_json: number };
}

/** Every metric a run reports, each as a bootstrap mean with an interval. */
export interface RunMetrics {
  map: Interval;
  p10: Interval;
  ndcg10: Interval;
  rprec: Interval;
  recall100: Interval;
  judged10: Interval;
}

/** One (ranker, query field) run, keyed `"bm25|title"`. */
export interface Run {
  key: string;
  ranker: string;
  field: string;
  metrics: RunMetrics;
}

export type MetricKey = keyof RunMetrics;

/** `runs.v1`: the nine runs and the counts that frame every mean. */
export interface Runs {
  schema: "runs.v1";
  generated_at: string;
  n_topics: number;
  n_topics_all: number;
  n_rel_total: number;
  n_judged_docs: number;
  n_single_rel_topics: number;
  rankers: { key: string; label: string }[];
  fields: { key: string; label: string }[];
  runs: Run[];
}

/** `per_topic.v1`: average precision per topic, per run. */
export interface PerTopic {
  schema: "per_topic.v1";
  generated_at: string;
  topics: number[];
  ap: Record<string, number[] | undefined>;
}

/** One paired comparison, `a` minus `b`, over the evaluable topics. */
export interface Comparison {
  key: string;
  a: string;
  b: string;
  label: string;
  mean_diff: number;
  lo: number;
  hi: number;
  wins: number;
  losses: number;
  ties: number;
  per_topic: { num: number; diff: number }[];
}

/** `comparisons.v1`: the eight paired differences the page argues from. */
export interface Comparisons {
  schema: "comparisons.v1";
  generated_at: string;
  pairs: Comparison[];
}

/** `pr_curves.v1`: mean interpolated precision at eleven recall levels. */
export interface PrCurves {
  schema: "pr_curves.v1";
  generated_at: string;
  recall_levels: number[];
  curves: Record<string, number[] | undefined>;
}

/** One evaluable topic and how each run scored on it. */
export interface TopicRow {
  num: number;
  title: string;
  n_rel: number;
  n_judged: number;
  ap: Record<string, number | undefined>;
  best_run: string;
}

/** `topics.v1`: the topics with at least one relevant document in the slice. */
export interface Topics {
  schema: "topics.v1";
  generated_at: string;
  rows: TopicRow[];
}

/** `bm25_grid.v1`: MAP over the k1 x b grid, with the tuning caveat (R3). */
export interface Bm25Grid {
  schema: "bm25_grid.v1";
  generated_at: string;
  k1: number[];
  b: number[];
  /** Rows are k1, columns are b. */
  map: number[][];
  best: { k1: number; b: number; map: number };
  default: { k1: number; b: number; map: number };
  caveat: string;
}

/** `treatments.v1`: stemming and stopwording crossed, scored with BM25. */
export interface Treatments {
  schema: "treatments.v1";
  generated_at: string;
  ranker: string;
  field: string;
  rows: {
    key: string;
    label: string;
    n_types: number;
    map: Interval;
    p10: Interval;
  }[];
}
