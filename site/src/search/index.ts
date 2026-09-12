/**
 * The index the browser searches: fetched once, inflated with
 * `DecompressionStream`, and decoded into typed arrays (W2).
 *
 * The file is `search_index.v1`, written by `pipeline/search_export.py`. It
 * carries stems, posting lists, document lengths and one citation line per
 * article -- DOCNO, date, headline -- and never any article text (D1/D2).
 */

/** The JSON as it comes out of the gzip, before any decoding. */
export interface RawIndex {
  schema: string;
  n_docs: number;
  /** Rounded to 6 dp in the file; `decodeIndex` recomputes it instead. */
  avgdl: number;
  k1: number;
  b: number;
  docs: { no: string; d: string; h: string }[];
  doc_len: number[];
  /** `stem -> [gap, tf, gap, tf, ...]`; the first gap is absolute. */
  terms: Record<string, number[]>;
}

/** One term's postings, ascending by document index. */
export interface Postings {
  docs: Int32Array;
  tf: Int32Array;
}

export interface SearchIndex {
  nDocs: number;
  avgdl: number;
  k1: number;
  b: number;
  docs: { no: string; d: string; h: string }[];
  docLen: Int32Array;
  terms: Map<string, Postings>;
}

const SCHEMA = "search_index.v1";
const PATH = "data/search/index.json.gz";

/**
 * Decode the raw payload. Pure, so node can test it with `zlib` while the
 * browser goes through `DecompressionStream`.
 *
 * Postings arrive gap-encoded: a term in documents 0, 3 and 4 is written
 * `[0, tf, 3, tf, 1, tf]`, so decoding is a running sum. `docLen[i]` and
 * `docs[i]` address the same document as a posting's document index.
 *
 * `avgdl` is recomputed from `doc_len` rather than read from the file: the
 * field is rounded to 6 dp, while the pipeline scored with the full double.
 * Recomputing makes BM25 parity exact by construction instead of exact to
 * about eleven significant figures.
 */
export function decodeIndex(raw: RawIndex): SearchIndex {
  if (raw.schema !== SCHEMA) {
    throw new Error(
      `Unexpected search index schema "${raw.schema}" (expected "${SCHEMA}")`,
    );
  }
  const docLen = Int32Array.from(raw.doc_len);
  let total = 0;
  for (let i = 0; i < docLen.length; i += 1) total += docLen[i];

  const terms = new Map<string, Postings>();
  for (const [term, flat] of Object.entries(raw.terms)) {
    const n = flat.length >> 1;
    const docs = new Int32Array(n);
    const tf = new Int32Array(n);
    let previous = 0;
    for (let i = 0; i < n; i += 1) {
      previous += flat[2 * i];
      docs[i] = previous;
      tf[i] = flat[2 * i + 1];
    }
    terms.set(term, { docs, tf });
  }

  return {
    nDocs: raw.n_docs,
    avgdl: raw.n_docs > 0 ? total / raw.n_docs : 0,
    k1: raw.k1,
    b: raw.b,
    docs: raw.docs,
    docLen,
    terms,
  };
}

/** The element type of a `fetch` body stream, spelled once. */
type Bytes = Uint8Array<ArrayBuffer>;

/** gzip's magic number, which is how we know whether to inflate at all. */
const MAGIC = [0x1f, 0x8b];

/**
 * Read enough of `body` to see the first two bytes, then hand back a stream
 * that still starts at byte zero.
 *
 * Sniffing beats trusting the server. GitHub Pages serves `.gz` as an opaque
 * download with no `Content-Encoding`, so the page must inflate it itself
 * (W2) -- but `vite preview` and some CDNs label the same file
 * `Content-Encoding: gzip`, in which case the browser has already inflated it
 * and inflating again would fail. Two bytes settle it either way.
 */
async function peek(
  body: ReadableStream<Bytes>,
): Promise<{ stream: ReadableStream<Bytes>; gzipped: boolean }> {
  const reader = body.getReader();
  const head: Bytes[] = [];
  let seen = 0;
  while (seen < MAGIC.length) {
    const { done, value } = await reader.read();
    if (done) break;
    head.push(value);
    seen += value.length;
  }
  const prefix: number[] = [];
  for (const chunk of head) {
    for (let i = 0; i < chunk.length && prefix.length < MAGIC.length; i += 1) {
      prefix.push(chunk[i]);
    }
  }
  const gzipped = MAGIC.every((byte, i) => prefix[i] === byte);

  const stream = new ReadableStream<Bytes>({
    start(controller) {
      for (const chunk of head) controller.enqueue(chunk);
    },
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) controller.close();
      else controller.enqueue(value);
    },
    cancel(reason: unknown) {
      void reader.cancel(reason);
    },
  });
  return { stream, gzipped };
}

/**
 * Fetch and decode the index: one 1.25 MB download on the first query, and
 * nothing after that for the rest of the visit.
 */
export async function loadIndex(
  fetchImpl: typeof fetch = fetch,
): Promise<SearchIndex> {
  const url = `${import.meta.env.BASE_URL}${PATH}`;
  const res = await fetchImpl(url, { cache: "force-cache" });
  if (!res.ok) throw new Error(`Could not load the index (${res.status})`);
  if (!res.body) throw new Error("Could not load the index (no response body)");

  const { stream, gzipped } = await peek(res.body);
  let source: ReadableStream<Bytes> = stream;
  if (gzipped) {
    if (typeof DecompressionStream === "undefined") {
      throw new Error(
        "This browser cannot inflate the index: it has no DecompressionStream. " +
          "Any current Chrome, Firefox, Safari or Edge does.",
      );
    }
    source = stream.pipeThrough(new DecompressionStream("gzip"));
  }
  const raw = (await new Response(source).json()) as RawIndex;
  return decodeIndex(raw);
}
