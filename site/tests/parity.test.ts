// tsconfig limits `types` to vite/client for the bundle; the tests run in
// node and read fixtures off disk, so they ask for the node types here.
/// <reference types="node" />

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

import { beforeAll, describe, expect, it } from "vitest";

import { decodeIndex, type RawIndex, type SearchIndex } from "../src/search/index.ts";
import { search, type Ranker } from "../src/search/rank.ts";

interface GoldenHit {
  no: string;
  score: number;
}

function repoFile(relative: string): string {
  return fileURLToPath(new URL(`../../${relative}`, import.meta.url));
}

function readGolden<T>(name: string): T {
  return JSON.parse(readFileSync(repoFile(`tests/golden/${name}`), "utf8")) as T;
}

const titles = readGolden<Record<string, string>>("topic_titles.json");
const GOLDEN: Record<Ranker, Record<string, GoldenHit[]>> = {
  bm25: readGolden("top10_bm25_title.json"),
  tfidf_raw: readGolden("top10_tfidf_raw_title.json"),
};

let index: SearchIndex;

beforeAll(() => {
  // The browser inflates with DecompressionStream; node has zlib, and both
  // hand the same bytes to the same decoder.
  const gz = readFileSync(
    fileURLToPath(new URL("../public/data/search/index.json.gz", import.meta.url)),
  );
  index = decodeIndex(JSON.parse(gunzipSync(gz).toString("utf8")) as RawIndex);
});

describe("the decoded index", () => {
  it("matches the corpus the pipeline indexed", () => {
    expect(index.nDocs).toBe(5368);
    expect(index.docs.length).toBe(5368);
    expect(index.docLen.length).toBe(5368);
    expect(index.k1).toBe(1.2);
    expect(index.b).toBe(0.75);
    // Recomputed from doc_len rather than read from the rounded field.
    expect(index.avgdl).toBeCloseTo(193.128539, 6);
  });

  it("decodes gaps into ascending document indices", () => {
    for (const [stem, postings] of index.terms) {
      expect(postings.docs.length, stem).toBe(postings.tf.length);
      let previous = -1;
      for (let i = 0; i < postings.docs.length; i += 1) {
        expect(postings.docs[i]).toBeGreaterThan(previous);
        expect(postings.tf[i]).toBeGreaterThan(0);
        previous = postings.docs[i];
      }
      break;
    }
  });
});

for (const ranker of ["bm25", "tfidf_raw"] as const) {
  describe(`${ranker} parity with the pipeline`, () => {
    it("ranks all 250 topic titles exactly as the Python run did", () => {
      const golden = GOLDEN[ranker];
      const problems: string[] = [];
      let checked = 0;

      for (const [num, expected] of Object.entries(golden)) {
        const title = titles[num];
        expect(title, `topic ${num} missing from topic_titles.json`).toBeTypeOf(
          "string",
        );
        const { hits } = search(index, title, ranker, 10);
        const got = hits.map((h) => h.no);
        const want = expected.map((h) => h.no);
        if (got.join(" ") !== want.join(" ")) {
          problems.push(`topic ${num}: ${got.join(",")} != ${want.join(",")}`);
          continue;
        }
        for (let i = 0; i < expected.length; i += 1) {
          const delta = Math.abs(hits[i].score - expected[i].score);
          // The fixture is rounded to 6 dp, so 1e-6 is the whole tolerance.
          if (delta > 1e-6) {
            problems.push(
              `topic ${num} rank ${i + 1}: ${hits[i].score} != ${expected[i].score}`,
            );
          }
        }
        checked += 1;
      }

      expect(problems.slice(0, 10)).toEqual([]);
      expect(problems.length).toBe(0);
      expect(checked).toBe(Object.keys(golden).length);
      expect(checked).toBe(243);
    });

    it("returns nothing for the topics the fixture leaves out", () => {
      const golden = GOLDEN[ranker];
      const missing = Object.keys(titles).filter((num) => !(num in golden));
      expect(missing.length).toBe(250 - 243);
      for (const num of missing) {
        expect(search(index, titles[num], ranker, 10).hits, `topic ${num}`).toEqual(
          [],
        );
      }
    });
  });
}

describe("explain", () => {
  it("contributions sum to the score of the hit", () => {
    for (const ranker of ["bm25", "tfidf_raw"] as const) {
      const { hits } = search(index, "Channel tunnel", ranker, 10);
      expect(hits.length).toBe(10);
      for (const hit of hits) {
        expect(hit.explain.length).toBeGreaterThan(0);
        const total = hit.explain.reduce((sum, row) => sum + row.contribution, 0);
        expect(total).toBeCloseTo(hit.score, 9);
        for (const row of hit.explain) {
          expect(row.df).toBeGreaterThan(0);
          expect(row.tf).toBeGreaterThan(0);
        }
      }
    }
  });

  it("reports stems the index has never seen", () => {
    const { hits, terms, unknown } = search(
      index,
      "zzzqqq the Chunnel",
      "bm25",
      10,
    );
    // "the" is a stopword and never reaches the index; the other two are
    // simply absent from this corpus.
    expect(terms).toEqual(["zzzqqq", "chunnel"]);
    expect(unknown).toEqual(["zzzqqq", "chunnel"]);
    expect(hits).toEqual([]);
  });
});

describe("latency", () => {
  it("answers a two-word query well inside the 50 ms budget", () => {
    search(index, "Channel tunnel", "bm25", 10); // warm the norm cache
    search(index, "Channel tunnel", "tfidf_raw", 10);
    for (const ranker of ["bm25", "tfidf_raw"] as const) {
      const start = performance.now();
      for (let i = 0; i < 20; i += 1) search(index, "Channel tunnel", ranker, 10);
      const each = (performance.now() - start) / 20;
      expect(each, `${ranker} took ${each.toFixed(2)} ms`).toBeLessThan(50);
    }
  });
});
