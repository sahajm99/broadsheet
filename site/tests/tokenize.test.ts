import { describe, expect, it } from "vitest";

import { analyze, STOPWORDS, tokenize } from "../src/search/tokenize.ts";

const SENTENCE = "The jet's engine roared 3 times, B2 bombers!";

describe("tokenize", () => {
  it("lowercases, splits on non-alphanumerics and drops digit-bearing tokens", () => {
    expect(tokenize(SENTENCE)).toEqual([
      "the",
      "jet",
      "s",
      "engine",
      "roared",
      "times",
      "bombers",
    ]);
  });

  it("has nothing to say about empty or all-numeric text", () => {
    expect(tokenize("")).toEqual([]);
    expect(tokenize("1991 747 -- 3.5%")).toEqual([]);
  });
});

describe("analyze", () => {
  it("removes stopwords before stemming", () => {
    // "the" and "s" are both on the course list; if stemming ran first the
    // surface forms would no longer match the list.
    expect(analyze(SENTENCE, true, true)).toEqual([
      "jet",
      "engin",
      "roar",
      "time",
      "bomber",
    ]);
  });

  it("is the tokenizer when neither step is asked for", () => {
    expect(analyze(SENTENCE, false, false)).toEqual(tokenize(SENTENCE));
  });

  it("does either step on its own", () => {
    expect(analyze(SENTENCE, true, false)).toEqual([
      "jet",
      "engine",
      "roared",
      "times",
      "bombers",
    ]);
    expect(analyze(SENTENCE, false, true)).toEqual([
      "the",
      "jet",
      "s",
      "engin",
      "roar",
      "time",
      "bomber",
    ]);
  });
});

describe("STOPWORDS", () => {
  it("is the course list the index was built with", () => {
    expect(STOPWORDS.size).toBe(523);
    expect(STOPWORDS.has("the")).toBe(true);
    expect(STOPWORDS.has("able")).toBe(true);
    expect(STOPWORDS.has("")).toBe(false);
    for (const word of STOPWORDS) expect(word).toBe(word.trim().toLowerCase());
  });
});
