// tsconfig limits `types` to vite/client for the bundle; the tests run in
// node and read fixtures off disk, so they ask for the node types here.
/// <reference types="node" />

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { stem } from "../src/search/porter.ts";

/** `tests/golden/porter/` at the repository root, two levels above `site/`. */
function golden(name: string): string[] {
  const path = fileURLToPath(
    new URL(`../../tests/golden/porter/${name}`, import.meta.url),
  );
  return readFileSync(path, "utf8").split("\n").map((line) => line.trim());
}

describe("porter", () => {
  it("reproduces Martin Porter's published vectors word for word", () => {
    const words = golden("voc.txt");
    const expected = golden("output.txt");
    expect(words.length).toBe(expected.length);

    const mismatches: string[] = [];
    for (let i = 0; i < words.length; i += 1) {
      const word = words[i];
      if (!word) continue;
      const got = stem(word);
      if (got !== expected[i]) {
        mismatches.push(`${word} -> ${got} (expected ${expected[i]})`);
      }
    }
    // 23,531 words; anything above zero means the port and the pipeline
    // would index different stems.
    expect(mismatches.slice(0, 20)).toEqual([]);
    expect(mismatches.length).toBe(0);
  });

  it("leaves words of two letters or fewer alone", () => {
    expect(stem("")).toBe("");
    expect(stem("a")).toBe("a");
    expect(stem("as")).toBe("as");
    expect(stem("is")).toBe("is");
  });

  it("stems the examples from the paper", () => {
    expect(stem("caresses")).toBe("caress");
    expect(stem("ponies")).toBe("poni");
    // The paper writes agreed -> agree; the reference implementation then
    // runs step 5, whose m > 1 test strips the final e. `pipeline/porter.py`
    // gives "agre" too, which is the point of checking.
    expect(stem("agreed")).toBe("agre");
    expect(stem("plastered")).toBe("plaster");
    expect(stem("hopping")).toBe("hop");
    expect(stem("happy")).toBe("happi");
    expect(stem("sky")).toBe("sky");
    expect(stem("relational")).toBe("relat");
    expect(stem("digitizer")).toBe("digit");
    expect(stem("controll")).toBe("control");
    expect(stem("roll")).toBe("roll");
  });
});
