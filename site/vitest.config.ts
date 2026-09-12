import { defineConfig } from "vitest/config";

/**
 * The search tests run in node: they read the golden fixtures and the gzipped
 * index straight off disk with `node:fs`, so nothing here needs a DOM or a
 * dev server. A separate config (rather than `test` inside `vite.config.ts`)
 * keeps the site's `base` and Plotly chunking out of the test run.
 */
export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // Inflating and decoding a 4 MB index once, then ranking 486 queries,
    // is comfortably inside this but not inside vitest's 5 s default.
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});
