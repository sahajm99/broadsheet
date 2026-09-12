/**
 * The search hero: the form, the ranker switch, the example chips, and the
 * engine behind them.
 *
 * Everything here happens in the visitor's browser. The first query fetches
 * one gzipped index (W2), inflates it, and from then on every ranking is
 * local: no server, no API, no article text ever leaving the pipeline's
 * citation lines. `site/tests/parity.test.ts` checks that this code ranks all
 * 243 answerable TREC topics exactly as `pipeline/rank.py` does (W3).
 */

import { loadJson } from "./data.ts";
import { loadIndex, type SearchIndex } from "./search/index.ts";
import {
  isRanker,
  search,
  type Ranker,
  type SearchResult,
} from "./search/rank.ts";
import { clearExtra, renderError, renderResults } from "./search/ui.ts";
import type { Meta } from "./types.ts";

const TOP_K = 10;

export interface SearchShell {
  form: HTMLFormElement;
  input: HTMLInputElement;
  ranker: HTMLSelectElement;
  results: HTMLOListElement;
  status: HTMLElement;
}

/** Announce one line above the results. Empty text clears the region. */
export function setStatus(shell: SearchShell, text: string): void {
  shell.status.textContent = text;
}

/**
 * Find the hero's elements. Returns null when the page has no hero, so the
 * chrome can mount on a page that does not carry one.
 */
export function findShell(): SearchShell | null {
  const form = document.querySelector<HTMLFormElement>("#q-form");
  const input = document.querySelector<HTMLInputElement>("#q");
  const ranker = document.querySelector<HTMLSelectElement>("#ranker");
  const results = document.querySelector<HTMLOListElement>("#results");
  const status = document.querySelector<HTMLElement>("#q-status");
  if (!form || !input || !ranker || !results || !status) return null;
  return { form, input, ranker, results, status };
}

/**
 * The index is fetched at most once per page, and a second query arriving
 * while the first is still downloading waits on the same promise rather than
 * starting a second 1.25 MB download.
 */
let pending: Promise<SearchIndex> | null = null;
let index: SearchIndex | null = null;

function ensureIndex(): Promise<SearchIndex> {
  if (index) return Promise.resolve(index);
  pending ??= loadIndex()
    .then((loaded) => {
      index = loaded;
      return loaded;
    })
    .catch((err: unknown) => {
      // A failed fetch must not poison every later attempt, or the retry
      // button would be unable to succeed.
      pending = null;
      throw err;
    });
  return pending;
}

/**
 * How big the download is, in MB, for the line the reader sees before paying
 * for it. `claims.json` has already filled the `.index-note` span by the time
 * anyone can type, so the usual path costs no request; `meta.json` is the
 * fallback for a page whose claims failed to load.
 */
async function indexSizeMb(): Promise<string | null> {
  const printed = document
    .querySelector<HTMLElement>('.index-note [data-stat^="index_gz_mb"]')
    ?.textContent?.trim();
  if (printed && /^[\d.]+$/.test(printed)) return printed;
  try {
    const meta = await loadJson<Meta>("meta.json");
    return meta.index_gz_mb.toFixed(1);
  } catch {
    return null;
  }
}

function rankerOf(shell: SearchShell): Ranker {
  return isRanker(shell.ranker.value) ? shell.ranker.value : "bm25";
}

/**
 * The one line the reader gets about what just happened. A query of nothing
 * but stopwords is a different failure from a query of real words the corpus
 * has never printed, and saying so is cheaper than leaving them to guess.
 */
function statusFor(
  result: SearchResult,
  query: string,
  ms: number,
): string {
  if (result.terms.length === 0) {
    return "Every word in that query is a stopword, so there is nothing to look up.";
  }
  if (result.hits.length === 0) {
    return "No article contains any of these stems.";
  }
  const n = result.total.toLocaleString("en-US");
  return `${n} articles ranked in ${ms.toFixed(0)} ms for “${query}”`;
}

async function runQuery(shell: SearchShell, query: string): Promise<void> {
  const ranker = rankerOf(shell);
  clearExtra(shell.results);

  if (!index) {
    const mb = await indexSizeMb();
    setStatus(
      shell,
      mb ? `Loading the index (${mb} MB) …` : "Loading the index …",
    );
  }

  let ready: SearchIndex;
  try {
    ready = await ensureIndex();
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    setStatus(shell, "The index could not be loaded.");
    renderError(shell.results, message, () => {
      void runQuery(shell, query);
    });
    return;
  }

  const started = performance.now();
  const result = search(ready, query, ranker, TOP_K);
  const ms = performance.now() - started;

  renderResults(shell.results, result);
  setStatus(shell, statusFor(result, query, ms));
}

/**
 * Wire the hero. `onSearch` replaces the built-in engine, which is only ever
 * useful to a test; the page calls `initSearch()` and gets the real thing.
 */
export function initSearch(
  onSearch?: (query: string, ranker: string) => void,
): SearchShell | null {
  const shell = findShell();
  if (!shell) return null;

  const run = (): void => {
    const query = shell.input.value.trim();
    // Focus stays where the reader is typing, whatever the outcome.
    shell.input.focus();
    if (!query) {
      setStatus(shell, "Type a query first.");
      return;
    }
    if (onSearch) {
      onSearch(query, shell.ranker.value);
      return;
    }
    void runQuery(shell, query);
  };

  shell.form.addEventListener("submit", (e) => {
    e.preventDefault();
    run();
  });

  // A chip is a query, not a link: it fills the box so the reader can see and
  // edit what was asked before it is asked again.
  for (const chip of document.querySelectorAll<HTMLButtonElement>(".chip")) {
    chip.addEventListener("click", () => {
      shell.input.value = chip.dataset.q ?? chip.textContent ?? "";
      run();
    });
  }

  // Changing the weighting re-runs the query that is already on screen, which
  // is the only way to see what the weighting actually does.
  shell.ranker.addEventListener("change", () => {
    if (shell.results.childElementCount > 0 || shell.input.value.trim()) run();
  });

  return shell;
}
