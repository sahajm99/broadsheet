/**
 * Rendering for the search hero: the result list, the explain panel that shows
 * the arithmetic behind every ranking, and the error block.
 *
 * The explain panel is the point of the whole exercise (W1). Every engine can
 * show ten headlines; almost none will tell you that `tunnel` carried 71% of
 * the score because it appears in 214 of 5,368 articles and six times in this
 * one. The numbers here are the same numbers the ranker added up, not a
 * re-derivation, so a row that looks wrong is a ranker that is wrong.
 */

import { el } from "../figure.ts";
import { fixed, int, isoDate, score as fmtScore } from "../fmt.ts";
import type { ExplainRow, Hit, SearchResult } from "./rank.ts";

const COLUMNS = [
  "stem",
  "in query (qtf)",
  "in article (tf)",
  "articles with it (df)",
  "idf",
  "share of score",
];

/** `1` -> `"01"`, so the ranks line up in a mono column. */
function rankLabel(i: number): string {
  return String(i + 1).padStart(2, "0");
}

function explainTable(hit: Hit): HTMLTableElement {
  const table = el("table", "explain-table");
  table.appendChild(
    el(
      "caption",
      undefined,
      "Every query stem this article contains, and what it added to the score",
    ),
  );

  const head = el("tr");
  COLUMNS.forEach((name, i) => {
    const th = el("th", i === 0 ? undefined : "num", name);
    th.scope = "col";
    head.appendChild(th);
  });
  const thead = el("thead");
  thead.appendChild(head);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const row of hit.explain) {
    tbody.appendChild(explainRow(row, hit.score));
  }
  table.appendChild(tbody);
  return table;
}

function explainRow(row: ExplainRow, total: number): HTMLTableRowElement {
  const tr = el("tr");
  const term = el("th", "stem", row.term);
  term.scope = "row";
  tr.appendChild(term);
  tr.appendChild(el("td", "num", String(row.qtf)));
  tr.appendChild(el("td", "num", String(row.tf)));
  tr.appendChild(el("td", "num", int(row.df)));
  tr.appendChild(el("td", "num", fixed(row.idf, 3)));
  const share =
    total > 0
      ? `${fixed(row.contribution, 3)} (${((row.contribution / total) * 100).toFixed(0)}%)`
      : fixed(row.contribution, 3);
  tr.appendChild(el("td", "num share", share));
  return tr;
}

/** "Query terms after stopwords and stemming: …", and what was not found. */
function termNotes(result: SearchResult): HTMLElement[] {
  const notes: HTMLElement[] = [];
  notes.push(
    el(
      "p",
      "explain-note",
      `Query terms after stopwords and stemming: ${
        result.terms.length > 0 ? result.terms.join(", ") : "none"
      }`,
    ),
  );
  if (result.unknown.length > 0) {
    notes.push(
      el("p", "explain-note", `Not in the index: ${result.unknown.join(", ")}`),
    );
  }
  return notes;
}

function explainPanel(hit: Hit, result: SearchResult): HTMLDetailsElement {
  const details = el("details", "result-why");
  details.appendChild(el("summary", undefined, "Why this ranked here"));
  const body = el("div", "result-why-body");
  body.appendChild(explainTable(hit));
  for (const note of termNotes(result)) body.appendChild(note);
  details.appendChild(body);
  return details;
}

function resultItem(hit: Hit, i: number, result: SearchResult): HTMLLIElement {
  const li = el("li");
  li.appendChild(el("p", "result-rank", rankLabel(i)));
  li.appendChild(el("strong", "result-headline", hit.headline));

  const meta = el("p", "result-meta");
  const date = el("span", "date", isoDate(hit.date));
  meta.appendChild(date);
  meta.appendChild(el("span", "docno", hit.no));
  const score = el("span", "score", fmtScore(hit.score));
  score.title = "Score under the selected weighting";
  meta.appendChild(score);
  li.appendChild(meta);

  li.appendChild(explainPanel(hit, result));
  return li;
}

/**
 * The block below the list: it carries the stem notes when nothing matched,
 * and the error message when the index could not be fetched. Created on first
 * use because `index.html` belongs to the page shell, not to this module.
 */
export function extraBlock(results: HTMLOListElement): HTMLElement {
  const existing = results.parentElement?.querySelector<HTMLElement>(".q-extra");
  if (existing) return existing;
  const block = el("div", "q-extra");
  results.insertAdjacentElement("afterend", block);
  return block;
}

export function clearExtra(results: HTMLOListElement): void {
  extraBlock(results).replaceChildren();
}

/** Fill the ordered list with up to `k` citations, each with its arithmetic. */
export function renderResults(
  results: HTMLOListElement,
  result: SearchResult,
): void {
  const extra = extraBlock(results);
  extra.replaceChildren();
  results.replaceChildren(
    ...result.hits.map((hit, i) => resultItem(hit, i, result)),
  );
  // Nothing matched: the status line has already said so, so what belongs
  // here is the evidence -- what the query actually became, and which of its
  // stems this corpus has never printed.
  if (result.hits.length === 0) {
    for (const note of termNotes(result)) extra.appendChild(note);
  }
}

/** An error with a way out of it, in the same shape the charts use. */
export function renderError(
  results: HTMLOListElement,
  message: string,
  onRetry: () => void,
): void {
  results.replaceChildren();
  const extra = extraBlock(results);
  const box = el("div", "fig-error");
  box.appendChild(el("p", undefined, message));
  const retry = el("button", "fig-retry", "Try again");
  retry.type = "button";
  retry.addEventListener("click", onRetry);
  box.appendChild(retry);
  extra.replaceChildren(box);
}
