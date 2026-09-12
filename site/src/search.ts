/**
 * The search hero's shell. The form, the ranker switch and the example chips
 * are wired to each other and to the status line here; the engine that turns
 * a query into results arrives in the next task, and until it does a submit
 * says so rather than doing nothing.
 */

const PENDING = "Search arrives in the next task";

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
 * Wire the hero. `onSearch` is what Task 7 supplies; without it every path
 * that would run a query reports that the engine is not here yet.
 */
export function initSearch(
  onSearch?: (query: string, ranker: string) => void,
): SearchShell | null {
  const shell = findShell();
  if (!shell) return null;

  const run = (): void => {
    const query = shell.input.value.trim();
    if (!query) {
      setStatus(shell, "Type a query first.");
      shell.input.focus();
      return;
    }
    if (!onSearch) {
      setStatus(shell, PENDING);
      return;
    }
    onSearch(query, shell.ranker.value);
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

  // Changing the weighting re-runs the query that is already on screen.
  shell.ranker.addEventListener("change", () => {
    if (shell.results.childElementCount > 0 || shell.input.value.trim()) run();
  });

  return shell;
}
