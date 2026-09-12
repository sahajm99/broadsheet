/**
 * The shared frame around every chart: claim, context line, plot, the same
 * rows as a table for keyboard and screen-reader readers, and a source note.
 */
export interface FigureSpec {
  id: string;
  /** The claim. Also the chart's accessible name, via `alt`. */
  title: string;
  subtitle: string;
  note: string;
  alt: string;
  table: { columns: string[]; rows: (string | number)[][] };
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function buildTable(spec: FigureSpec): HTMLDetailsElement {
  const details = el("details", "fig-table");
  details.appendChild(el("summary", undefined, "Show the numbers"));

  const table = el("table");
  // The claim is already the heading above the figure; the caption only has to
  // say what the table is.
  table.appendChild(el("caption", undefined, "Data behind this figure"));

  const thead = el("thead");
  const headRow = el("tr");
  for (const name of spec.table.columns) {
    const th = el("th", undefined, name);
    th.scope = "col";
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const row of spec.table.rows) {
    const tr = el("tr");
    row.forEach((cell, i) => {
      const text = String(cell);
      if (i === 0) {
        const th = el("th", undefined, text);
        th.scope = "row";
        tr.appendChild(th);
      } else {
        tr.appendChild(el("td", undefined, text));
      }
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  details.appendChild(table);
  return details;
}

/**
 * Replace `container`'s content with the figure frame and return the div the
 * chart module should hand to Plotly.
 */
export function mountFigure(
  container: HTMLElement,
  spec: FigureSpec,
): HTMLDivElement {
  container.replaceChildren();

  const title = el("h3", "fig-title", spec.title);
  title.id = `${spec.id}-title`;
  container.appendChild(title);
  container.appendChild(el("p", "fig-sub", spec.subtitle));

  const plot = el("div", "plot");
  plot.id = `${spec.id}-plot`;
  plot.setAttribute("role", "img");
  plot.setAttribute("aria-label", spec.alt);
  // A wrapper the plot can scroll inside, so a chart too wide for a phone
  // scrolls on its own without taking the claim and the table with it.
  const scroll = el("div", "plot-scroll");
  scroll.appendChild(plot);
  container.appendChild(scroll);

  container.appendChild(buildTable(spec));
  container.appendChild(el("p", "fig-note", spec.note));

  return plot;
}

/**
 * Where a control that belongs to a figure is inserted: before the plot's
 * scroll wrapper, so the control never scrolls sideways with the plot.
 */
export function controlSlot(plot: HTMLElement): HTMLElement {
  return plot.parentElement ?? plot;
}

/**
 * Re-state a mounted figure's claim, context line, accessible name and table
 * for new data, leaving the plot div and any control beside it in place, so a
 * picker keeps its focus and Plotly keeps its canvas across a change.
 */
export function updateFigure(container: HTMLElement, spec: FigureSpec): void {
  const title = container.querySelector(".fig-title");
  if (title) title.textContent = spec.title;
  const sub = container.querySelector(".fig-sub");
  if (sub) sub.textContent = spec.subtitle;
  container.querySelector(".plot")?.setAttribute("aria-label", spec.alt);
  const old = container.querySelector("details.fig-table");
  if (!old) return;
  const next = buildTable(spec);
  next.open = (old as HTMLDetailsElement).open;
  old.replaceWith(next);
}

/** A failed fetch replaces one figure, never the story around it. */
export function showError(
  container: HTMLElement,
  message: string,
  retry: () => void,
): void {
  container.replaceChildren();
  // The live region is mounted empty and filled afterwards, so a screen reader
  // announces the message as a change rather than missing it in the markup it
  // already read.
  const box = el("div", "fig-error");
  box.setAttribute("role", "status");
  container.appendChild(box);
  box.appendChild(el("p", undefined, message));
  const btn = el("button", "fig-retry", "Try again");
  btn.type = "button";
  btn.addEventListener("click", retry);
  box.appendChild(btn);
}
