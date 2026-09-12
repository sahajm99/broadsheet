import "@fontsource-variable/ibm-plex-sans/wght.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/search.css";
import "./styles/figure.css";
import "./styles/panel.css";

import { loadJson } from "./data.ts";
import { el } from "./figure.ts";
import { whenVisible } from "./lazy.ts";
import { initNav } from "./nav.ts";
import { initSearch } from "./search.ts";
import { fillStats } from "./stats-fill.ts";
import { initTheme, toggleTheme } from "./theme.ts";
import type { Claims } from "./types.ts";

type Render = (container: HTMLElement) => Promise<void>;

/**
 * One entry per `data-chart` name in index.html, and one dynamic import per
 * entry, so Plotly loads only when a figure that needs it scrolls into view.
 */
const CHARTS: Record<string, Render | undefined> = {
  funnel: (c) => import("./charts/funnel.ts").then((m) => m.render(c)),
  heaps: (c) => import("./charts/heaps.ts").then((m) => m.render(c)),
  zipf: (c) => import("./charts/zipf.ts").then((m) => m.render(c)),
  lengths: (c) => import("./charts/lengths.ts").then((m) => m.render(c)),
  topterms: (c) => import("./charts/topterms.ts").then((m) => m.render(c)),
  rankers: (c) => import("./charts/rankers.ts").then((m) => m.render(c)),
  apstrip: (c) => import("./charts/apstrip.ts").then((m) => m.render(c)),
  paired: (c) => import("./charts/paired.ts").then((m) => m.render(c)),
  prcurves: (c) => import("./charts/prcurves.ts").then((m) => m.render(c)),
  fields: (c) => import("./charts/fields.ts").then((m) => m.render(c)),
  grid: (c) => import("./charts/grid.ts").then((m) => m.render(c)),
  treatments: (c) => import("./charts/treatments.ts").then((m) => m.render(c)),
};

const NOTICE_ID = "claims-notice";

/**
 * Every number quoted in the prose comes from claims.json, so a failure to
 * load it leaves placeholders in the middle of sentences. That is worth a
 * visible notice at the top of the page, not only a console line.
 */
function showClaimsNotice(): void {
  const main = document.querySelector("main");
  if (!main || document.getElementById(NOTICE_ID)) return;
  const box = el("div", "fig-error page-notice");
  box.id = NOTICE_ID;
  box.setAttribute("role", "status");
  // Mounted empty, then filled, so the message is announced as a change.
  main.prepend(box);
  box.appendChild(
    el(
      "p",
      undefined,
      "The numbers quoted in the text below could not be loaded, so they are showing as placeholders. The charts fetch their own data and are not affected.",
    ),
  );
  const btn = el("button", "fig-retry", "Try again");
  btn.type = "button";
  btn.addEventListener("click", fillNumbers);
  box.appendChild(btn);
}

function fillNumbers(): void {
  loadJson<Claims>("claims.json")
    .then((claims) => {
      fillStats(claims.values);
      document.getElementById(NOTICE_ID)?.remove();
    })
    .catch((err: unknown) => {
      console.error("Could not fill the numbers in the prose", err);
      showClaimsNotice();
    });
}

initTheme();
initNav();
document.getElementById("theme-toggle")?.addEventListener("click", toggleTheme);

initSearch();
fillNumbers();

for (const container of document.querySelectorAll<HTMLElement>("[data-chart]")) {
  const name = container.dataset.chart ?? "";
  const render = CHARTS[name];
  if (!render) {
    console.warn(`main: no chart registered for "${name}"`);
    continue;
  }
  whenVisible(container, () => {
    render(container).catch((err: unknown) => {
      console.error(`main: chart "${name}" failed`, err);
    });
  });
}
