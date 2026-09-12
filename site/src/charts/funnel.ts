import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { int } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { FunnelStep, Vocab } from "../types.ts";
import { CONFIG, layoutTemplate, reversed, series } from "./theme.ts";

const NOTE =
  "Source: the Financial Times 1991 collection (TREC disk 4), counted by the rebuilt course pipeline.";

/**
 * The steps the claim is about. Named by `step` rather than by position, with
 * a named fallback, so a pipeline that inserts a step in the middle still
 * produces a sentence about stemming.
 */
function stepAt(
  steps: FunnelStep[],
  key: string,
  fallback: FunnelStep,
): FunnelStep {
  return steps.find((s) => s.step === key) ?? fallback;
}

/** The step that folds away the most word forms, read off the data. */
function biggestFold(steps: FunnelStep[]): { step: FunnelStep; drop: number } {
  let best = { step: steps[0] as FunnelStep, drop: 0 };
  steps.forEach((step, i) => {
    const prev = steps[i - 1];
    if (!prev) return;
    const drop = prev.n_types - step.n_types;
    if (drop > best.drop) best = { step, drop };
  });
  return best;
}

export async function render(container: HTMLElement): Promise<void> {
  let vocab: Vocab;
  try {
    vocab = await loadJson<Vocab>("vocab.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error
        ? err.message
        : "Could not load the vocabulary data.",
      () => void render(container),
    );
    return;
  }

  const steps = vocab.funnel;
  const first = steps[0];
  const last = steps[steps.length - 1];
  if (!first || !last) {
    showError(container, "The vocabulary file carries no funnel steps.", () =>
      void render(container),
    );
    return;
  }
  const beforeStem = stepAt(
    steps,
    "minus_stopwords",
    steps[steps.length - 2] ?? first,
  );
  const stemmed = stepAt(steps, "stemmed", last);
  const fold = biggestFold(steps);

  const spec: FigureSpec = {
    id: "fig-funnel",
    title: `Stemming folds ${int(beforeStem.n_types)} word forms into ${int(stemmed.n_types)} stems`,
    subtitle: `Distinct word forms left after each step of the tokeniser. Tokens fall from ${int(first.n_tokens)} to ${int(last.n_tokens)} over the same ${int(steps.length)} steps; both counts are in the table.`,
    note: NOTE,
    alt: `Horizontal bar chart of ${int(steps.length)} tokenising steps. Distinct word forms fall from ${int(first.n_types)} after "${first.label}" to ${int(stemmed.n_types)} after "${stemmed.label}". The largest single fold is at "${fold.step.label}", which takes ${int(fold.drop)} word forms out.`,
    table: {
      columns: ["Step", "Word forms", "Tokens"],
      rows: steps.map((s) => [s.label, int(s.n_types), int(s.n_tokens)]),
    },
  };

  const plot = mountFigure(container, spec);
  const maxTypes = Math.max(...steps.map((s) => s.n_types));

  function draw(): Promise<unknown> {
    const c = series();
    const ink2 = cssVar("--ink-2");
    const rows = reversed(steps);
    const trace: Partial<Plotly.PlotData> = {
      type: "bar",
      orientation: "h",
      y: rows.map((s) => s.label),
      x: rows.map((s) => s.n_types),
      customdata: rows.map((s) => s.n_tokens),
      text: rows.map((s) => int(s.n_types)),
      textposition: "outside",
      textfont: { color: ink2, size: 12 },
      cliponaxis: false,
      hovertemplate:
        "%{y}<br><b>%{x:,} word forms</b><br>%{customdata:,} tokens<extra></extra>",
      marker: { color: c.s1 },
      width: 0.6,
    };

    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      bargap: 0.3,
      margin: { ...base.margin, r: 72, b: 48 },
      xaxis: {
        ...base.xaxis,
        rangemode: "tozero",
        range: [0, maxTypes * 1.12],
        // Short SI ticks ("20k") stay upright at phone width, where the full
        // counts would have to rotate; the bar labels carry the exact numbers.
        exponentformat: "SI",
        tickangle: 0,
        title: { text: "Distinct word forms", standoff: 8 },
      },
      yaxis: { ...base.yaxis, automargin: true, showgrid: false },
    };
    return Plotly.react(plot, [trace], layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
