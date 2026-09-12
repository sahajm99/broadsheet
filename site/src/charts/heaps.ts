import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { fixed, int } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { onThemeChange } from "../theme.ts";
import type { Vocab } from "../types.ts";
import {
  CONFIG,
  horizontalLegend,
  layoutTemplate,
  series,
} from "./theme.ts";

const NOTE =
  "Source: the Financial Times 1991 collection, stopwords removed and no stemming. The dashed line is a least-squares fit on log-log axes; the observed curve bends away from it at both ends, so the exponent is an average over the run, not a law.";

/** 1000 -> "1k", 1000000 -> "1M": compact tick text for a log axis. */
function siLabel(v: number): string {
  if (v >= 1e6) return `${v / 1e6}M`;
  if (v >= 1e3) return `${v / 1e3}k`;
  return String(v);
}

/**
 * Tick values for a log axis. Plotly's own "D2" minor ticks label a decade's
 * subdivisions with a bare mantissa ("5" for five thousand), which reads as a
 * mistake beside an SI decade label, so the ticks are written out instead.
 * Values outside the drawn range are ignored by Plotly.
 */
function logTicks(mantissas: number[]): { vals: number[]; text: string[] } {
  const vals: number[] = [];
  for (let e = 0; e <= 7; e += 1) {
    for (const m of mantissas) vals.push(m * 10 ** e);
  }
  return { vals, text: vals.map(siLabel) };
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

  const points = vocab.heaps;
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last) {
    showError(container, "The vocabulary file carries no Heaps points.", () =>
      void render(container),
    );
    return;
  }
  const { k, beta } = vocab.heaps_fit;
  const fitAt = (tokens: number): number => k * tokens ** beta;

  const spec: FigureSpec = {
    id: "fig-heaps",
    title: `New words keep arriving: vocabulary grows as tokens^${fixed(beta, 2)}`,
    subtitle: `Distinct word forms against tokens read, ${int(points.length)} checkpoints on log-log axes. The dashed line is the fitted power law, ${fixed(k, 2)} x tokens^${fixed(beta, 2)}.`,
    note: NOTE,
    alt: `Log-log line chart. Distinct word forms rise from ${int(first.n_types)} after ${int(first.n_tokens)} tokens to ${int(last.n_types)} after ${int(last.n_tokens)} tokens, and the curve is still climbing at the end of the collection. A dashed least-squares line of slope ${fixed(beta, 2)} runs beside it.`,
    table: {
      columns: ["Tokens read", "Word forms", "Fitted word forms"],
      rows: points.map((p) => [
        int(p.n_tokens),
        int(p.n_types),
        int(fitAt(p.n_tokens)),
      ]),
    },
  };

  const plot = mountFigure(container, spec);
  const xTicks = logTicks([1]);
  const yTicks = logTicks([1, 2, 5]);

  function draw(): Promise<unknown> {
    const c = series();
    const observed: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "lines+markers",
      name: "Word forms seen",
      x: points.map((p) => p.n_tokens),
      y: points.map((p) => p.n_types),
      line: { color: c.s1, width: 1.6 },
      marker: { color: c.s1, size: 4 },
      hovertemplate:
        "<b>%{y:,} word forms</b><br>after %{x:,} tokens<extra></extra>",
    };
    const fit: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "lines",
      name: `Fitted ${fixed(k, 2)} x tokens^${fixed(beta, 2)}`,
      x: [first.n_tokens, last.n_tokens],
      y: [fitAt(first.n_tokens), fitAt(last.n_tokens)],
      line: { color: c.s2, width: 1.4, dash: "dash" },
      hoverinfo: "skip",
    };

    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: true,
      legend: horizontalLegend(1.03),
      margin: { ...base.margin, l: 8, t: 30, b: 52 },
      hovermode: "closest",
      xaxis: {
        ...base.xaxis,
        type: "log",
        tickmode: "array",
        tickvals: xTicks.vals,
        ticktext: xTicks.text,
        automargin: true,
        title: { text: "Tokens read", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        type: "log",
        tickmode: "array",
        tickvals: yTicks.vals,
        ticktext: yTicks.text,
        automargin: true,
        title: { text: "Distinct word forms", standoff: 8 },
      },
    };
    return Plotly.react(plot, [observed, fit], layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
