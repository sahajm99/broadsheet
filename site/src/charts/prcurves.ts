import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { fixed, int, score } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { onThemeChange } from "../theme.ts";
import type { PrCurves, Runs } from "../types.ts";
import {
  CONFIG,
  horizontalLegend,
  layoutTemplate,
  rankerColor,
} from "./theme.ts";

interface Curve {
  ranker: string;
  label: string;
  precision: number[];
}

export async function render(container: HTMLElement): Promise<void> {
  let pr: PrCurves;
  let runs: Runs;
  try {
    [pr, runs] = await Promise.all([
      loadJson<PrCurves>("pr_curves.json"),
      loadJson<Runs>("runs.json"),
    ]);
  } catch (err) {
    showError(
      container,
      err instanceof Error
        ? err.message
        : "Could not load the precision-recall curves.",
      () => void render(container),
    );
    return;
  }

  const levels = pr.recall_levels;
  const curves: Curve[] = [];
  for (const r of runs.rankers) {
    const precision = pr.curves[`${r.key}|title`];
    if (!precision || precision.length !== levels.length) continue;
    curves.push({ ranker: r.key, label: r.label, precision });
  }
  const bm25 = curves.find((c) => c.ranker === "bm25");
  const raw = curves.find((c) => c.ranker === "tfidf_raw");
  if (!levels.length || !bm25 || !raw) {
    showError(
      container,
      "The curve file carries no title-query curves for these rankers.",
      () => void render(container),
    );
    return;
  }

  const title = `At recall 0 BM25 reaches ${score(bm25.precision[0] ?? 0)} precision, the course weighting ${score(raw.precision[0] ?? 0)}`;

  const spec: FigureSpec = {
    id: "fig-prcurves",
    title,
    subtitle: `Eleven-point interpolated precision, averaged over ${int(runs.n_topics)} topics, title queries. At each recall level a run is credited with the best precision it reaches at or beyond that level, which is why every curve starts flat and only falls.`,
    note: `Source: FT 1991 slice of TREC disk 4, judged by the NIST Robust 2004 qrels; unjudged articles count as not relevant, so every curve is a lower bound.`,
    alt: `Line chart. Interpolated precision at ${int(levels.length)} recall levels for ${int(curves.length)} rankers on title-only queries. ${title}.`,
    table: {
      columns: ["Recall", ...curves.map((c) => c.label)],
      rows: levels.map((r, i) => [
        fixed(r, 1),
        ...curves.map((c) => score(c.precision[i] ?? 0)),
      ]),
    },
  };

  const plot = mountFigure(container, spec);

  function draw(): Promise<unknown> {
    const traces: Partial<Plotly.PlotData>[] = curves.map((c) => {
      const color = rankerColor(c.ranker);
      return {
        type: "scatter",
        mode: "lines+markers",
        name: c.label,
        x: levels,
        y: c.precision,
        hovertemplate: `Recall %{x:.1f}<br><b>%{y:.3f}</b> precision<extra>${c.label}</extra>`,
        line: { color, width: 2 },
        marker: { size: 6, color },
      };
    });

    const top = Math.max(...curves.flatMap((c) => c.precision));
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: true,
      legend: horizontalLegend(),
      margin: { ...base.margin, t: 34, r: 24, b: 48 },
      hovermode: "closest",
      xaxis: {
        ...base.xaxis,
        range: [-0.02, 1.02],
        dtick: 0.2,
        zeroline: false,
        title: { text: "Recall", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        range: [0, top * 1.1],
        automargin: true,
        zeroline: false,
        title: { text: "Interpolated precision", standoff: 8 },
      },
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
