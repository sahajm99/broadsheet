import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { ciText, int, score, signed } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { Comparison, Comparisons, Runs } from "../types.ts";
import {
  CONFIG,
  errorBars,
  horizontalLegend,
  layoutTemplate,
  rankerColor,
} from "./theme.ts";

/** Half a thousandth: below this an interval bound prints as a flat 0.000. */
const NEAR_ZERO = 0.0005;

function excludesZero(lo: number, hi: number): boolean {
  return lo > 0 || hi < 0;
}

/** The sentence an interval that only just clears zero has to carry with it. */
function hairline(p: Comparison): string {
  return excludesZero(p.lo, p.hi) &&
    Math.min(Math.abs(p.lo), Math.abs(p.hi)) < NEAR_ZERO
    ? ", an interval that clears zero by less than a thousandth and so is compatible with no effect"
    : "";
}

export async function render(container: HTMLElement): Promise<void> {
  let runs: Runs;
  let comparisons: Comparisons;
  try {
    [runs, comparisons] = await Promise.all([
      loadJson<Runs>("runs.json"),
      loadJson<Comparisons>("comparisons.json"),
    ]);
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the run data.",
      () => void render(container),
    );
    return;
  }

  const fields = runs.fields;
  const rankers = runs.rankers;
  const desc = comparisons.pairs.find((p) => p.key === "desc_vs_title");
  const narr = comparisons.pairs.find((p) => p.key === "narr_vs_title");
  if (!fields.length || !rankers.length || !desc) {
    showError(
      container,
      "The run file is missing the fields this chart compares.",
      () => void render(container),
    );
    return;
  }

  const metrics = (ranker: string, field: string) =>
    runs.runs.find((r) => r.ranker === ranker && r.field === field)?.metrics.map;

  const title = `Adding the description changes BM25 MAP by ${signed(desc.mean_diff)} (${signed(desc.lo)} to ${signed(desc.hi)})`;
  const narrSentence = narr
    ? ` Adding the narrative as well moves it by ${signed(narr.mean_diff)} (${signed(narr.lo)} to ${signed(narr.hi)})${hairline(narr)}.`
    : "";

  const tableRows: (string | number)[][] = [];
  for (const r of rankers) {
    for (const f of fields) {
      const m = metrics(r.key, f.key);
      if (!m) continue;
      tableRows.push([r.label, f.label, score(m.mean), ciText(m.lo, m.hi)]);
    }
  }

  const spec: FigureSpec = {
    id: "fig-fields",
    title,
    subtitle: `Mean average precision by which parts of the topic become the query, each with a 95% bootstrap interval over ${int(runs.n_topics)} topics.${narrSentence}`,
    note: `Source: FT 1991 slice of TREC disk 4, judged by the NIST Robust 2004 qrels; a longer query is also a slower one, and neither cost is priced here.`,
    alt: `Dot plot. Mean average precision for ${int(rankers.length)} rankers across ${int(fields.length)} query fields, each with a 95% bootstrap interval. ${title}.`,
    table: {
      columns: ["Ranker", "Query field", "MAP", "95% interval"],
      rows: tableRows,
    },
  };

  const plot = mountFigure(container, spec);

  function draw(): Promise<unknown> {
    // Rankers are drawn as three rows inside each field's slot; the offset is
    // symmetric about the tick so the field label stays centred on its group.
    const step = 0.22;
    const centre = (rankers.length - 1) / 2;
    const traces: Partial<Plotly.PlotData>[] = rankers.map((r, ri) => {
      const color = rankerColor(r.key);
      const cells = fields.map((f, fi) => ({
        y: fi + (ri - centre) * step,
        m: metrics(r.key, f.key),
        label: f.label,
      }));
      const drawable = cells.filter((c) => c.m !== undefined);
      const mean = drawable.map((c) => c.m!.mean);
      return {
        type: "scatter",
        mode: "markers",
        name: r.label,
        x: mean,
        y: drawable.map((c) => c.y),
        customdata: drawable.map((c) => [c.label, c.m!.lo, c.m!.hi]),
        hovertemplate: `%{customdata[0]}<br><b>%{x:.3f}</b> MAP<br>95% interval %{customdata[1]:.3f} to %{customdata[2]:.3f}<extra>${r.label}</extra>`,
        marker: { size: 9, color },
        error_x: {
          ...errorBars(
            mean,
            drawable.map((c) => c.m!.lo),
            drawable.map((c) => c.m!.hi),
            color,
          ),
          width: 4,
        },
      };
    });

    const hi = Math.max(
      ...runs.runs.map((r) => r.metrics.map.hi),
    );
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: true,
      legend: horizontalLegend(),
      hovermode: "closest",
      margin: { ...base.margin, t: 34, r: 24, b: 48 },
      xaxis: {
        ...base.xaxis,
        range: [0, hi * 1.1],
        zeroline: false,
        title: { text: "Mean average precision", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        // Inverted, so the shortest query is the top row and the reader goes
        // down the chart as the query gets longer.
        range: [fields.length - 0.5, -0.5],
        automargin: true,
        showgrid: false,
        zeroline: false,
        tickmode: "array",
        tickvals: fields.map((_, i) => i),
        ticktext: fields.map((f) => f.label),
        tickfont: { color: cssVar("--ink-2") },
      },
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
