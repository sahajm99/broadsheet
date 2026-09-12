import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { ciText, int, score, signed } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { Comparisons, Runs, Treatments } from "../types.ts";
import {
  CONFIG,
  MONO_FAMILY,
  errorBars,
  layoutTemplate,
  rankerColor,
} from "./theme.ts";

export async function render(container: HTMLElement): Promise<void> {
  let treatments: Treatments;
  let comparisons: Comparisons;
  let runs: Runs;
  try {
    [treatments, comparisons, runs] = await Promise.all([
      loadJson<Treatments>("treatments.json"),
      loadJson<Comparisons>("comparisons.json"),
      loadJson<Runs>("runs.json"),
    ]);
  } catch (err) {
    showError(
      container,
      err instanceof Error
        ? err.message
        : "Could not load the index treatments.",
      () => void render(container),
    );
    return;
  }

  const rows = treatments.rows;
  const stem = comparisons.pairs.find((p) => p.key === "stem_vs_nostem");
  if (!rows.length || !stem) {
    showError(
      container,
      "The treatments file is missing the rows this chart compares.",
      () => void render(container),
    );
    return;
  }

  const stop = comparisons.pairs.find((p) => p.key === "stop_vs_nostop");
  const title = `Stemming changes MAP by ${signed(stem.mean_diff)} (${signed(stem.lo)} to ${signed(stem.hi)})`;
  const stopSentence = stop
    ? ` Removing stopwords moves it by ${signed(stop.mean_diff)} (${signed(stop.lo)} to ${signed(stop.hi)}) and leaves ${int(stop.ties)} of ${int(runs.n_topics)} topics scoring exactly the same either way.`
    : "";

  const spec: FigureSpec = {
    id: "fig-treatments",
    title,
    subtitle: `${treatments.ranker === "bm25" ? "BM25" : treatments.ranker} on title queries over ${int(rows.length)} index treatments, mean average precision with a 95% bootstrap interval over ${int(runs.n_topics)} topics.${stopSentence}`,
    note: "Source: FT 1991 slice of TREC disk 4, judged by the NIST Robust 2004 qrels; beside each treatment is how many distinct terms it leaves in the index.",
    alt: `Dot plot. Mean average precision for BM25 on title queries over ${int(rows.length)} index treatments crossing stemming with stopword removal, each with a 95% bootstrap interval, and each treatment's vocabulary size. ${title}.`,
    table: {
      columns: [
        "Treatment",
        "Distinct terms",
        "MAP",
        "95% interval",
        "P@10",
      ],
      rows: rows.map((r) => [
        r.label,
        int(r.n_types),
        score(r.map.mean),
        ciText(r.map.lo, r.map.hi),
        score(r.p10.mean),
      ]),
    },
  };

  const plot = mountFigure(container, spec);

  function draw(): Promise<unknown> {
    const color = rankerColor(treatments.ranker);
    const ink = cssVar("--ink");
    const ink2 = cssVar("--ink-2");
    const muted = cssVar("--muted");
    const mean = rows.map((r) => r.map.mean);

    const trace: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "text+markers",
      showlegend: false,
      x: mean,
      y: rows.map((_, i) => i),
      text: rows.map((r) => score(r.map.mean)),
      textposition: "top center",
      textfont: { color: ink, family: MONO_FAMILY, size: 12 },
      customdata: rows.map((r) => [r.map.lo, r.map.hi, r.label]),
      hovertemplate:
        "<b>%{x:.3f}</b> MAP<br>95% interval %{customdata[0]:.3f} to %{customdata[1]:.3f}<extra>%{customdata[2]}</extra>",
      marker: { size: 9, color },
      error_x: {
        ...errorBars(
          mean,
          rows.map((r) => r.map.lo),
          rows.map((r) => r.map.hi),
          color,
        ),
        width: 4,
      },
    };

    // On a phone the right-hand gutter would be narrower than the label that
    // goes in it, so the vocabulary size moves under the treatment's name and
    // the plot gets the width back.
    const narrow = plot.clientWidth < 560;

    // The vocabulary size is a second column rather than a second axis: it is
    // printed at the right edge, in a gutter the x range leaves free.
    const annotations: Partial<Plotly.Annotations>[] = narrow
      ? []
      : rows.map((r, i) => ({
          xref: "paper" as const,
          yref: "y" as const,
          x: 1,
          y: i,
          xanchor: "right" as const,
          yanchor: "middle" as const,
          showarrow: false,
          text: int(r.n_types),
          font: { color: ink2, family: MONO_FAMILY, size: 12 },
        }));
    if (!narrow) {
      annotations.push({
        xref: "paper",
        yref: "paper",
        x: 1,
        y: 1,
        xanchor: "right",
        yanchor: "bottom",
        showarrow: false,
        text: "distinct terms",
        font: { color: muted, size: 11 },
      });
    }

    const hi = Math.max(...rows.map((r) => r.map.hi));
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      annotations,
      showlegend: false,
      hovermode: "closest",
      margin: { ...base.margin, t: 26, r: 24, b: 48 },
      xaxis: {
        ...base.xaxis,
        range: [0, hi * (narrow ? 1.12 : 1.5)],
        zeroline: false,
        title: { text: "Mean average precision", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        // Inverted so the file's first treatment, the one the rest of the
        // page uses, is the top row.
        range: [rows.length - 0.5, -0.5],
        automargin: true,
        showgrid: false,
        zeroline: false,
        tickmode: "array",
        tickvals: rows.map((_, i) => i),
        ticktext: rows.map((r) =>
          narrow ? `${r.label}<br>${int(r.n_types)} terms` : r.label,
        ),
        tickfont: { color: ink2, size: narrow ? 11 : 13 },
      },
    };
    return Plotly.react(plot, [trace], layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
