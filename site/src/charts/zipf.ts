import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { fixed, int } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { Vocab } from "../types.ts";
import {
  CONFIG,
  MONO_FAMILY,
  horizontalLegend,
  layoutTemplate,
  series,
} from "./theme.ts";

const NOTE =
  "Source: the Financial Times 1991 collection, stopworded with the course list and Porter stemmed. The dashed line is a least-squares fit to the head of the distribution.";

/** How many of the heaviest stems are named beside the curve. */
const N_LABELLED = 10;

/**
 * The names sit in a column in the empty lower-left of the plot and point up
 * at their own marker, because ranks one to ten fall inside a single decade of
 * a log axis and no label can be parked beside its point without covering the
 * next one. These are log-space coordinates: Plotly wants the log of the value
 * for an annotation on a log axis. The axis is extended one decade to the left
 * of rank one to make room for the column.
 */
const X_MIN = -1.15;
const LABEL_X = -0.02;
const LABEL_TOP_GAP = 0.55;
const LABEL_STEP = 0.255;

/** Decade tick values and their text; Plotly ignores any outside the range. */
function decadeTicks(): { vals: number[]; text: string[] } {
  const vals = [0, 1, 2, 3, 4, 5, 6, 7].map((e) => 10 ** e);
  return {
    vals,
    text: vals.map((v) =>
      v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : String(v),
    ),
  };
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

  const points = vocab.zipf;
  const head = points[0];
  const tail = points[points.length - 1];
  if (!head || !tail) {
    showError(container, "The vocabulary file carries no Zipf points.", () =>
      void render(container),
    );
    return;
  }
  const { slope, intercept } = vocab.zipf_fit;
  const exponent = fixed(-slope, 2);
  const fitAt = (rank: number): number => 10 ** intercept * rank ** slope;
  const labelled = points.filter((p) => p.rank <= N_LABELLED);

  // The law the textbook states is 1/rank exactly. Whether this collection is
  // flatter or steeper than that is read off the fit, never assumed.
  const against =
    -slope < 0.95
      ? ", flatter than the textbook 1/rank"
      : -slope > 1.05
        ? ", steeper than the textbook 1/rank"
        : ", close to the textbook 1/rank";

  const spec: FigureSpec = {
    id: "fig-zipf",
    title: `Frequency falls as roughly 1/rank^${exponent}${against}`,
    subtitle: `Collection frequency against rank for ${int(tail.rank)} stems, log-log, sampled at ${int(points.length)} ranks. The ${int(labelled.length)} heaviest stems are named; the dashed line is the fitted power law.`,
    note: NOTE,
    alt: `Log-log scatter of stem frequency against rank. The heaviest stem, "${head.term}", occurs ${int(head.freq)} times; frequency falls away roughly as one over rank to the power ${exponent} and reaches a single occurrence by rank ${int(tail.rank)}.`,
    table: {
      columns: ["Rank", "Stem", "Occurrences", "Fitted occurrences"],
      rows: points.map((p) => [
        int(p.rank),
        p.term,
        int(p.freq),
        int(fitAt(p.rank)),
      ]),
    },
  };

  const plot = mountFigure(container, spec);
  const ticks = decadeTicks();
  const xMax = Math.log10(tail.rank) + 0.12;
  const yMax = Math.log10(head.freq) + 0.28;

  function draw(): Promise<unknown> {
    const c = series();
    const ink2 = cssVar("--ink-2");

    const observed: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "lines+markers",
      name: "Stem frequency",
      x: points.map((p) => p.rank),
      y: points.map((p) => p.freq),
      text: points.map((p) => p.term),
      line: { color: c.s1, width: 1.4 },
      marker: { color: c.s1, size: 4 },
      hovertemplate:
        "rank %{x:,}: %{text}<br><b>%{y:,} occurrences</b><extra></extra>",
    };
    const top: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "markers",
      name: `${int(labelled.length)} heaviest stems`,
      x: labelled.map((p) => p.rank),
      y: labelled.map((p) => p.freq),
      text: labelled.map((p) => p.term),
      marker: { color: c.accent, size: 8 },
      hovertemplate:
        "rank %{x:,}: %{text}<br><b>%{y:,} occurrences</b><extra></extra>",
    };
    const fit: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "lines",
      name: `Fitted rank^${fixed(slope, 2)}`,
      x: [head.rank, tail.rank],
      y: [fitAt(head.rank), fitAt(tail.rank)],
      line: { color: c.s2, width: 1.4, dash: "dash" },
      hoverinfo: "skip",
    };

    const labelTop = Math.log10(head.freq) - LABEL_TOP_GAP;
    const annotations: Partial<Plotly.Annotations>[] = labelled.map((p, i) => ({
      x: Math.log10(p.rank),
      y: Math.log10(p.freq),
      ax: LABEL_X,
      ay: labelTop - i * LABEL_STEP,
      axref: "x",
      ayref: "y",
      text: p.term,
      xanchor: "right",
      showarrow: true,
      arrowhead: 2,
      arrowsize: 0.8,
      arrowwidth: 0.8,
      arrowcolor: c.grid,
      standoff: 5,
      font: { family: MONO_FAMILY, size: 11, color: ink2 },
    }));

    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      annotations,
      showlegend: true,
      legend: horizontalLegend(1.03),
      margin: { ...base.margin, l: 8, t: 30, b: 52 },
      hovermode: "closest",
      xaxis: {
        ...base.xaxis,
        type: "log",
        tickmode: "array",
        tickvals: ticks.vals,
        ticktext: ticks.text,
        range: [X_MIN, xMax],
        automargin: true,
        title: { text: "Rank of the stem", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        type: "log",
        tickmode: "array",
        tickvals: ticks.vals,
        ticktext: ticks.text,
        range: [-0.18, yMax],
        automargin: true,
        title: { text: "Occurrences in the collection", standoff: 8 },
      },
    };
    return Plotly.react(plot, [observed, top, fit], layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
