import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { int, pct0, score } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { PerTopic, Runs } from "../types.ts";
import {
  CONFIG,
  hexToRgba,
  layoutTemplate,
  rankerColor,
} from "./theme.ts";

/** Half the height of the band a row's points are spread over. */
const JITTER = 0.17;

/**
 * A topic's offset inside its row: a hash of the topic number, so the cloud
 * is reproducible between renders and between themes rather than reshuffling
 * every time the chart is drawn.
 */
function jitter(num: number): number {
  let h = Math.imul(num ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return ((h >>> 0) / 0x100000000 - 0.5) * 2 * JITTER;
}

interface Row {
  ranker: string;
  label: string;
  ap: number[];
  mean: number;
  zeros: number;
}

export async function render(container: HTMLElement): Promise<void> {
  let per: PerTopic;
  let runs: Runs;
  try {
    [per, runs] = await Promise.all([
      loadJson<PerTopic>("per_topic.json"),
      loadJson<Runs>("runs.json"),
    ]);
  } catch (err) {
    showError(
      container,
      err instanceof Error
        ? err.message
        : "Could not load the per-topic scores.",
      () => void render(container),
    );
    return;
  }

  const topics = per.topics;
  const rows: Row[] = [];
  for (const r of runs.rankers) {
    const ap = per.ap[`${r.key}|title`];
    if (!ap || ap.length !== topics.length) continue;
    rows.push({
      ranker: r.key,
      label: r.label,
      ap,
      mean: ap.reduce((a, b) => a + b, 0) / ap.length,
      zeros: ap.filter((v) => v === 0).length,
    });
  }
  if (!rows.length || !topics.length) {
    showError(
      container,
      "The per-topic file carries no title-query scores.",
      () => void render(container),
    );
    return;
  }

  const n = topics.length;
  const worst = rows.reduce((a, b) => (b.zeros > a.zeros ? b : a));
  // Every ranker failing the same number of topics is the more interesting
  // fact, and naming one of them would read as if the others did better.
  const tied = rows.every((r) => r.zeros === worst.zeros);
  const title = tied
    ? `${pct0(worst.zeros / n)} of topics score zero under every ranker`
    : `${pct0(worst.zeros / n)} of topics score zero under ${worst.label}`;

  const deadEverywhere = topics.filter((_, i) =>
    rows.every((r) => r.ap[i] === 0),
  ).length;

  const spec: FigureSpec = {
    id: "fig-apstrip",
    title,
    subtitle: `Average precision on each of the ${int(n)} evaluable topics, title queries. Points are offset vertically by a hash of the topic number so they do not hide one another; the upright tick is the mean. ${int(deadEverywhere)} ${deadEverywhere === 1 ? "topic returns" : "topics return"} nothing relevant under any ranker here.`,
    note: `Source: FT 1991 slice of TREC disk 4, judged by the NIST Robust 2004 qrels; ${int(n)} of ${int(runs.n_topics_all)} topics have a relevant article inside the slice.`,
    alt: `Strip plot. One row per ranker, one point per topic, positioned by that topic's average precision on title queries, with the mean marked. ${title}.`,
    table: {
      columns: ["Topic", ...rows.map((r) => r.label)],
      rows: topics.map((num, i) => [
        num,
        ...rows.map((r) => score(r.ap[i] ?? 0)),
      ]),
    },
  };

  const plot = mountFigure(container, spec);

  function draw(): Promise<unknown> {
    const ink = cssVar("--ink");
    const traces: Partial<Plotly.PlotData>[] = [];
    rows.forEach((r, i) => {
      const color = rankerColor(r.ranker);
      traces.push({
        type: "scatter",
        mode: "markers",
        name: r.label,
        showlegend: false,
        x: r.ap,
        y: topics.map((num) => i + jitter(num)),
        customdata: topics,
        hovertemplate: `Topic %{customdata}<br><b>%{x:.3f}</b> average precision<extra>${r.label}</extra>`,
        marker: {
          size: 7,
          color: hexToRgba(color, 0.75),
          line: { color, width: 1 },
        },
      });
      traces.push({
        type: "scatter",
        mode: "markers",
        showlegend: false,
        x: [r.mean],
        y: [i],
        hovertemplate: `<b>%{x:.3f}</b> mean average precision<extra>${r.label}</extra>`,
        marker: {
          symbol: "line-ns-open",
          size: 26,
          color: ink,
          line: { color: ink, width: 2 },
        },
      });
    });

    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      hovermode: "closest",
      margin: { ...base.margin, t: 16, r: 24, b: 48 },
      xaxis: {
        ...base.xaxis,
        range: [-0.03, 1.03],
        zeroline: false,
        dtick: 0.2,
        title: { text: "Average precision", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        // Plotly stacks a numeric axis upwards, so the range is inverted to
        // put the first ranker in the file at the top of the chart.
        range: [rows.length - 0.5, -0.5],
        automargin: true,
        showgrid: false,
        zeroline: false,
        tickmode: "array",
        tickvals: rows.map((_, i) => i),
        ticktext: rows.map((r) => `${r.label}<br>mean ${score(r.mean)}`),
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
