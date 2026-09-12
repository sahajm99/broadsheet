import { loadJson } from "../data.ts";
import {
  controlSlot,
  mountFigure,
  showError,
  updateFigure,
  type FigureSpec,
} from "../figure.ts";
import { int, signed } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { Comparison, Comparisons, Runs } from "../types.ts";
import { CONFIG, MONO_FAMILY, hexToRgba, layoutTemplate, series } from "./theme.ts";

/**
 * How each pair reads in a sentence and on a legend swatch. The file's own
 * `label` is the fallback, so a pair added upstream still draws and still
 * gets an honest, if flatter, claim.
 */
interface Phrase {
  subject: string;
  opponent: string;
  a: string;
  b: string;
}

const PHRASES: Record<string, Phrase | undefined> = {
  bm25_vs_raw: {
    subject: "BM25",
    opponent: "the course weighting",
    a: "BM25",
    b: "Course tf-idf",
  },
  log_vs_raw: {
    subject: "Log tf-idf",
    opponent: "the course weighting",
    a: "Log tf-idf",
    b: "Course tf-idf",
  },
  bm25_vs_log: {
    subject: "BM25",
    opponent: "log tf-idf",
    a: "BM25",
    b: "Log tf-idf",
  },
  desc_vs_title: {
    subject: "Adding the description",
    opponent: "the title alone",
    a: "Title + description",
    b: "Title only",
  },
  narr_vs_title: {
    subject: "Adding the description and narrative",
    opponent: "the title alone",
    a: "Title + desc + narrative",
    b: "Title only",
  },
  raw_narr_vs_title: {
    subject: "Adding the description and narrative to the course weighting",
    opponent: "the title alone",
    a: "Title + desc + narrative",
    b: "Title only",
  },
  stem_vs_nostem: {
    subject: "Stemming",
    opponent: "leaving the words unstemmed",
    a: "Stemmed",
    b: "Unstemmed",
  },
  stop_vs_nostop: {
    subject: "Removing stopwords",
    opponent: "keeping them",
    a: "Stopwords removed",
    b: "Stopwords kept",
  },
};

/** Half a thousandth: below this an interval bound prints as a flat 0.000. */
const NEAR_ZERO = 0.0005;

function excludesZero(lo: number, hi: number): boolean {
  return lo > 0 || hi < 0;
}

function picker(
  pairs: Comparison[],
  current: string,
  onChange: (key: string) => void,
): HTMLDivElement {
  const wrap = document.createElement("div");
  wrap.className = "fig-picker";

  const select = document.createElement("select");
  select.id = "fig-paired-pair";
  for (const p of pairs) {
    const opt = document.createElement("option");
    opt.value = p.key;
    opt.textContent = p.label;
    opt.selected = p.key === current;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => {
    onChange(select.value);
  });

  const label = document.createElement("label");
  label.htmlFor = select.id;
  label.textContent = "Comparison";

  wrap.append(label, select);
  return wrap;
}

export async function render(container: HTMLElement): Promise<void> {
  let comparisons: Comparisons;
  let runs: Runs;
  try {
    [comparisons, runs] = await Promise.all([
      loadJson<Comparisons>("comparisons.json"),
      loadJson<Runs>("runs.json"),
    ]);
  } catch (err) {
    showError(
      container,
      err instanceof Error
        ? err.message
        : "Could not load the paired comparisons.",
      () => void render(container),
    );
    return;
  }

  const first = comparisons.pairs[0];
  if (!first) {
    showError(container, "The comparisons file carries no pairs.", () => {
      void render(container);
    });
    return;
  }
  let pair: Comparison = first;

  /** Descending, so the topics a change helps are read first, left to right. */
  function sorted(p: Comparison): { num: number; diff: number }[] {
    return p.per_topic.slice().sort((x, y) => y.diff - x.diff);
  }

  function specFor(): FigureSpec {
    const ph = PHRASES[pair.key];
    const title = ph
      ? `${ph.subject} wins ${int(pair.wins)}, loses ${int(pair.losses)}, ties ${int(pair.ties)} against ${ph.opponent}`
      : `${pair.label}: ${int(pair.wins)} wins, ${int(pair.losses)} losses, ${int(pair.ties)} ties`;
    // An interval that clears zero by less than the third decimal prints as
    // "+0.000", which would read as a decision the data cannot support.
    const hairline =
      excludesZero(pair.lo, pair.hi) &&
      Math.min(Math.abs(pair.lo), Math.abs(pair.hi)) < NEAR_ZERO
        ? ` That interval clears zero by less than a thousandth, so it is compatible with no effect at all.`
        : "";
    return {
      id: "fig-paired",
      title,
      subtitle: `Per-topic difference in average precision, ${pair.label}. The line is the mean ${signed(pair.mean_diff)} and the band around it is the 95% bootstrap interval over ${int(runs.n_topics)} topics, ${signed(pair.lo)} to ${signed(pair.hi)}.${hairline}`,
      note: "Source: FT 1991 slice of TREC disk 4, judged by the NIST Robust 2004 qrels; a topic the two runs score identically on counts as a tie and draws no bar.",
      alt: `Bar chart. One bar per topic, the difference in average precision for ${pair.label}, sorted from the largest gain to the largest loss, with the mean and its 95% bootstrap interval drawn across. ${title}.`,
      table: {
        columns: ["Topic", "Difference in average precision"],
        rows: sorted(pair).map((d) => [d.num, signed(d.diff)]),
      },
    };
  }

  const plot = mountFigure(container, specFor());
  controlSlot(plot).before(
    picker(comparisons.pairs, pair.key, (key) => {
      const next = comparisons.pairs.find((p) => p.key === key);
      if (!next) return;
      pair = next;
      updateFigure(container, specFor());
      void draw();
    }),
  );

  function draw(): Promise<unknown> {
    const c = series();
    const ink = cssVar("--ink");
    const ink2 = cssVar("--ink-2");
    const ph = PHRASES[pair.key];
    const rows = sorted(pair);
    const idx = rows.map((_, i) => i);

    const side = (
      keep: (d: number) => boolean,
      color: string,
      name: string,
    ): Partial<Plotly.PlotData> => {
      const kept = idx.filter((i) => keep(rows[i]!.diff));
      return {
        type: "bar",
        name,
        x: kept,
        y: kept.map((i) => rows[i]!.diff),
        customdata: kept.map((i) => rows[i]!.num),
        hovertemplate:
          "Topic %{customdata}<br><b>%{y:+.3f}</b> average precision<extra></extra>",
        marker: { color },
      };
    };

    const traces = [
      side((d) => d > 0, c.accent, `${ph?.a ?? "The first run"} wins the topic`),
      side((d) => d < 0, c.s2, `${ph?.b ?? "The second run"} wins the topic`),
    ];

    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      barmode: "overlay",
      bargap: 0.2,
      showlegend: true,
      legend: {
        orientation: "h",
        x: 0,
        y: 1.02,
        yanchor: "bottom",
        font: { color: ink2 },
      },
      margin: { ...base.margin, l: 8, t: 34, r: 24, b: 48 },
      hovermode: "closest",
      xaxis: {
        ...base.xaxis,
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        range: [-1, rows.length],
        title: {
          text: `${int(rows.length)} topics, sorted by difference`,
          standoff: 8,
        },
      },
      yaxis: {
        ...base.yaxis,
        automargin: true,
        zeroline: true,
        zerolinecolor: ink2,
        zerolinewidth: 1,
        title: { text: "Difference in average precision", standoff: 8 },
      },
      shapes: [
        {
          type: "rect",
          xref: "paper",
          yref: "y",
          x0: 0,
          x1: 1,
          y0: pair.lo,
          y1: pair.hi,
          fillcolor: hexToRgba(c.muted, 0.18),
          line: { width: 0 },
          layer: "below",
        },
        {
          type: "line",
          xref: "paper",
          yref: "y",
          x0: 0,
          x1: 1,
          y0: pair.mean_diff,
          y1: pair.mean_diff,
          line: { color: ink, width: 1.5 },
          layer: "above",
        },
      ],
      annotations: [
        {
          xref: "paper",
          yref: "y",
          x: 1,
          y: pair.mean_diff,
          xanchor: "right",
          yanchor: pair.mean_diff >= 0 ? "bottom" : "top",
          showarrow: false,
          text: `mean ${signed(pair.mean_diff)}`,
          font: { color: ink, family: MONO_FAMILY, size: 12 },
        },
      ],
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
