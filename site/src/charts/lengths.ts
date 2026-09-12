import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { int, pct } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { Corpus, Postings } from "../types.ts";
import { CONFIG, layoutTemplate, series } from "./theme.ts";

const NOTE =
  "Source: the inverted index over the Financial Times 1991 collection. Article length is counted in stemmed, stopworded terms; bins beyond the longest observed length are empty and are not drawn.";

/** Below this plot width the two panels stack instead of sitting side by side. */
const WIDE = 560;

interface Bin {
  lo: number;
  /** null on the last bin, which is open at the top. */
  hi: number | null;
  count: number;
}

/**
 * Log-spaced edges plus one count per edge, as the contract writes them: bin
 * `i` is `[bins[i], bins[i+1])` and the last one catches everything above.
 * Empty bins at the top carry nothing, so they are dropped from the figure and
 * from the table with it.
 */
function toBins(hist: { bins: number[]; counts: number[] }): Bin[] {
  const all: Bin[] = hist.counts.map((count, i) => {
    const lo = hist.bins[i] ?? 0;
    const next = hist.bins[i + 1];
    return { lo, hi: next === undefined ? null : next - 1, count };
  });
  let last = all.length - 1;
  while (last > 0 && (all[last] as Bin).count === 0) last -= 1;
  return all.slice(0, last + 1);
}

/** "1", "3-4", "4,097+" — the same range, with or without separators. */
function binLabel(bin: Bin, separators: boolean): string {
  const n = (v: number): string => (separators ? int(v) : String(v));
  if (bin.hi === null) return `${n(bin.lo)}+`;
  return bin.hi === bin.lo ? n(bin.lo) : `${n(bin.lo)}-${n(bin.hi)}`;
}

/** Every other tick, so a phone-width panel never stacks its labels. */
function everyOther(labels: string[]): string[] {
  return labels.filter((_, i) => i % 2 === 0);
}

export async function render(container: HTMLElement): Promise<void> {
  let postings: Postings;
  let corpus: Corpus;
  try {
    [postings, corpus] = await Promise.all([
      loadJson<Postings>("postings.json"),
      loadJson<Corpus>("corpus.json"),
    ]);
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the index data.",
      () => void render(container),
    );
    return;
  }

  const dfBins = toBins(postings.hist);
  const lenBins = toBins(corpus.doc_len);
  if (!dfBins.length || !lenBins.length) {
    showError(container, "The index files carry no length bins.", () =>
      void render(container),
    );
    return;
  }
  const s = postings.summary;

  const spec: FigureSpec = {
    id: "fig-lengths",
    title: `${pct(s.df1_share)} of stems occur in a single article, and the median article is ${int(corpus.doc_len_summary.median)} terms long`,
    subtitle: `Posting-list length for ${int(s.n_terms)} stems, beside article length for ${int(corpus.n_docs)} articles. Both are counted into log-spaced bins, so each bar covers a wider range than the one before it.`,
    note: NOTE,
    alt: `Two bar histograms. Posting-list length is heavily skewed: ${pct(s.df1_share)} of the ${int(s.n_terms)} stems appear in exactly one article, while the longest list runs to ${int(s.max_len)} articles. Article length is far more even, with a median of ${int(corpus.doc_len_summary.median)} terms and a maximum of ${int(corpus.doc_len_summary.max)}.`,
    table: {
      columns: ["Measure", "Bin", "Count"],
      rows: [
        ...dfBins.map((b) => [
          "Posting-list length (articles)",
          binLabel(b, true),
          int(b.count),
        ]),
        ...lenBins.map((b) => [
          "Article length (terms)",
          binLabel(b, true),
          int(b.count),
        ]),
      ],
    },
  };

  const plot = mountFigure(container, spec);
  const dfLabels = dfBins.map((b) => binLabel(b, false));
  const lenLabels = lenBins.map((b) => binLabel(b, false));
  let wide = plot.clientWidth >= WIDE;

  function draw(): Promise<unknown> {
    const c = series();
    const ink2 = cssVar("--ink-2");

    const dfTrace: Partial<Plotly.PlotData> = {
      type: "bar",
      name: "Stems",
      x: dfLabels,
      y: dfBins.map((b) => b.count),
      marker: { color: c.s1 },
      hovertemplate: "%{x} articles<br><b>%{y:,} stems</b><extra></extra>",
      xaxis: "x",
      yaxis: "y",
    };
    const lenTrace: Partial<Plotly.PlotData> = {
      type: "bar",
      name: "Articles",
      x: lenLabels,
      y: lenBins.map((b) => b.count),
      marker: { color: c.s3 },
      hovertemplate: "%{x} terms<br><b>%{y:,} articles</b><extra></extra>",
      xaxis: "x2",
      yaxis: "y2",
    };

    const panelTitle = (
      text: string,
      x: number,
      y: number,
    ): Partial<Plotly.Annotations> => ({
      text,
      x,
      y,
      xref: "paper",
      yref: "paper",
      xanchor: "left",
      yanchor: "bottom",
      showarrow: false,
      font: { size: 13, color: cssVar("--ink") },
    });

    const base = layoutTemplate();
    // The panels are placed by hand rather than with `grid`, so each panel
    // title can be put exactly above its own panel in both layouts.
    const xDomain: [number, number] = wide ? [0, 0.44] : [0, 1];
    const xDomain2: [number, number] = wide ? [0.56, 1] : [0, 1];
    const yDomain: [number, number] = wide ? [0, 1] : [0.58, 1];
    const yDomain2: [number, number] = wide ? [0, 1] : [0, 0.36];

    const xAxis = {
      ...base.xaxis,
      type: "category" as const,
      showgrid: false,
      tickmode: "array" as const,
      tickangle: -45,
      tickfont: { color: ink2, size: wide ? 10 : 9 },
    };
    const yAxis = {
      ...base.yaxis,
      rangemode: "tozero" as const,
      separatethousands: true,
      exponentformat: "none" as const,
      automargin: true,
    };

    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      bargap: 0.18,
      margin: {
        ...base.margin,
        l: 8,
        r: 8,
        t: wide ? 30 : 26,
        b: wide ? 88 : 56,
      },
      annotations: [
        panelTitle("Stems per posting-list length, in articles", 0, 1.03),
        panelTitle(
          "Articles per article length, in terms",
          wide ? 0.56 : 0,
          wide ? 1.03 : yDomain2[1] + 0.04,
        ),
      ],
      xaxis: {
        ...xAxis,
        domain: xDomain,
        anchor: "y",
        tickvals: wide ? dfLabels : everyOther(dfLabels),
        ...(wide
          ? { title: { text: "Articles containing the stem", standoff: 6 } }
          : {}),
      },
      xaxis2: {
        ...xAxis,
        domain: xDomain2,
        anchor: "y2",
        tickvals: wide ? lenLabels : everyOther(lenLabels),
        ...(wide
          ? { title: { text: "Terms in the article", standoff: 6 } }
          : {}),
      },
      yaxis: {
        ...yAxis,
        domain: yDomain,
        anchor: "x",
        title: { text: "Stems", standoff: 6 },
      },
      yaxis2: {
        ...yAxis,
        domain: yDomain2,
        anchor: "x2",
        title: { text: "Articles", standoff: 6 },
      },
    };
    return Plotly.react(plot, [dfTrace, lenTrace], layout, CONFIG);
  }

  // Two panels side by side stop being readable on a phone, so the layout
  // flips to a stack and back as the figure is resized.
  new ResizeObserver(() => {
    const next = plot.clientWidth >= WIDE;
    if (next !== wide) {
      wide = next;
      void draw();
    }
  }).observe(plot);

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
