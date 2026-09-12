import { loadJson } from "../data.ts";
import {
  controlSlot,
  mountFigure,
  showError,
  updateFigure,
  type FigureSpec,
} from "../figure.ts";
import { ciText, int, score, signed } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { Comparisons, MetricKey, Run, Runs } from "../types.ts";
import {
  CONFIG,
  MONO_FAMILY,
  errorBars,
  layoutTemplate,
  rankerColor,
  reversed,
} from "./theme.ts";

/**
 * The six metrics `runs.json` carries, in the order the picker offers them.
 * `gloss` is what the subtitle says about the metric, so a reader who changes
 * the picker is told what the new number means rather than left to infer it
 * from an abbreviation.
 */
const METRICS: { key: MetricKey; label: string; gloss: string }[] = [
  {
    key: "map",
    label: "MAP",
    gloss: "Mean average precision over the whole ranking",
  },
  { key: "p10", label: "P@10", gloss: "Precision in the top ten results" },
  {
    key: "ndcg10",
    label: "nDCG@10",
    gloss: "Discounted gain in the top ten, relevance treated as yes or no",
  },
  {
    key: "rprec",
    label: "R-precision",
    gloss:
      "Precision at R, where R is how many relevant articles the topic has",
  },
  {
    key: "recall100",
    label: "Recall@100",
    gloss: "Share of the topic's relevant articles found in the top hundred",
  },
  {
    key: "judged10",
    label: "judged@10",
    gloss:
      "Share of the top ten an assessor actually looked at, which is a coverage figure and not a quality one",
  },
];

const JUDGED: MetricKey = "judged10";

/** An interval that does not straddle zero; the file's bounds are ordered. */
function excludesZero(lo: number, hi: number): boolean {
  return lo > 0 || hi < 0;
}

function picker(
  current: MetricKey,
  onChange: (key: MetricKey) => void,
): HTMLDivElement {
  const wrap = document.createElement("div");
  wrap.className = "fig-picker";

  const select = document.createElement("select");
  select.id = "fig-rankers-metric";
  for (const m of METRICS) {
    const opt = document.createElement("option");
    opt.value = m.key;
    opt.textContent = m.label;
    opt.selected = m.key === current;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => {
    onChange(select.value as MetricKey);
  });

  const label = document.createElement("label");
  label.htmlFor = select.id;
  label.textContent = "Metric";

  wrap.append(label, select);
  return wrap;
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

  // Ranker order comes from the file, so the rows read in the order the rest
  // of the page names the rankers in.
  const rows: { ranker: string; label: string; run: Run }[] = [];
  for (const r of runs.rankers) {
    const run = runs.runs.find((x) => x.ranker === r.key && x.field === "title");
    if (run) rows.push({ ranker: r.key, label: r.label, run });
  }
  const bm25 = rows.find((r) => r.ranker === "bm25");
  const raw = rows.find((r) => r.ranker === "tfidf_raw");
  if (!bm25 || !raw) {
    showError(
      container,
      "The run file carries no title-query runs for these rankers.",
      () => void render(container),
    );
    return;
  }

  const vsRaw = comparisons.pairs.find((p) => p.key === "bm25_vs_raw");
  let metric: MetricKey = "map";

  function claim(m: (typeof METRICS)[number]): string {
    const a = bm25!.run.metrics[m.key];
    const b = raw!.run.metrics[m.key];
    if (m.key !== "map") {
      return `On ${m.label}, BM25 scores ${score(a.mean)} and the course weighting ${score(b.mean)}`;
    }
    // What settles MAP is the paired difference, not the distance between two
    // heavily overlapping per-run intervals, so the branch reads that.
    const separated = vsRaw ? excludesZero(vsRaw.lo, vsRaw.hi) : false;
    return separated
      ? `BM25 beats the course weighting on MAP: ${score(a.mean)} vs ${score(b.mean)}`
      : `BM25 and the course weighting are within the interval on MAP: ${score(a.mean)} vs ${score(b.mean)}`;
  }

  function specFor(): FigureSpec {
    const m = METRICS.find((x) => x.key === metric) ?? METRICS[0]!;
    const title = claim(m);
    const interval = `a 95% bootstrap interval over ${int(runs.n_topics)} topics`;
    let tail = "";
    if (m.key === "map" && vsRaw) {
      tail = ` The paired difference between them is ${signed(vsRaw.mean_diff)}, ${interval} of ${signed(vsRaw.lo)} to ${signed(vsRaw.hi)}.`;
    } else if (m.key !== "map") {
      tail = " The paired test this page argues from is on average precision only.";
    }
    const subtitle = `${m.gloss}, title queries, each with ${interval}.${tail}`;
    const showJudged = m.key !== JUDGED;
    const columns = ["Ranker", m.label, "95% interval"];
    if (showJudged) columns.push("judged@10");
    return {
      id: "fig-rankers",
      title,
      subtitle,
      note: `Source: FT 1991 slice of TREC disk 4, judged by the NIST Robust 2004 qrels; ${int(runs.n_topics)} of ${int(runs.n_topics_all)} topics have a relevant article inside the slice.`,
      alt: `Dot plot. ${m.label} for ${int(rows.length)} rankers on title-only queries, each with a 95% bootstrap interval. ${title}.`,
      table: {
        columns,
        rows: rows.map((r) => {
          const v = r.run.metrics[m.key];
          const cells: (string | number)[] = [
            r.label,
            score(v.mean),
            ciText(v.lo, v.hi),
          ];
          if (showJudged) cells.push(score(r.run.metrics[JUDGED].mean));
          return cells;
        }),
      },
    };
  }

  const plot = mountFigure(container, specFor());
  controlSlot(plot).before(
    picker(metric, (key) => {
      metric = key;
      updateFigure(container, specFor());
      void draw();
    }),
  );

  function draw(): Promise<unknown> {
    const m = METRICS.find((x) => x.key === metric) ?? METRICS[0]!;
    const ink = cssVar("--ink");
    // The ranker names are long enough to eat a phone's plot area, so they
    // come down a size rather than pushing the markers off the right edge.
    const narrow = plot.clientWidth < 560;
    const traces: Partial<Plotly.PlotData>[] = rows.map((r) => {
      const v = r.run.metrics[m.key];
      const color = rankerColor(r.ranker);
      return {
        type: "scatter",
        mode: "text+markers",
        name: r.label,
        showlegend: false,
        x: [v.mean],
        y: [r.label],
        text: [score(v.mean)],
        textposition: "top center",
        textfont: { color: ink, family: MONO_FAMILY, size: 12 },
        customdata: [[v.lo, v.hi]],
        hovertemplate: `<b>%{x:.3f}</b> ${m.label}<br>95% interval %{customdata[0]:.3f} to %{customdata[1]:.3f}<extra>%{y}</extra>`,
        marker: { size: 9, color },
        error_x: { ...errorBars([v.mean], [v.lo], [v.hi], color), width: 4 },
      };
    });

    // One range for every metric would spend most of the axis on P@10, so the
    // range follows the metric. It always starts at zero, so how far a marker
    // sits from the left edge keeps meaning "how much of the metric".
    const hi = Math.max(...rows.map((r) => r.run.metrics[m.key].hi));
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      hovermode: "closest",
      margin: { ...base.margin, t: 20, r: 24, b: 48 },
      xaxis: {
        ...base.xaxis,
        range: [0, hi * 1.12],
        zeroline: false,
        title: { text: m.label, standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        type: "category",
        automargin: true,
        showgrid: false,
        categoryorder: "array",
        categoryarray: reversed(rows.map((r) => r.label)),
        tickfont: { color: cssVar("--ink-2"), size: narrow ? 11 : 13 },
      },
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
