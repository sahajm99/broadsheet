import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { fixed, score } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { Bm25Grid } from "../types.ts";
import {
  CONFIG,
  MONO_FAMILY,
  inkOn,
  layoutTemplate,
  seqColorscale,
} from "./theme.ts";

/** `#rgb` or `#rrggbb` to three channels; black for anything unparseable. */
function parseHex(hex: string): [number, number, number] {
  const m = /^#?([\da-f]{3}|[\da-f]{6})$/i.exec(hex.trim());
  if (!m) return [0, 0, 0];
  const d = m[1]!;
  const full =
    d.length === 3
      ? d
          .split("")
          .map((c) => c + c)
          .join("")
      : d;
  const n = Number.parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function toHex(rgb: [number, number, number]): string {
  return `#${rgb.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
}

/**
 * The colour the ramp gives position `t`, so each cell's label can be picked
 * against the fill it actually lands on rather than against an average of the
 * whole scale.
 */
function sampleScale(stops: [number, string][], t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i += 1) {
    const [p0, c0] = stops[i - 1]!;
    const [p1, c1] = stops[i]!;
    if (clamped > p1 && i < stops.length - 1) continue;
    const span = p1 - p0;
    const f = span === 0 ? 0 : (clamped - p0) / span;
    const a = parseHex(c0);
    const b = parseHex(c1);
    return toHex([
      a[0] + (b[0] - a[0]) * f,
      a[1] + (b[1] - a[1]) * f,
      a[2] + (b[2] - a[2]) * f,
    ]);
  }
  return stops[stops.length - 1]?.[1] ?? "#000000";
}

export async function render(container: HTMLElement): Promise<void> {
  let grid: Bm25Grid;
  try {
    grid = await loadJson<Bm25Grid>("bm25_grid.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the BM25 grid.",
      () => void render(container),
    );
    return;
  }

  const k1s = grid.k1;
  const bs = grid.b;
  const flat = grid.map.flat();
  if (!k1s.length || !bs.length || !flat.length) {
    showError(container, "The BM25 grid file carries no cells.", () => {
      void render(container);
    });
    return;
  }

  const k1Text = k1s.map((v) => fixed(v, 1));
  const bText = bs.map((v) => fixed(v, 2));
  const zmin = Math.min(...flat);
  const zmax = Math.max(...flat);
  const isBest = (k1: number, b: number) =>
    k1 === grid.best.k1 && b === grid.best.b;
  const isDefault = (k1: number, b: number) =>
    k1 === grid.default.k1 && b === grid.default.b;

  const title = `Best grid cell ${score(grid.best.map)} at k1 = ${fixed(grid.best.k1, 1)}, b = ${fixed(grid.best.b, 2)}; default ${score(grid.default.map)}`;

  const spec: FigureSpec = {
    id: "fig-grid",
    title,
    subtitle: `The grid was ${grid.caveat}, so its best cell is a ceiling rather than a setting that would generalise. Each cell is MAP for BM25 on title queries; the outlined cell is the published default, k1 = ${fixed(grid.default.k1, 1)} and b = ${fixed(grid.default.b, 2)}.`,
    note: "Source: FT 1991 slice of TREC disk 4, judged by the NIST Robust 2004 qrels; the left-hand column switches document length normalisation off entirely, which is a sign the slice's articles are too uniform in length for it to pay.",
    alt: `Heat map. Mean average precision for BM25 over a grid of k1 and b values, with the published default outlined and the best cell marked. ${title}.`,
    table: {
      columns: ["k1", ...bText.map((b) => `b = ${b}`)],
      rows: k1s.map((k1, i) => [
        k1Text[i] ?? fixed(k1, 1),
        ...bs.map((b, j) => {
          const v = grid.map[i]?.[j];
          if (v === undefined) return "—";
          const mark = isBest(k1, b)
            ? " (best)"
            : isDefault(k1, b)
              ? " (default)"
              : "";
          return `${score(v)}${mark}`;
        }),
      ]),
    },
  };

  const plot = mountFigure(container, spec);

  function draw(): Promise<unknown> {
    const stops = seqColorscale();
    const span = zmax - zmin;
    const ink2 = cssVar("--ink-2");

    const trace: Partial<Plotly.PlotData> = {
      type: "heatmap",
      x: bText,
      y: k1Text,
      z: grid.map,
      colorscale: stops,
      zmin,
      zmax,
      xgap: 2,
      ygap: 2,
      hovertemplate:
        "k1 = %{y}, b = %{x}<br><b>%{z:.3f}</b> MAP<extra></extra>",
      colorbar: {
        tickformat: ".2f",
        outlinewidth: 0,
        thickness: 12,
        len: 0.9,
        tickfont: { color: ink2 },
        title: { text: "MAP", side: "top" },
      },
    };

    // Twenty-five cells on a phone leave about fifty pixels each, so the
    // value has to come down a couple of points to sit inside its cell.
    const narrow = plot.clientWidth < 560;
    const annotations: Partial<Plotly.Annotations>[] = [];
    k1s.forEach((k1, i) => {
      bs.forEach((b, j) => {
        const v = grid.map[i]?.[j];
        if (v === undefined) return;
        const fill = sampleScale(stops, span === 0 ? 1 : (v - zmin) / span);
        const best = isBest(k1, b);
        const def = isDefault(k1, b);
        const mark = best && def ? "best and default" : best ? "best" : def ? "default" : "";
        annotations.push({
          // Category coordinates, like the outline below: the tick labels are
          // numerals, and Plotly would read a numeral as a fraction of the
          // way to the next category rather than as the category itself.
          x: j,
          y: i,
          text: mark ? `${score(v)}<br>${mark}` : score(v),
          showarrow: false,
          align: "center",
          font: {
            color: inkOn(fill),
            family: MONO_FAMILY,
            size: narrow ? 10 : mark ? 12 : 13,
          },
        });
      });
    });

    // Shapes on a category axis take the category's index, so the default
    // cell's outline is placed by position rather than by value.
    const di = k1s.indexOf(grid.default.k1);
    const dj = bs.indexOf(grid.default.b);
    const shapes: Partial<Plotly.Shape>[] = [];
    if (di >= 0 && dj >= 0) {
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "y",
        x0: dj - 0.5,
        x1: dj + 0.5,
        y0: di - 0.5,
        y1: di + 0.5,
        line: { color: cssVar("--ink"), width: 2 },
        fillcolor: "rgba(0,0,0,0)",
        layer: "above",
      });
    }

    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      annotations,
      shapes,
      showlegend: false,
      margin: { ...base.margin, l: 8, t: 16, r: 8, b: 56 },
      xaxis: {
        ...base.xaxis,
        type: "category",
        showgrid: false,
        automargin: true,
        title: { text: "b, how hard document length is normalised", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        type: "category",
        showgrid: false,
        automargin: true,
        title: { text: "k1, term frequency saturation", standoff: 8 },
      },
    };
    return Plotly.react(plot, [trace], layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
