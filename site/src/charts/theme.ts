import { cssVar } from "../theme.ts";

/**
 * Plotly chrome derived from the CSS tokens, so light and dark never drift
 * apart: hairline rules one step off the surface, no modebar, axis text in
 * the secondary ink, and the page's own sans everywhere.
 */
export function layoutTemplate(): Partial<Plotly.Layout> {
  const surface = cssVar("--surface");
  const grid = cssVar("--rule");
  const ink = cssVar("--ink");
  const ink2 = cssVar("--ink-2");
  const axis = {
    gridcolor: grid,
    zerolinecolor: grid,
    linecolor: grid,
    tickfont: { color: ink2 },
  };
  return {
    paper_bgcolor: surface,
    plot_bgcolor: surface,
    font: {
      family: '"IBM Plex Sans Variable", system-ui, sans-serif',
      color: ink2,
      size: 13,
    },
    xaxis: { ...axis },
    yaxis: { ...axis },
    margin: { l: 8, r: 16, t: 8, b: 40 },
    hoverlabel: {
      bgcolor: cssVar("--surface-2"),
      bordercolor: grid,
      font: {
        color: ink,
        family: '"IBM Plex Sans Variable", system-ui, sans-serif',
        size: 13,
      },
    },
  };
}

/**
 * Mono for the machine's view of text: stems, document numbers and scores
 * inside a chart are set in the same face the page sets them in (W4).
 */
export const MONO_FAMILY = '"IBM Plex Mono", ui-monospace, monospace';

/** Series colours, read fresh on every render so the toggle recolours charts. */
export const series = () => ({
  s1: cssVar("--s1"),
  s2: cssVar("--s2"),
  s3: cssVar("--s3"),
  s4: cssVar("--s4"),
  accent: cssVar("--accent"),
  muted: cssVar("--muted"),
  grid: cssVar("--rule"),
});

/**
 * The ranker each series colour belongs to, fixed here so a ranker keeps its
 * colour in every chart on the page.
 */
export function rankerColor(key: string): string {
  const s = series();
  if (key.startsWith("tfidf_raw")) return s.s1;
  if (key.startsWith("tfidf_log")) return s.s2;
  if (key.startsWith("bm25")) return s.s3;
  return s.s4;
}

/**
 * Plotly stacks the first y category at the bottom, so a list drawn down the
 * page is reversed before it is handed over: the first item in the file ends
 * up at the top, in the same order as the sentence above the chart.
 */
export function reversed<T>(xs: T[]): T[] {
  return xs.slice().reverse();
}

/**
 * The legend every chart uses: one row above the plot, left aligned with the
 * claim, in the secondary ink. `y` moves it clear of a taller plot area.
 */
export function horizontalLegend(y = 1.04): Partial<Plotly.Legend> {
  return {
    orientation: "h",
    x: 0,
    y,
    yanchor: "bottom",
    font: { color: cssVar("--ink-2") },
  };
}

export const CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
};

/**
 * A CSS hex colour as `rgba(...)`, for fills that must sit behind a line
 * without being read as a second series. Falls back to the input if the token
 * is not a hex, so a theme change can never leave a chart colourless.
 */
export function hexToRgba(hex: string, alpha: number): string {
  const m = /^#?([\da-f]{3}|[\da-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const digits = m[1]!;
  const full =
    digits.length === 3
      ? digits
          .split("")
          .map((c) => c + c)
          .join("")
      : digits;
  const n = Number.parseInt(full, 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * The five sequential stops as a Plotly colorscale, low to high. The dark
 * tokens are the light ones flipped in lightness, so `--seq-5` is always the
 * end that contrasts most with the page and "more" always reads as "louder".
 */
export function seqColorscale(): [number, string][] {
  return [0, 0.25, 0.5, 0.75, 1].map((pos, i) => [
    pos,
    cssVar(`--seq-${i + 1}`),
  ]);
}

/** WCAG relative luminance of a hex colour; 0 for anything unparseable. */
function luminance(hex: string): number {
  const m = /^#?([\da-f]{3}|[\da-f]{6})$/i.exec(hex.trim());
  if (!m) return 0;
  const d = m[1]!;
  const full =
    d.length === 3
      ? d
          .split("")
          .map((c) => c + c)
          .join("")
      : d;
  const n = Number.parseInt(full, 16);
  const channel = (v: number): number => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return (
    0.2126 * channel((n >> 16) & 255) +
    0.7152 * channel((n >> 8) & 255) +
    0.0722 * channel(n & 255)
  );
}

/** The WCAG contrast ratio between two hex colours, 1 (same) to 21 (extreme). */
export function contrastRatio(a: string, b: string): number {
  const la = luminance(a);
  const lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/**
 * `--ink` or `--surface`, whichever reads better on `bg`. Label colours inside
 * a filled shape are picked this way rather than hard-coded, because the two
 * themes put the same token at opposite ends of the lightness range.
 */
export function inkOn(bg: string): string {
  const ink = cssVar("--ink");
  const surface = cssVar("--surface");
  return contrastRatio(bg, ink) >= contrastRatio(bg, surface) ? ink : surface;
}

/**
 * An asymmetric interval as Plotly error bars. Shared by every chart that
 * draws a 95% interval, so the hairline weight is decided in one place.
 */
export function errorBars(
  mid: number[],
  lo: number[],
  hi: number[],
  color: string,
): Plotly.ErrorBar {
  return {
    type: "data",
    symmetric: false,
    array: mid.map((p, i) => (hi[i] ?? p) - p),
    arrayminus: mid.map((p, i) => p - (lo[i] ?? p)),
    color,
    thickness: 1,
    width: 3,
  };
}
