import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { int } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type { Vocab } from "../types.ts";
import {
  CONFIG,
  MONO_FAMILY,
  layoutTemplate,
  reversed,
  series,
} from "./theme.ts";

const NOTE =
  "Source: the Financial Times 1991 collection, tokenised, stopworded with the course list and Porter stemmed. Collection frequency counts every occurrence; document frequency counts the articles a stem appears in.";

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

  const terms = vocab.top_terms;
  const top = terms[0];
  const second = terms[1];
  if (!top) {
    showError(container, "The vocabulary file carries no top terms.", () =>
      void render(container),
    );
    return;
  }

  const spec: FigureSpec = {
    id: "fig-topterms",
    title: `\u201c${top.term}\u201d is the most frequent stem after stopwording`,
    subtitle: `Collection frequency of the ${int(terms.length)} heaviest stems out of ${int(vocab.treatments.stem_stop.n_types)}. The course stopword list keeps honorifics, so \u201c${top.term}\u201d survives it; document frequency is in the hover and the table.`,
    note: NOTE,
    alt: `Horizontal bar chart of the ${int(terms.length)} most frequent stems. \u201c${top.term}\u201d leads with ${int(top.cf)} occurrences across ${int(top.df)} articles${second ? `, ahead of \u201c${second.term}\u201d with ${int(second.cf)}` : ""}. The rest are the vocabulary of a financial daily: money, companies, markets and government.`,
    table: {
      columns: ["Stem", "Occurrences", "Articles"],
      rows: terms.map((t) => [t.term, int(t.cf), int(t.df)]),
    },
  };

  const plot = mountFigure(container, spec);
  const maxCf = Math.max(...terms.map((t) => t.cf));

  function draw(): Promise<unknown> {
    const c = series();
    const ink2 = cssVar("--ink-2");
    const rows = reversed(terms);
    const trace: Partial<Plotly.PlotData> = {
      type: "bar",
      orientation: "h",
      y: rows.map((t) => t.term),
      x: rows.map((t) => t.cf),
      customdata: rows.map((t) => t.df),
      text: rows.map((t) => int(t.cf)),
      textposition: "outside",
      textfont: { color: ink2, size: 11 },
      cliponaxis: false,
      hovertemplate:
        "%{y}<br><b>%{x:,} occurrences</b><br>in %{customdata:,} articles<extra></extra>",
      marker: { color: c.s1 },
      width: 0.68,
    };

    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      bargap: 0.25,
      margin: { ...base.margin, r: 64, b: 48 },
      xaxis: {
        ...base.xaxis,
        rangemode: "tozero",
        range: [0, maxCf * 1.1],
        separatethousands: true,
        exponentformat: "none",
        title: { text: "Occurrences in the collection", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        automargin: true,
        showgrid: false,
        tickfont: { color: ink2, family: MONO_FAMILY, size: 12 },
      },
    };
    return Plotly.react(plot, [trace], layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
