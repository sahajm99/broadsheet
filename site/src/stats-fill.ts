import {
  fixed,
  int,
  pct,
  pct0,
  pctpt,
  ratio,
  score,
  signed,
} from "./fmt.ts";

type Formatter = (value: number) => string;

const FORMATTERS: Record<string, Formatter> = {
  int,
  pct,
  pct0,
  pctpt,
  ratio,
  score,
  signed,
  raw: (v) => String(v),
};

/**
 * `data-stat="key|2"` asks for two decimals. Digits are a format in their own
 * right because most numbers on this page are metric means and sizes, where
 * the only decision is how many decimals the sentence can carry.
 */
function formatterFor(name: string): Formatter | undefined {
  if (/^\d$/.test(name)) {
    const digits = Number(name);
    return (v) => fixed(v, digits);
  }
  return FORMATTERS[name];
}

/**
 * Fill every `<span data-stat="key|fmt">` from the `values` bag of
 * claims.json, so no number that describes the data is hand-typed in the
 * HTML (W7).
 */
export function fillStats(values: Record<string, number | string>): void {
  for (const node of document.querySelectorAll<HTMLElement>("[data-stat]")) {
    const spec = node.dataset.stat ?? "";
    const [key, fmtName = "raw"] = spec.split("|");
    const value = key === undefined ? undefined : values[key];
    if (value === undefined) {
      console.warn(`stats-fill: claims.json has no key "${key}"`);
      continue;
    }
    if (typeof value === "string") {
      node.textContent = value;
      continue;
    }
    const fmt = formatterFor(fmtName);
    if (!fmt) {
      console.warn(`stats-fill: unknown format "${fmtName}" for "${key}"`);
      continue;
    }
    node.textContent = fmt(value);
  }
}

