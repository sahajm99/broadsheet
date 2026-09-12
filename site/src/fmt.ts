/** Number formatting. Every number shown on the page passes through here. */

const nf = new Intl.NumberFormat("en-US");

/** 319795 -> "319,795" */
export function int(n: number): string {
  return nf.format(Math.round(n));
}

/** 0.2431 -> "24.3%" (a proportion). */
export function pct(p: number, digits = 1): string {
  return `${(p * 100).toFixed(digits)}%`;
}

/** 0.2431 -> "24%" */
export function pct0(p: number): string {
  return pct(p, 0);
}

/** 43.7 -> "43.7%" (a number already in percentage points). */
export function pctpt(v: number, digits = 1): string {
  return `${v.toFixed(digits)}%`;
}

/** 2.015 -> "2.0x"; 31.41 -> "31x" (a second decimal on a big ratio is noise). */
export function ratio(r: number): string {
  return r >= 10 ? `${Math.round(r)}x` : `${r.toFixed(1)}x`;
}

/** Fixed decimals, with a minus sign that survives rounding to zero. */
export function fixed(v: number, digits = 2): string {
  const s = v.toFixed(digits);
  return s === `-${(0).toFixed(digits)}` ? (0).toFixed(digits) : s;
}

/** A retrieval score or a metric mean: three decimals, as TREC reports them. */
export function score(v: number): string {
  return fixed(v, 3);
}

/** A signed difference, so a gain of 0.02 reads as "+0.020". */
export function signed(v: number, digits = 3): string {
  const s = fixed(v, digits);
  return v > 0 ? `+${s}` : s;
}

/** (0.041, 0.093) -> "95% CI 0.041 to 0.093" */
export function ciText(lo: number, hi: number, digits = 3): string {
  return `95% CI ${fixed(lo, digits)} to ${fixed(hi, digits)}`;
}

/** "1991-05-14" -> "14 May 1991". Dates in this corpus are always ISO. */
export function isoDate(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return iso;
  const d = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
}
