/**
 * The course tokenizer, matching `pipeline/tokenize.py`: lowercase, split on
 * non-alphanumerics, discard any token containing a digit; then optionally
 * stopword-filter and stem (X1).
 */

import stopwordList from "./stopwords.json";
import { stem } from "./porter.ts";

/**
 * The 523-word course list, written by the pipeline into `stopwords.json` so
 * the browser filters with exactly the list the index was built with (D6).
 */
export const STOPWORDS: ReadonlySet<string> = new Set<string>(
  stopwordList as string[],
);

const TOKEN = /[a-z0-9]+/g;
const HAS_DIGIT = /[0-9]/;

/**
 * Alphanumeric runs of the lowercased text, minus anything with a digit.
 *
 * Dropping digit-bearing tokens is the course rule, so `"3"`, `"1991"` and
 * `"B2"` all disappear while `"jet"` and `"s"` (from `"jet's"`) survive.
 */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const match of text.toLowerCase().matchAll(TOKEN)) {
    const token = match[0];
    if (!HAS_DIGIT.test(token)) out.push(token);
  }
  return out;
}

/**
 * Tokenize, drop stopwords when asked, stem when asked.
 *
 * Stopwords are removed **before** stemming, which is what the pipeline does:
 * the list is a list of surface words, not of stems. Reversing the two steps
 * changes the top ten for a good fraction of the topics.
 */
export function analyze(
  text: string,
  useStop: boolean,
  useStem: boolean,
): string[] {
  let tokens = tokenize(text);
  if (useStop) tokens = tokens.filter((token) => !STOPWORDS.has(token));
  if (useStem) tokens = tokens.map(stem);
  return tokens;
}
