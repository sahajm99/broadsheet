/**
 * The Porter stemmer (Porter 1980), ported line for line from
 * `pipeline/porter.py`, which is itself a port of Martin Porter's reference
 * implementation at tartarus.org.
 *
 * The port keeps the reference's documented departures from the paper (marked
 * DEPARTURE below), because Porter's published test vectors
 * (`tests/golden/porter/voc.txt` and `output.txt`, 23,531 words) encode the
 * behaviour of that implementation and both languages are tested against them.
 * If these two implementations ever disagree, the browser ranks differently
 * from every number printed on the page (X2).
 *
 * The algorithm assumes a lowercase ASCII word; the tokenizer guarantees that.
 * Words of two letters or fewer come back unchanged.
 */

const VOWELS = "aeiou";

/**
 * Porter's buffer: `b[0..k]` is the word, `j` a general-purpose offset set by
 * `ends` to the index of the character before a matched suffix. `b` may stay
 * longer than `k + 1`; everything reads through `k`.
 */
class Stemmer {
  private b: string;
  private k: number;
  private j = 0;

  constructor(word: string) {
    this.b = word;
    this.k = word.length - 1;
  }

  // --- the primitives of the paper ---------------------------------------

  /**
   * True when `b[i]` is a consonant. `y` is a consonant at the start of a word
   * and after a vowel, a vowel otherwise.
   */
  private cons(i: number): boolean {
    const ch = this.b[i];
    if (VOWELS.includes(ch)) return false;
    if (ch === "y") return i === 0 || !this.cons(i - 1);
    return true;
  }

  /** The measure of `b[0..j]`: the number of VC sequences in it. */
  private m(): number {
    let n = 0;
    let i = 0;
    const j = this.j;
    for (;;) {
      if (i > j) return n;
      if (!this.cons(i)) break;
      i += 1;
    }
    i += 1;
    for (;;) {
      for (;;) {
        if (i > j) return n;
        if (this.cons(i)) break;
        i += 1;
      }
      i += 1;
      n += 1;
      for (;;) {
        if (i > j) return n;
        if (!this.cons(i)) break;
        i += 1;
      }
      i += 1;
    }
  }

  /** True when `b[0..j]` contains a vowel. */
  private vowelInStem(): boolean {
    for (let i = 0; i <= this.j; i += 1) if (!this.cons(i)) return true;
    return false;
  }

  /** True when `b[i]` and `b[i-1]` are the same consonant. */
  private doublec(i: number): boolean {
    if (i < 1 || this.b[i] !== this.b[i - 1]) return false;
    return this.cons(i);
  }

  /**
   * True when `b[i-2..i]` is consonant-vowel-consonant and the last consonant
   * is not w, x or y. Used to restore a final `e`.
   */
  private cvc(i: number): boolean {
    if (i < 2 || !this.cons(i) || this.cons(i - 1) || !this.cons(i - 2)) {
      return false;
    }
    return !"wxy".includes(this.b[i]);
  }

  // --- suffix helpers -----------------------------------------------------

  /** True when `b[0..k]` ends with `s`; sets `j` to the index before it. */
  private ends(s: string): boolean {
    const length = s.length;
    if (length > this.k + 1) return false;
    if (this.b.slice(this.k - length + 1, this.k + 1) !== s) return false;
    this.j = this.k - length;
    return true;
  }

  /** Replace everything after `j` with `s`. */
  private setto(s: string): void {
    this.b = this.b.slice(0, this.j + 1) + s;
    this.k = this.j + s.length;
  }

  /** `setto(s)` when the stem still has a measure greater than zero. */
  private r(s: string): void {
    if (this.m() > 0) this.setto(s);
  }

  // --- the steps ----------------------------------------------------------

  /**
   * Plurals and past participles: caresses -> caress, ponies -> poni,
   * agreed -> agree, plastered -> plaster, hopping -> hop.
   */
  private step1ab(): void {
    if (this.b[this.k] === "s") {
      if (this.ends("sses")) this.k -= 2;
      else if (this.ends("ies")) this.setto("i");
      else if (this.b[this.k - 1] !== "s") this.k -= 1;
    }
    if (this.ends("eed")) {
      if (this.m() > 0) this.k -= 1;
    } else if ((this.ends("ed") || this.ends("ing")) && this.vowelInStem()) {
      this.k = this.j;
      if (this.ends("at")) this.setto("ate");
      else if (this.ends("bl")) this.setto("ble");
      else if (this.ends("iz")) this.setto("ize");
      else if (this.doublec(this.k)) {
        this.k -= 1;
        if ("lsz".includes(this.b[this.k])) this.k += 1;
      } else if (this.m() === 1 && this.cvc(this.k)) this.setto("e");
    }
  }

  /**
   * Terminal y -> i when the stem holds a vowel: happy -> happi and
   * enjoy -> enjoi, while sky stays sky. (The reference's later revision tests
   * the preceding letter instead; the published vectors were made with this
   * rule, so this rule is what the port keeps.)
   */
  private step1c(): void {
    if (this.ends("y") && this.vowelInStem()) {
      this.b = `${this.b.slice(0, this.k)}i${this.b.slice(this.k + 1)}`;
    }
  }

  /**
   * Double suffixes to single ones: relational -> relate (then relat in
   * step 4), hesitanci -> hesitance, digitizer -> digitize.
   */
  private step2(): void {
    if (this.k < 1) return;
    const ch = this.b[this.k - 1];
    if (ch === "a") {
      if (this.ends("ational")) this.r("ate");
      else if (this.ends("tional")) this.r("tion");
    } else if (ch === "c") {
      if (this.ends("enci")) this.r("ence");
      else if (this.ends("anci")) this.r("ance");
    } else if (ch === "e") {
      if (this.ends("izer")) this.r("ize");
    } else if (ch === "l") {
      // DEPARTURE: the paper has abli -> able; the reference uses the more
      // general bli -> ble.
      if (this.ends("bli")) this.r("ble");
      else if (this.ends("alli")) this.r("al");
      else if (this.ends("entli")) this.r("ent");
      else if (this.ends("eli")) this.r("e");
      else if (this.ends("ousli")) this.r("ous");
    } else if (ch === "o") {
      if (this.ends("ization")) this.r("ize");
      else if (this.ends("ation")) this.r("ate");
      else if (this.ends("ator")) this.r("ate");
    } else if (ch === "s") {
      if (this.ends("alism")) this.r("al");
      else if (this.ends("iveness")) this.r("ive");
      else if (this.ends("fulness")) this.r("ful");
      else if (this.ends("ousness")) this.r("ous");
    } else if (ch === "t") {
      if (this.ends("aliti")) this.r("al");
      else if (this.ends("iviti")) this.r("ive");
      else if (this.ends("biliti")) this.r("ble");
    } else if (ch === "g") {
      // DEPARTURE: not in the paper; it stems archaeologi, biologi, ...
      if (this.ends("logi")) this.r("log");
    }
  }

  /**
   * -ic-, -full, -ness and friends: triplicate -> triplic,
   * hopefulness -> hope, goodness -> good.
   */
  private step3(): void {
    const ch = this.b[this.k];
    if (ch === "e") {
      if (this.ends("icate")) this.r("ic");
      else if (this.ends("ative")) this.r("");
      else if (this.ends("alize")) this.r("al");
    } else if (ch === "i") {
      if (this.ends("iciti")) this.r("ic");
    } else if (ch === "l") {
      if (this.ends("ical")) this.r("ic");
      else if (this.ends("ful")) this.r("");
    } else if (ch === "s") {
      if (this.ends("ness")) this.r("");
    }
  }

  /**
   * Strip -ant, -ence, -ion and the rest when the stem measures above one:
   * revival -> reviv, allowance -> allow, adoption -> adopt.
   */
  private step4(): void {
    if (this.k < 1) return;
    const ch = this.b[this.k - 1];
    if (ch === "a") {
      if (!this.ends("al")) return;
    } else if (ch === "c") {
      if (!(this.ends("ance") || this.ends("ence"))) return;
    } else if (ch === "e") {
      if (!this.ends("er")) return;
    } else if (ch === "i") {
      if (!this.ends("ic")) return;
    } else if (ch === "l") {
      if (!(this.ends("able") || this.ends("ible"))) return;
    } else if (ch === "n") {
      if (
        !(
          this.ends("ant") ||
          this.ends("ement") ||
          this.ends("ment") ||
          this.ends("ent")
        )
      ) {
        return;
      }
    } else if (ch === "o") {
      // DEPARTURE: -ion goes only when preceded by s or t.
      if (this.ends("ion") && this.j >= 0 && "st".includes(this.b[this.j])) {
        // fall through to the measure test
      } else if (this.ends("ou")) {
        // fall through to the measure test
      } else return;
    } else if (ch === "s") {
      if (!this.ends("ism")) return;
    } else if (ch === "t") {
      if (!(this.ends("ate") || this.ends("iti"))) return;
    } else if (ch === "u") {
      if (!this.ends("ous")) return;
    } else if (ch === "v") {
      if (!this.ends("ive")) return;
    } else if (ch === "z") {
      if (!this.ends("ize")) return;
    } else return;
    if (this.m() > 1) this.k = this.j;
  }

  /**
   * Tidy up: a final e goes unless the stem is short, and a double l
   * collapses. probate -> probat, controll -> control, roll -> roll.
   */
  private step5(): void {
    this.j = this.k;
    if (this.b[this.k] === "e") {
      const a = this.m();
      if (a > 1 || (a === 1 && !this.cvc(this.k - 1))) this.k -= 1;
    }
    if (this.b[this.k] === "l" && this.doublec(this.k) && this.m() > 1) {
      this.k -= 1;
    }
  }

  run(): string {
    this.step1ab();
    this.step1c();
    this.step2();
    this.step3();
    this.step4();
    this.step5();
    return this.b.slice(0, this.k + 1);
  }
}

/**
 * Memoised, as the Python side is: a query repeats words the reader has
 * already typed, and the parity test stems 23,531 words twice over.
 */
const cache = new Map<string, string>();
const CACHE_LIMIT = 1 << 17;

/**
 * The Porter stem of a lowercase ASCII word. Words of two letters or fewer
 * are returned unchanged, as in the reference.
 */
export function stem(word: string): string {
  if (word.length <= 2) return word;
  const hit = cache.get(word);
  if (hit !== undefined) return hit;
  const value = new Stemmer(word).run();
  if (cache.size < CACHE_LIMIT) cache.set(word, value);
  return value;
}
