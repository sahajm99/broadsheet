"""The Porter stemmer (Porter 1980), ported from Martin Porter's own
reference implementation at tartarus.org.

This is a line-for-line port of the reference `stem.c` / `porter.py`, keeping
the documented departures from the 1980 paper that the reference makes (they
are marked DEPARTURE below), because Porter's published test vectors
(`tests/golden/porter/voc.txt` and `output.txt`, 23,531 words) encode the
behaviour of that implementation, not of the paper.

The algorithm assumes a lowercase ASCII word; the tokenizer guarantees that.
Words of two letters or fewer come back unchanged.
"""

from __future__ import annotations

from functools import lru_cache

_VOWELS = frozenset("aeiou")


class _Stemmer:
    """Porter's buffer: `b[0..k]` is the word, `j` a general-purpose offset
    set by `_ends` to the index of the character before a matched suffix."""

    def __init__(self, word: str) -> None:
        self.b = word
        self.k = len(word) - 1
        self.j = 0

    # --- the primitives of the paper -------------------------------------

    def _cons(self, i: int) -> bool:
        """True when `b[i]` is a consonant. `y` is a consonant at the start of
        a word and after a vowel, a vowel otherwise."""
        ch = self.b[i]
        if ch in _VOWELS:
            return False
        if ch == "y":
            return i == 0 or not self._cons(i - 1)
        return True

    def _m(self) -> int:
        """The measure of `b[0..j]`: the number of VC sequences in it."""
        n = 0
        i = 0
        j = self.j
        while True:
            if i > j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowel_in_stem(self) -> bool:
        """True when `b[0..j]` contains a vowel."""
        return any(not self._cons(i) for i in range(self.j + 1))

    def _doublec(self, i: int) -> bool:
        """True when `b[i]` and `b[i-1]` are the same consonant."""
        if i < 1 or self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i: int) -> bool:
        """True when `b[i-2..i]` is consonant-vowel-consonant and the last
        consonant is not w, x or y. Used to restore a final `e`."""
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    # --- suffix helpers ---------------------------------------------------

    def _ends(self, s: str) -> bool:
        """True when `b[0..k]` ends with `s`; sets `j` to the index before it."""
        length = len(s)
        if length > self.k + 1:
            return False
        if self.b[self.k - length + 1 : self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s: str) -> None:
        """Replace everything after `j` with `s`."""
        self.b = self.b[: self.j + 1] + s
        self.k = self.j + len(s)

    def _r(self, s: str) -> None:
        """`_setto(s)` when the stem still has a measure greater than zero."""
        if self._m() > 0:
            self._setto(s)

    # --- the steps --------------------------------------------------------

    def _step1ab(self) -> None:
        """Plurals and past participles: caresses -> caress, ponies -> poni,
        agreed -> agree, plastered -> plaster, hopping -> hop."""
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowel_in_stem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                if self.b[self.k] in "lsz":
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self) -> None:
        """Terminal y -> i when the stem holds a vowel: happy -> happi and
        enjoy -> enjoi, while sky stays sky. (The reference's later revision
        tests the preceding letter instead; the published vectors were made
        with this rule, so this rule is what the port keeps.)"""
        if self._ends("y") and self._vowel_in_stem():
            self.b = self.b[: self.k] + "i" + self.b[self.k + 1 :]

    def _step2(self) -> None:
        """Double suffixes to single ones: relational -> relate (then relat in
        step 4), hesitanci -> hesitance, digitizer -> digitize."""
        if self.k < 1:
            return
        ch = self.b[self.k - 1]
        if ch == "a":
            if self._ends("ational"):
                self._r("ate")
            elif self._ends("tional"):
                self._r("tion")
        elif ch == "c":
            if self._ends("enci"):
                self._r("ence")
            elif self._ends("anci"):
                self._r("ance")
        elif ch == "e":
            if self._ends("izer"):
                self._r("ize")
        elif ch == "l":
            # DEPARTURE: the paper has abli -> able; the reference uses the
            # more general bli -> ble.
            if self._ends("bli"):
                self._r("ble")
            elif self._ends("alli"):
                self._r("al")
            elif self._ends("entli"):
                self._r("ent")
            elif self._ends("eli"):
                self._r("e")
            elif self._ends("ousli"):
                self._r("ous")
        elif ch == "o":
            if self._ends("ization"):
                self._r("ize")
            elif self._ends("ation"):
                self._r("ate")
            elif self._ends("ator"):
                self._r("ate")
        elif ch == "s":
            if self._ends("alism"):
                self._r("al")
            elif self._ends("iveness"):
                self._r("ive")
            elif self._ends("fulness"):
                self._r("ful")
            elif self._ends("ousness"):
                self._r("ous")
        elif ch == "t":
            if self._ends("aliti"):
                self._r("al")
            elif self._ends("iviti"):
                self._r("ive")
            elif self._ends("biliti"):
                self._r("ble")
        elif ch == "g":
            # DEPARTURE: not in the paper; it stems archaeologi, biologi, ...
            if self._ends("logi"):
                self._r("log")

    def _step3(self) -> None:
        """-ic-, -full, -ness and friends: triplicate -> triplic,
        hopefulness -> hope, goodness -> good."""
        ch = self.b[self.k]
        if ch == "e":
            if self._ends("icate"):
                self._r("ic")
            elif self._ends("ative"):
                self._r("")
            elif self._ends("alize"):
                self._r("al")
        elif ch == "i":
            if self._ends("iciti"):
                self._r("ic")
        elif ch == "l":
            if self._ends("ical"):
                self._r("ic")
            elif self._ends("ful"):
                self._r("")
        elif ch == "s":
            if self._ends("ness"):
                self._r("")

    def _step4(self) -> None:
        """Strip -ant, -ence, -ion and the rest when the stem measures above
        one: revival -> reviv, allowance -> allow, adoption -> adopt."""
        if self.k < 1:
            return
        ch = self.b[self.k - 1]
        if ch == "a":
            if not self._ends("al"):
                return
        elif ch == "c":
            if not (self._ends("ance") or self._ends("ence")):
                return
        elif ch == "e":
            if not self._ends("er"):
                return
        elif ch == "i":
            if not self._ends("ic"):
                return
        elif ch == "l":
            if not (self._ends("able") or self._ends("ible")):
                return
        elif ch == "n":
            if not (
                self._ends("ant")
                or self._ends("ement")
                or self._ends("ment")
                or self._ends("ent")
            ):
                return
        elif ch == "o":
            # DEPARTURE: -ion goes only when preceded by s or t.
            if self._ends("ion") and self.j >= 0 and self.b[self.j] in "st":
                pass
            elif self._ends("ou"):
                pass
            else:
                return
        elif ch == "s":
            if not self._ends("ism"):
                return
        elif ch == "t":
            if not (self._ends("ate") or self._ends("iti")):
                return
        elif ch == "u":
            if not self._ends("ous"):
                return
        elif ch == "v":
            if not self._ends("ive"):
                return
        elif ch == "z":
            if not self._ends("ize"):
                return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self) -> None:
        """Tidy up: a final e goes unless the stem is short, and a double l
        collapses. probate -> probat, controll -> control, roll -> roll."""
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def run(self) -> str:
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[: self.k + 1]


@lru_cache(maxsize=1 << 17)
def stem(word: str) -> str:
    """The Porter stem of a lowercase ASCII word.

    Words of two letters or fewer are returned unchanged, as in the reference.
    Results are memoised: a corpus pass stems the same few thousand types
    hundreds of thousands of times.
    """
    if len(word) <= 2:
        return word
    return _Stemmer(word).run()
