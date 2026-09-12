"""Porter stemmer tests.

The arbiter is Martin Porter's own vector file: `voc.txt` (input words) and
`output.txt` (their stems), 23,531 lines each, committed under
`tests/golden/porter/`.
"""

from pathlib import Path

import pytest

from pipeline.porter import stem

GOLDEN = Path(__file__).parent / "golden" / "porter"


def _read(name: str) -> list[str]:
    return GOLDEN.joinpath(name).read_text(encoding="utf-8").split("\n")


def test_golden_vectors_have_matching_lengths():
    voc, out = _read("voc.txt"), _read("output.txt")
    assert len(voc) == len(out)
    assert len([w for w in voc if w]) == 23531


def test_every_golden_word_stems_to_the_published_stem():
    voc, out = _read("voc.txt"), _read("output.txt")
    mismatches = [
        (word, expected, stem(word))
        for word, expected in zip(voc, out)
        if word and stem(word) != expected
    ]
    assert mismatches == []


@pytest.mark.parametrize(
    "word,expected",
    [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("relational", "relat"),
        ("sky", "sky"),
        ("agreed", "agre"),
        ("generalization", "gener"),
        ("oscillators", "oscil"),
        ("hopping", "hop"),
        ("feed", "feed"),
        ("at", "at"),
    ],
)
def test_documented_examples(word, expected):
    assert stem(word) == expected


@pytest.mark.parametrize("word", ["", "a", "is", "by"])
def test_short_words_are_returned_unchanged(word):
    assert stem(word) == word
