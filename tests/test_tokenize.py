"""Tokenizer and analyzer tests (course rules: lowercase, split on
non-alphanumerics, drop any token containing a digit)."""

from pipeline.tokenize import analyze, load_stopwords, tokenize


def test_tokenize_splits_lowercases_and_drops_tokens_with_digits():
    assert tokenize("The jet's engine roared 3 times, B2 bombers!") == [
        "the",
        "jet",
        "s",
        "engine",
        "roared",
        "times",
        "bombers",
    ]


def test_tokenize_empty_text():
    assert tokenize("") == []
    assert tokenize("1991 747 -- 3.5%") == []


def test_analyze_stops_then_stems():
    assert analyze(
        "The jet's engine roared 3 times, B2 bombers!",
        stopwords=frozenset({"the", "s"}),
        stem=True,
    ) == ["jet", "engin", "roar", "time", "bomber"]


def test_analyze_without_stopwords_or_stemming_is_tokenize():
    text = "The jet's engine roared 3 times, B2 bombers!"
    assert analyze(text, stopwords=None, stem=False) == tokenize(text)


def test_analyze_stop_only_and_stem_only():
    text = "The jet's engine roared 3 times, B2 bombers!"
    assert analyze(text, stopwords=frozenset({"the", "s"}), stem=False) == [
        "jet",
        "engine",
        "roared",
        "times",
        "bombers",
    ]
    assert analyze(text, stopwords=None, stem=True) == [
        "the",
        "jet",
        "s",
        "engin",
        "roar",
        "time",
        "bomber",
    ]


def test_load_stopwords_is_the_course_list():
    stopwords = load_stopwords()
    assert len(stopwords) == 523
    assert "the" in stopwords
    assert "able" in stopwords
    assert "" not in stopwords
    assert all(w == w.strip().lower() for w in stopwords)
