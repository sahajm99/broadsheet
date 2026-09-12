"""Index builder tests against the hand-written mini corpus.

Every expected number here is counted by hand from `tests/fixtures/mini`, so a
change in the tokenizer, the stemmer or the posting layout fails loudly.
"""

from pathlib import Path

import pytest

from pipeline.constants import TREATMENTS
from pipeline.index import Index, build_all, build_index
from pipeline.parse import load_corpus

FIXTURE = Path(__file__).parent / "fixtures" / "mini"
STOPWORDS = frozenset({"the", "and", "a", "to", "s"})


@pytest.fixture(scope="module")
def docs():
    return load_corpus(FIXTURE)


@pytest.fixture(scope="module")
def index(docs):
    return build_index(docs, STOPWORDS, stem=True)


def test_index_covers_every_document(index):
    assert index.n_docs == 7
    assert index.docnos == [
        "FT911-1",
        "FT911-2",
        "FT911-3",
        "FT911-4",
        "FT911-5",
        "FT911-6",
        "FT911-10",
    ]
    assert len(index.doc_len) == 7
    assert index.stem is True
    assert index.stop is True


def test_jet_postings_count_every_surface_form(index):
    # doc1 "The jet flew. Jets fly fast, and the jet's engine roared 3 times."
    # -> jet, jets->jet and jet('s)->jet, so tf 3; doc10 "... about jets." -> 1.
    assert index.df("jet") == 2
    assert index.postings["jet"] == [(0, 3), (6, 1)]


def test_stemming_keeps_banker_apart_from_bank(index):
    # doc2 "Banks lend money; the bank lent 100 pounds to a banker."
    assert index.postings["bank"] == [(1, 2)]
    assert index.postings["banker"] == [(1, 1)]
    assert index.df("bank") == 1


def test_doc_len_counts_terms_kept(index):
    # doc6 "A short one." -> "a" is a stopword, so ["short", "one"].
    assert index.doc_len[5] == 2
    assert index.total_tokens == sum(index.doc_len)
    assert index.avgdl == pytest.approx(index.total_tokens / 7)


def test_postings_are_sorted_by_doc_index_and_positive(index):
    for term, plist in index.postings.items():
        idxs = [d for d, _ in plist]
        assert idxs == sorted(idxs), term
        assert len(set(idxs)) == len(idxs), term
        assert all(tf > 0 for _, tf in plist), term


def test_df_of_missing_term_is_zero(index):
    assert index.df("nosuchtermanywhere") == 0


def test_without_stopwords_or_stemming_the_survives(docs):
    plain = build_index(docs, None, stem=False)
    # "the" appears in docs FT911-1, 2, 3 and 4 and in no other fixture doc.
    assert plain.df("the") == 4
    assert plain.stem is False
    assert plain.stop is False
    assert plain.total_tokens > build_index(docs, STOPWORDS, stem=False).total_tokens


def test_index_uses_the_text_field_only(docs):
    # "Cranwell" is only in doc1's headline, never in any text.
    assert build_index(docs, None, stem=False).df("cranwell") == 0


def test_build_all_returns_the_four_treatments(docs):
    indexes = build_all(docs, STOPWORDS)
    assert set(indexes) == set(TREATMENTS)
    for key, (stem, stop) in TREATMENTS.items():
        assert isinstance(indexes[key], Index)
        assert indexes[key].stem is stem
        assert indexes[key].stop is stop
    assert indexes["stem_stop"].total_tokens == indexes["nostem_stop"].total_tokens
    assert len(indexes["stem_stop"].postings) <= len(indexes["nostem_stop"].postings)
    assert indexes["stem_nostop"].total_tokens > indexes["stem_stop"].total_tokens
