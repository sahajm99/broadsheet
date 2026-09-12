"""Parser tests against the hand-written mini corpus."""

from pathlib import Path

import pytest

from pipeline.parse import Doc, clean_headline, iso_date, load_corpus, parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


def test_load_corpus_returns_every_doc_in_docno_order():
    docs = load_corpus(FIXTURE)
    assert [d.docno for d in docs] == [
        "FT911-1",
        "FT911-2",
        "FT911-3",
        "FT911-4",
        "FT911-5",
        "FT911-6",
        "FT911-10",
    ]


def test_first_doc_fields():
    doc = load_corpus(FIXTURE)[0]
    assert isinstance(doc, Doc)
    assert doc.headline == "Jets over Cranwell"
    assert doc.date == "1991-05-14"
    assert "engine roared" in doc.text


def test_parse_file_reads_one_record():
    docs = parse_file(FIXTURE / "ft_mini_2")
    assert len(docs) == 1
    assert docs[0].docno == "FT911-10"
    assert docs[0].headline == "Tenth"
    assert docs[0].date == "1991-05-15"


def test_text_excludes_markup_and_other_fields():
    doc = load_corpus(FIXTURE)[1]  # FT911-2, the record carrying a BYLINE
    assert "<" not in doc.text
    assert "CORRESPONDENT" not in doc.text
    assert "Financial Times" not in doc.text
    assert doc.text.startswith("Banks lend money")


def test_clean_headline_strips_the_ft_dateline_prefix_and_wrapping():
    assert (
        clean_headline("FT  14 MAY 91 / (CORRECTED) Jubilee\nof a jet")
        == "(CORRECTED) Jubilee of a jet"
    )


def test_clean_headline_leaves_a_headline_without_the_prefix_alone():
    assert clean_headline("\n  Survey of Japan   (12):  Banks \n") == "Survey of Japan (12): Banks"


def test_clean_headline_cuts_to_200_chars():
    assert len(clean_headline("FT  14 MAY 91 / " + "word " * 100)) == 200


def test_iso_date():
    assert iso_date("910514") == "1991-05-14"
    assert iso_date("911231") == "1991-12-31"
    assert iso_date("901002") == "1990-10-02"


def test_missing_raw_dir_raises_pointing_at_source_md(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_corpus(tmp_path / "not-there")
    assert "SOURCE.md" in str(excinfo.value)


def test_empty_raw_dir_raises_pointing_at_source_md(tmp_path):
    empty = tmp_path / "ft911"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        load_corpus(empty)
    assert "SOURCE.md" in str(excinfo.value)
