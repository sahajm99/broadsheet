"""Topic and qrels loading, against the real TREC files and a mini fixture.

The real files are committed (they are NIST's, not the Financial Times'), so
every count here is checked against the collection the evaluation actually
runs on.
"""

from pathlib import Path

import pytest

from pipeline.topics import FIELDS, Topic, load_qrels, load_topics, query_text

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


@pytest.fixture(scope="module")
def topics():
    return load_topics()


@pytest.fixture(scope="module")
def by_num(topics):
    return {t.num: t for t in topics}


def test_every_topic_is_loaded_in_ascending_order(topics):
    assert len(topics) == 250
    nums = [t.num for t in topics]
    assert nums == sorted(nums)
    assert nums[0] == 301
    assert nums[-1] == 700
    assert all(isinstance(t, Topic) for t in topics)


def test_first_topic_fields(topics):
    first = topics[0]
    assert first.num == 301
    assert first.title == "International Organized Crime"
    assert first.desc.startswith("Identify organizations that participate")
    assert "Description:" not in first.desc
    assert first.narr.startswith("A relevant document must as a minimum")
    assert "Narrative:" not in first.narr


def test_topic_352_is_the_chunnel_topic(by_num):
    t352 = by_num[352]
    assert t352.title == "British Chunnel impact"
    assert "Chunnel had on the British economy" in t352.desc
    assert "routine marketing ploys" in t352.narr


def test_the_601_700_block_with_bare_tags_and_a_title_on_its_own_line(by_num):
    assert 700 in by_num
    assert by_num[700].title == "gasoline tax U.S."
    assert by_num[700].desc.startswith("What are the arguments for and against")
    # 672 has no judgment in the FT911 slice but is still a topic in the file.
    assert by_num[672].title == "NRA membership profile"


def test_no_field_carries_markup_or_ragged_whitespace(topics):
    for t in topics:
        for value in (t.title, t.desc, t.narr):
            assert value == " ".join(value.split()), t.num
            assert "<" not in value, t.num
            assert value, t.num


def test_query_text_concatenates_the_requested_fields(by_num):
    t = by_num[352]
    assert query_text(t, "title") == "British Chunnel impact"
    td = query_text(t, "title_desc")
    assert td.startswith("British Chunnel impact")
    assert "Chunnel had on the British economy" in td
    tdn = query_text(t, "title_desc_narr")
    assert tdn.startswith(td)
    assert "routine marketing ploys" in tdn
    assert len(tdn) > len(td) > len(query_text(t, "title"))


def test_query_text_rejects_an_unknown_field(by_num):
    with pytest.raises(ValueError, match="field"):
        query_text(by_num[301], "narr")


def test_fields_lists_the_three_runs():
    assert list(FIELDS) == ["title", "title_desc", "title_desc_narr"]


def test_qrels_cover_every_judgment_in_the_file():
    qrels = load_qrels()
    assert sum(len(v) for v in qrels.values()) == 3019
    assert len(qrels) == 241
    assert all(v in (0, 1) for docs in qrels.values() for v in docs.values())


def test_qrels_binarise_relevance():
    qrels = load_qrels()
    assert sum(qrels[353].values()) == 10
    # 71 topics have at least one relevant document inside the FT911 slice.
    assert sum(1 for docs in qrels.values() if any(docs.values())) == 71


def test_qrels_filter_to_the_documents_given():
    fixture = FIXTURE / "qrels.txt"
    all_judged = load_qrels(fixture)
    assert all_judged == {
        1: {"FT911-1": 1, "FT911-10": 1, "FT911-2": 0},
        2: {"FT911-2": 1},
    }
    filtered = load_qrels(fixture, docnos={"FT911-1", "FT911-2", "FT911-404"})
    assert filtered == {1: {"FT911-1": 1, "FT911-2": 0}, 2: {"FT911-2": 1}}


def test_qrels_drop_a_topic_whose_judgments_are_all_filtered_out():
    filtered = load_qrels(FIXTURE / "qrels.txt", docnos={"FT911-10"})
    assert filtered == {1: {"FT911-10": 1}}


def test_fixture_topics_parse_both_layouts():
    topics = load_topics(FIXTURE / "topics.txt")
    assert [t.num for t in topics] == [1, 2]
    assert topics[0] == Topic(
        num=1, title="jet engine", desc="Documents about jets.", narr="Any jet."
    )
    assert topics[1] == Topic(
        num=2, title="bank money", desc="Documents about banks.", narr="Any bank."
    )
    assert query_text(topics[0], "title_desc_narr") == (
        "jet engine Documents about jets. Any jet."
    )
