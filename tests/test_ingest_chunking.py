from rag.ingest import (
    clean_text,
    _split_text_into_sections,
    _split_section_into_clauses,
    build_parent_child_chunks,
)
from langchain_core.documents import Document


SAMPLE_ACT_TEXT = (
    "परिच्छेद–१\n"
    "प्रारम्भिक\n"
    "\n"
    "१. संक्षिप्त नाम र प्रारम्भ: (क) यो ऐनको नाम कम्पनी ऐन, २०६३ रहेको छ ।\n"
    "(ख) यो ऐन सम्पूर्ण नेपालमा लागू हुनेछ ।\n"
    "(ग) यो ऐन तुरुन्त लागू हुनेछ ।\n"
    "\n"
    "२. परिभाषा: यस ऐनमा अन्यथा अर्थ नलागेमा:\n"
    "(क) कम्पनी भन्नाले प्रचलित कानून बमोजिम दर्ता भएको संस्था सम्झनु पर्छ ।\n"
    "(ख) सञ्चालक भन्नाले कम्पनीको सञ्चालक सम्झनु पर्छ ।\n"
)


def test_clean_text_collapses_whitespace_and_normalizes():
    assert clean_text("  Hello   \n\n World  ") == "Hello World"


def test_split_text_into_sections_finds_each_dafa():
    sections = _split_text_into_sections(SAMPLE_ACT_TEXT)

    # The preamble (परिच्छेद heading) becomes its own leading section,
    # followed by one section per दफा marker (१. and २.).
    assert len(sections) == 3
    assert "संक्षिप्त नाम" in sections[1]["title"]
    assert "परिभाषा" in sections[2]["title"]


def test_split_text_into_sections_tags_chapter():
    sections = _split_text_into_sections(SAMPLE_ACT_TEXT)
    assert all("परिच्छेद" in s["parichhed"] for s in sections)


def test_split_text_into_sections_handles_text_without_markers():
    text = "This document has no दफा markers at all, just plain prose."
    sections = _split_text_into_sections(text)
    assert len(sections) == 1
    assert sections[0]["text"] == clean_text(text)


def test_split_section_into_clauses_splits_on_clause_markers():
    section_text = (
        "१. संक्षिप्त नाम: (क) पहिलो खण्ड हो । (ख) दोस्रो खण्ड हो । (ग) तेस्रो खण्ड हो ।"
    )
    groups = _split_section_into_clauses(section_text)

    assert len(groups) > 1
    all_markers = ",".join(g["clause_markers"] for g in groups)
    assert "(क)" in all_markers
    assert "(ख)" in all_markers
    assert "(ग)" in all_markers
    assert groups[0]["has_preamble"] is True


def test_split_section_into_clauses_single_chunk_when_no_markers():
    section_text = "दफा जसमा कुनै खण्ड चिन्ह छैन ।"
    groups = _split_section_into_clauses(section_text)
    assert len(groups) == 1
    assert groups[0]["clause_markers"] == ""


def test_build_parent_child_chunks_links_children_to_parent():
    doc = Document(page_content=SAMPLE_ACT_TEXT, metadata={"source": "test.txt"})
    parents, children = build_parent_child_chunks([doc])

    assert len(parents) == 3  # preamble + two दफा sections
    assert len(children) > 0

    parent_ids = {p.metadata["parent_id"] for p in parents}
    assert all(c.metadata["parent_id"] in parent_ids for c in children)
    assert all(c.metadata["type"] == "child" for c in children)
    assert all(p.metadata["type"] == "parent" for p in parents)
