"""Citations: source names, links and snippets shown next to every answer."""
from rag.citation_tracker import (
    clean_snippet,
    infer_doc_url,
    normalize_source,
    stable_source_id,
)


def test_source_aliases_are_normalized():
    assert normalize_source("Daily-Med") == "dailymed"
    assert normalize_source("fda_label") == "dailymed"
    assert normalize_source("PMC") == "pubmed"
    assert normalize_source(None) == "unknown"


def test_pubmed_link_from_pmid():
    assert infer_doc_url("pubmed", "PMID: 12345678") == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_drugbank_link_only_for_drugbank_ids():
    assert infer_doc_url("drugbank", "DB00682") == "https://go.drugbank.com/drugs/DB00682"
    assert infer_doc_url("drugbank", "warfarin") == ""


def test_existing_url_is_kept():
    assert infer_doc_url("pubmed", "123456", "https://example.org/x") == "https://example.org/x"


def test_snippets_are_compacted_and_trimmed():
    assert clean_snippet("  warfarin\n\n  and   aspirin ") == "warfarin and aspirin"
    long = clean_snippet("x" * 500, max_chars=50)
    assert len(long) <= 52 and long.endswith("...")


def test_chunks_from_the_same_section_share_one_citation():
    a = stable_source_id("DailyMed", "abc-123", "warnings", chunk_id="1")
    b = stable_source_id("daily_med", "ABC-123", "WARNINGS", chunk_id="2")
    assert a == b
