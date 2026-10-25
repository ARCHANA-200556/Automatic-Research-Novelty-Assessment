"""
Unit tests for multi-source academic paper retrieval, deduplication, and benchmark mode.
"""

import pytest
from app.retrieval import (
    reconstruct_openalex_abstract,
    clean_crossref_abstract,
    normalize_doi,
    normalize_title_for_dedup,
    deduplicate_and_merge_papers,
    load_fallback_papers,
    retrieve_relevant_literature
)
from app.config import PaperMetadata


def test_reconstruct_openalex_abstract():
    inverted_index = {
        "Scientific": [0],
        "discovery": [1],
        "is": [2],
        "accelerated": [3],
        "by": [4],
        "AI": [5]
    }
    abstract = reconstruct_openalex_abstract(inverted_index)
    assert abstract == "Scientific discovery is accelerated by AI"


def test_clean_crossref_abstract():
    raw = "<jats:p>This is a <jats:bold>critical</jats:bold> study of nanomaterials.</jats:p>"
    cleaned = clean_crossref_abstract(raw)
    assert "This is a critical study of nanomaterials." in cleaned
    assert "<" not in cleaned


def test_normalize_doi():
    doi1 = "https://doi.org/10.1016/j.cell.2023.01.001"
    doi2 = "10.1016/j.cell.2023.01.001"
    doi3 = "http://dx.doi.org/10.1016/J.CELL.2023.01.001 "
    assert normalize_doi(doi1) == "10.1016/j.cell.2023.01.001"
    assert normalize_doi(doi2) == "10.1016/j.cell.2023.01.001"
    assert normalize_doi(doi3) == "10.1016/j.cell.2023.01.001"


def test_deduplicate_and_merge_papers():
    p1 = PaperMetadata(
        id="OA1",
        title="CRISPR Gene Editing for Sickle Cell",
        abstract="Brief summary.",
        doi="10.1000/182",
        source_api="OpenAlex",
        citation_count=10
    )
    p2 = PaperMetadata(
        id="CR1",
        title="CRISPR Gene Editing for Sickle Cell",
        abstract="Comprehensive expanded abstract describing therapeutic outcomes in patient cohorts.",
        doi="https://doi.org/10.1000/182",
        source_api="Crossref",
        citation_count=15
    )
    p3 = PaperMetadata(
        id="OA2",
        title="Quantum Dot Solar Cell Architecture",
        abstract="Study of perovskite stability.",
        doi="10.1000/999",
        source_api="OpenAlex",
        citation_count=50
    )

    merged = deduplicate_and_merge_papers([[p1], [p2], [p3]], limit=10)
    assert len(merged) == 2  # p1 and p2 merged by DOI into 1 item
    # Check that abstract was enriched with the longer one
    matched = [p for p in merged if p.doi and "10.1000/182" in p.doi][0]
    assert "Comprehensive expanded" in matched.abstract
    assert matched.citation_count >= 15


def test_load_fallback_papers():
    papers = load_fallback_papers()
    assert len(papers) > 0
    p = papers[0]
    assert p.id.startswith("W")
    assert len(p.title) > 5
    assert len(p.abstract) > 20
    assert p.is_fallback is True


def test_retrieve_literature_with_fallback():
    papers, stats = retrieve_relevant_literature(
        query_title="Novelty Assessment",
        force_fallback=True
    )
    assert len(papers) > 0
    assert stats["is_fallback"] is True
    assert stats["analyzed_count"] == len(papers)
