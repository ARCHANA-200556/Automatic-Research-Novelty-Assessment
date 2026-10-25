"""
Unit tests for input parser, PDF text extraction, query generation, and sample presets.
"""

import pytest
import io
import pypdf
from app.config import ResearchInput
from app.input_parser import (
    clean_text,
    build_manual_research_input,
    parse_pdf_document,
    load_presets,
    generate_academic_search_queries
)


def test_clean_text():
    dirty = "This  is   a   test\n\n\n\nof  hyphen-\nation and spacing."
    cleaned = clean_text(dirty)
    assert "hyphenation" in cleaned
    assert "  " not in cleaned


def test_build_manual_research_input():
    title = "Novel Graph Neural Network for Drug Discovery"
    abstract = "We present a graph transformer architecture to screen kinase inhibitors."
    keywords = "GNN, Drug Discovery, Transformers"
    objectives = "1. Train graph model\n2. Benchmark against AlphaFold"

    res = build_manual_research_input(title, abstract, keywords, objectives)
    assert res.title == title
    assert len(res.keywords) == 3
    assert len(res.objectives) == 2
    assert "Train graph model" in res.objectives[0]


def test_load_presets():
    presets = load_presets()
    assert len(presets) >= 1
    first = presets[0]
    assert "title" in first
    assert "abstract" in first
    assert "objectives" in first


def test_generate_academic_search_queries():
    proposal = ResearchInput(
        title="An Investigation into Perovskite Solar Cells Degradation Under Thermal Stress",
        abstract="We investigate halide segregation and chemical stability in triple-cation perovskite solar cells.",
        keywords=["Perovskite", "Solar Cells", "Degradation"]
    )
    queries = generate_academic_search_queries(proposal)
    assert len(queries) >= 2
    # Verify boilerplate was stripped from the clean query
    assert not queries[0].lower().startswith("an investigation into")
    assert "perovskite" in queries[0].lower()


def test_parse_pdf_in_memory():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    pdf_buffer = io.BytesIO()
    writer.write(pdf_buffer)
    pdf_buffer.seek(0)
    pdf_buffer.name = "test_empty.pdf"

    parsed, msg = parse_pdf_document(pdf_buffer)
    assert parsed is None or "little extractable text" in msg
