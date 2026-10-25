"""
Unit tests for semantic vector similarity analysis.
"""

import pytest
from app.config import PaperMetadata
from app.embeddings import compute_semantic_similarities, extract_key_overlapping_terms


def test_compute_semantic_similarities_bounds():
    proposal_text = "Automated novelty assessment using dense sentence embeddings and citation graphs."
    papers = [
        PaperMetadata(
            id="W1",
            title="Evaluating Novelty in Scientific Papers",
            abstract="We evaluate novelty and originality using sentence embeddings.",
            citation_count=50
        ),
        PaperMetadata(
            id="W2",
            title="Photosynthesis Mechanisms in Marine Algae",
            abstract="Study of light-harvesting protein complexes in phytoplankton.",
            citation_count=12
        )
    ]

    items, engine = compute_semantic_similarities(proposal_text, papers)
    assert len(items) == 2
    # The first item should be W1 because it is about novelty and embeddings
    assert items[0].paper.id == "W1"
    assert items[0].similarity_score > items[1].similarity_score
    # Both scores must be bounded in [0.0, 1.0]
    assert 0.0 <= items[0].similarity_score <= 1.0
    assert 0.0 <= items[1].similarity_score <= 1.0


def test_extract_key_overlapping_terms():
    text1 = "Graph neural networks for citation analysis and novelty detection."
    text2 = "Citation networks and graph topology for novelty evaluation."
    overlap = extract_key_overlapping_terms(text1, text2)
    assert any("graph" in t.lower() or "citation" in t.lower() or "novelty" in t.lower() for t in overlap)
