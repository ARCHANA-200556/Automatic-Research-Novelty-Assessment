"""
Unit tests for novelty scoring engine, formula invariants, and bounds.
"""

import pytest
from app.config import (
    ResearchInput,
    PaperMetadata,
    SimilarityItem,
    GapDimensionItem,
    CitationGraphMetrics
)
from app.novelty_scorer import compute_novelty_assessment


def test_compute_novelty_assessment_high_novelty():
    proposal = ResearchInput(
        title="Unprecedented Quantum-Biological Synthesis Model",
        abstract="Novel theoretical formulation spanning disparate scientific regimes."
    )

    # Low similarity papers (e.g. 0.15)
    ranked = [
        SimilarityItem(
            paper=PaperMetadata(id="W1", title="Quantum Entanglement Basics", abstract="Study of qubits.", citation_count=20),
            similarity_score=0.15,
            rank=1
        ),
        SimilarityItem(
            paper=PaperMetadata(id="W2", title="Cellular Membrane Dynamics", abstract="Study of lipids.", citation_count=40),
            similarity_score=0.10,
            rank=2
        )
    ]

    gaps = [
        GapDimensionItem(dimension="Objectives", proposed_aspect="A", prior_art_state="B", classification="POTENTIAL GAP", rationale="R", confidence=0.8),
        GapDimensionItem(dimension="Methods", proposed_aspect="A", prior_art_state="B", classification="POTENTIAL GAP", rationale="R", confidence=0.8),
        GapDimensionItem(dimension="Datasets", proposed_aspect="A", prior_art_state="B", classification="POTENTIAL GAP", rationale="R", confidence=0.8),
        GapDimensionItem(dimension="Domain", proposed_aspect="A", prior_art_state="B", classification="POTENTIAL GAP", rationale="R", confidence=0.8),
        GapDimensionItem(dimension="Contributions", proposed_aspect="A", prior_art_state="B", classification="POTENTIAL GAP", rationale="R", confidence=0.8)
    ]

    c_metrics = CitationGraphMetrics(citation_orthogonality_score=0.85)

    critique = {
        "comparisons": [],
        "distinctive_contributions": ["Cross-domain bridge"],
        "literature_limitations": ["Limited prior art"]
    }

    res = compute_novelty_assessment(
        proposal=proposal,
        ranked_papers=ranked,
        gap_items=gaps,
        gap_score=1.0,
        citation_metrics=c_metrics,
        rag_critique=critique,
        is_live_retrieval=True,
        embedding_engine_desc="Vector Space",
        llm_source_used="Rule Grounded"
    )

    # Must be bounded and reflect high novelty
    assert 0.0 <= res.overall_novelty_score <= 100.0
    assert res.overall_novelty_score > 75.0
    assert "High Potential Novelty" in res.novelty_tier


def test_compute_novelty_assessment_low_novelty_direct_overlap():
    proposal = ResearchInput(
        title="BERT Pretraining for Scientific Text",
        abstract="We pretrain BERT on scientific articles."
    )

    # Identical or near-identical prior art
    ranked = [
        SimilarityItem(
            paper=PaperMetadata(id="W1", title="SciBERT: A Pretrained Language Model for Scientific Text", abstract="Pretrained BERT on scientific literature.", citation_count=3000),
            similarity_score=0.95,
            rank=1
        )
    ]

    gaps = [
        GapDimensionItem(dimension="Objectives", proposed_aspect="A", prior_art_state="B", classification="VERIFIED OVERLAP", rationale="R", confidence=0.9)
    ]

    c_metrics = CitationGraphMetrics(citation_orthogonality_score=0.2)

    critique = {"comparisons": [], "distinctive_contributions": [], "literature_limitations": []}

    res = compute_novelty_assessment(
        proposal=proposal,
        ranked_papers=ranked,
        gap_items=gaps,
        gap_score=0.0,
        citation_metrics=c_metrics,
        rag_critique=critique,
        is_live_retrieval=True,
        embedding_engine_desc="Vector Space",
        llm_source_used="Rule Grounded"
    )

    # Heavy similarity penalty should lower the score drastically
    assert res.overall_novelty_score < 40.0
    assert "Direct Overlap" in res.novelty_tier
