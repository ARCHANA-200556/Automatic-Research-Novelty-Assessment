"""
Unit tests for dynamic multi-dimensional research gap analysis across diverse scientific domains.
"""

import pytest
from app.config import ResearchInput, PaperMetadata, SimilarityItem
from app.gap_analyzer import analyze_multidimensional_research_gaps


def test_analyze_multidimensional_research_gaps_edge_ai():
    proposal = ResearchInput(
        title="Novel Edge Multimodal Transformer for Tiny Microcontrollers",
        abstract="We design a sub-10M parameter vision-language model running on ARM Cortex-M microcontrollers.",
        objectives=["Deploy multimodal model on 2MB SRAM MCU", "Evaluate 2-bit quantization"]
    )

    ranked_papers = [
        SimilarityItem(
            paper=PaperMetadata(
                id="W1",
                title="Cloud-Scale Vision Transformers",
                abstract="Training massive 100B parameter vision models on supercomputers.",
                citation_count=500
            ),
            similarity_score=0.25,
            rank=1
        )
    ]

    gaps, gap_score = analyze_multidimensional_research_gaps(proposal, ranked_papers)
    assert len(gaps) == 5  # Objectives, Methods, Datasets, Domain, Contributions
    assert 0.0 <= gap_score <= 1.0
    
    # Low similarity should identify potential gaps
    potential_count = sum(1 for g in gaps if "POTENTIAL GAP" in g.classification)
    assert potential_count >= 3


def test_analyze_multidimensional_research_gaps_biomedical():
    # Test biomedical / CRISPR topic to ensure zero hardcoded NLP bias
    proposal = ResearchInput(
        title="Lipid Nanoparticle Formulations for In Vivo CRISPR Base Editing of Transthyretin Amyloidosis",
        abstract="We engineer ionizable lipid nanoparticles delivering Cas9 mRNA and sgRNA targeting hepatic TTR genes.",
        objectives=["Synthesize novel amino-lipid libraries", "Measure serum TTR knockdown in humanized mice"]
    )

    ranked_papers = [
        SimilarityItem(
            paper=PaperMetadata(
                id="BIO_1",
                title="CRISPR-Cas9 Therapeutics in Transthyretin Amyloidosis",
                abstract="Phase 1 clinical trial evaluating in vivo gene editing with lipid nanoparticles for hereditary ATTR.",
                citation_count=280
            ),
            similarity_score=0.88,
            rank=1
        )
    ]

    gaps, gap_score = analyze_multidimensional_research_gaps(proposal, ranked_papers)
    assert len(gaps) == 5
    # High similarity (0.88) with directly competing clinical trial should classify objectives as VERIFIED OVERLAP
    obj_gap = [g for g in gaps if g.dimension == "Research Objectives"][0]
    assert obj_gap.classification == "VERIFIED OVERLAP"
    # Ensure no hardcoded terms like 'SciBERT' or 'CiteGNN' appear in any rationale
    for g in gaps:
        assert "scibert" not in g.rationale.lower()
        assert "citegnn" not in g.rationale.lower()
