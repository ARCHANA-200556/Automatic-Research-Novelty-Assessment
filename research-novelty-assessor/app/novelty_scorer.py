"""
Novelty Scoring Engine and Confidence Assessment Module.
Implements the mathematically documented 0-100 novelty scoring equation
based on semantic similarity penalties, gap bonuses, and citation graph topology.
"""

import math
from typing import List, Tuple, Dict, Any
from app.config import (
    ResearchInput,
    SimilarityItem,
    GapDimensionItem,
    CitationGraphMetrics,
    NoveltyAssessmentResult,
    WEIGHT_MAX_SIMILARITY,
    WEIGHT_MEAN_SIMILARITY,
    WEIGHT_GAP_EVIDENCE,
    WEIGHT_CITATION_ORTHOGONALITY
)


def compute_novelty_assessment(
    proposal: ResearchInput,
    ranked_papers: List[SimilarityItem],
    gap_items: List[GapDimensionItem],
    gap_score: float,
    citation_metrics: CitationGraphMetrics,
    rag_critique: Dict[str, Any],
    is_live_retrieval: bool,
    embedding_engine_desc: str,
    llm_source_used: str,
    retrieval_stats: Dict[str, Any] = None,
    retrieval_diagnostics: str = None
) -> NoveltyAssessmentResult:
    """
    Calculates the documented 0-100 Potential Novelty Score, confidence coverage,
    and consolidates the complete explainable assessment.
    """
    # 1. Similarity metrics
    if ranked_papers:
        max_sim = ranked_papers[0].similarity_score
        top_k = ranked_papers[:min(5, len(ranked_papers))]
        mean_sim = sum(p.similarity_score for p in top_k) / len(top_k)
    else:
        max_sim = 0.0
        mean_sim = 0.0

    # 2. Factor calculations
    # Sub-component 1: Single closest paper divergence (1 - MaxSim)
    factor_max_sim = max(0.0, 1.0 - max_sim)
    
    # Sub-component 2: Local neighborhood divergence (1 - MeanTopK)
    factor_mean_sim = max(0.0, 1.0 - mean_sim)
    
    # Sub-component 3: Multi-dimensional gap bonus (GapScore)
    factor_gap = gap_score
    
    # Sub-component 4: Citation graph orthogonality (CitationScore)
    factor_citation = citation_metrics.citation_orthogonality_score

    # 3. Documented Weighted Composite Equation:
    # S = 100 * [ w_s * (1 - MaxSim) + w_m * (1 - MeanTopK) + w_g * GapScore + w_c * CitationScore ]
    raw_score = (
        (WEIGHT_MAX_SIMILARITY * factor_max_sim) +
        (WEIGHT_MEAN_SIMILARITY * factor_mean_sim) +
        (WEIGHT_GAP_EVIDENCE * factor_gap) +
        (WEIGHT_CITATION_ORTHOGONALITY * factor_citation)
    ) * 100.0

    overall_score = round(max(0.0, min(100.0, raw_score)), 1)

    # Sub-scores for explainability
    sub_semantic = round(((WEIGHT_MAX_SIMILARITY * factor_max_sim) + (WEIGHT_MEAN_SIMILARITY * factor_mean_sim)) * 100.0, 1)
    sub_gap = round((WEIGHT_GAP_EVIDENCE * factor_gap) * 100.0, 1)
    sub_citation = round((WEIGHT_CITATION_ORTHOGONALITY * factor_citation) * 100.0, 1)

    # 4. Novelty Tier Categorization
    if overall_score >= 80.0:
        novelty_tier = "High Potential Novelty (Pioneer Direction)"
    elif overall_score >= 60.0:
        novelty_tier = "Moderate Novelty (Substantial Extension)"
    elif overall_score >= 40.0:
        novelty_tier = "Incremental Advance (Notable Prior Overlap)"
    else:
        novelty_tier = "Direct Overlap (Saturated Topic Area)"

    # 5. Confidence and Evidence Coverage Score
    total_papers = len(ranked_papers)
    abstracts_available = sum(1 for p in ranked_papers if len(p.paper.abstract) > 50)
    
    volume_cov = min(1.0, total_papers / 8.0) * 0.40
    content_cov = (abstracts_available / max(1, total_papers)) * 0.35
    source_cov = (0.25 if is_live_retrieval else 0.20)
    confidence_coverage = round((volume_cov + content_cov + source_cov) * 100.0, 1)

    # 6. Generate narrative executive summary
    executive_summary = (
        f"The proposed research '{proposal.title}' achieved an estimated Potential Novelty Score of "
        f"{overall_score}/100 ({novelty_tier}) with an Evidence Coverage & Confidence of {confidence_coverage}%. "
        f"The maximum conceptual similarity to identified literature is {max_sim:.1%}, with an average top-5 "
        f"similarity of {mean_sim:.1%}. The multidimensional gap analysis identifies {int(gap_score * 5)}/5 "
        f"unaddressed potential gap dimensions. Citation topology analysis indicates a structural orthogonality "
        f"rating of {factor_citation:.2f}, suggesting the work "
        f"{'synthesizes disparate research subfields' if factor_citation > 0.6 else 'resides within an existing citation cluster'}."
    )

    return NoveltyAssessmentResult(
        proposal=proposal,
        overall_novelty_score=overall_score,
        novelty_tier=novelty_tier,
        confidence_coverage_score=confidence_coverage,
        semantic_penalty_score=sub_semantic,
        gap_bonus_score=sub_gap,
        citation_bonus_score=sub_citation,
        max_similarity=round(max_sim, 4),
        mean_similarity=round(mean_sim, 4),
        retrieved_papers=ranked_papers,
        gap_analysis=gap_items,
        citation_metrics=citation_metrics,
        executive_summary=executive_summary,
        methodological_comparisons=rag_critique.get("comparisons", []),
        distinctive_contributions=rag_critique.get("distinctive_contributions", []),
        limitations_and_recommendations=rag_critique.get("literature_limitations", []),
        retrieval_stats=retrieval_stats or {},
        retrieval_diagnostics=retrieval_diagnostics,
        is_live_retrieval=is_live_retrieval,
        llm_source_used=llm_source_used
    )
