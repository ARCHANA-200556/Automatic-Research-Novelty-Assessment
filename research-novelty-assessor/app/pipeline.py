"""
End-to-End Novelty Assessment Pipeline Orchestrator.
Unifies multi-source retrieval (OpenAlex, Semantic Scholar, Crossref),
sentence-transformers embeddings, grounded RAG comparative reasoning,
dynamic research gap analysis, and citation network construction.
"""

import logging
from typing import Tuple, Dict, Any, Optional
import networkx as nx
import plotly.graph_objects as go

from app.config import ResearchInput, NoveltyAssessmentResult
from app.retrieval import retrieve_relevant_literature
from app.embeddings import compute_semantic_similarities
from app.rag_engine import retrieve_grounded_evidence, generate_rag_comparative_critique
from app.gap_analyzer import analyze_multidimensional_research_gaps
from app.citation_graph import build_citation_network, create_plotly_network_figure
from app.novelty_scorer import compute_novelty_assessment

logger = logging.getLogger(__name__)


def run_full_novelty_assessment(
    proposal: ResearchInput,
    limit_papers: int = 10,
    force_fallback: bool = False,
    progress_callback = None
) -> Tuple[NoveltyAssessmentResult, nx.DiGraph, go.Figure]:
    """
    Executes the comprehensive automated novelty assessment workflow for ANY research topic.
    Optional progress_callback(step: int, total_steps: int, message: str) updates UI.
    """
    def update_progress(step: int, msg: str):
        if progress_callback:
            progress_callback(step, 6, msg)

    # 1. Multi-Source Literature Retrieval
    update_progress(1, "Searching OpenAlex, Semantic Scholar, and Crossref APIs...")
    papers, ret_status = retrieve_relevant_literature(
        query_title=proposal.title,
        keywords=proposal.keywords,
        limit=limit_papers,
        force_fallback=force_fallback,
        abstract=proposal.abstract,
        objectives=proposal.objectives
    )

    if not papers:
        diag = ret_status.get("diagnostics") or f"No peer-reviewed literature found for '{proposal.title}'."
        raise ValueError(diag)

    # 2. Semantic Embedding & Similarity Analysis
    update_progress(2, f"Embedding {len(papers)} candidate papers and computing cosine similarity...")
    proposal_repr = f"{proposal.title}. {proposal.abstract} " + " ".join(proposal.objectives)
    ranked_papers, emb_engine_desc = compute_semantic_similarities(proposal_repr, papers)

    # 3. Grounded RAG Evidence Extraction & Comparative Synthesis
    update_progress(3, "Extracting evidence passages and comparing methodologies and aims...")
    evidence_chunks = retrieve_grounded_evidence(proposal, ranked_papers, top_passages_count=10)
    rag_critique, llm_source_used = generate_rag_comparative_critique(
        proposal, ranked_papers, evidence_chunks
    )

    # 4. Dynamic Multi-Dimensional Research Gap Analysis
    update_progress(4, "Classifying research gaps across 5 dimensions (Objectives, Methods, Data, Domain, Contributions)...")
    gap_items, gap_score = analyze_multidimensional_research_gaps(proposal, ranked_papers)

    # 5. Citation Graph Analysis & Structural Centrality
    update_progress(5, "Constructing NetworkX citation graph and mapping topology...")
    G, citation_metrics = build_citation_network(proposal, ranked_papers)
    fig_network = create_plotly_network_figure(G)

    # 6. Novelty Scoring & Consolidated Report Compilation
    update_progress(6, "Calculating mathematical novelty score and compiling explainable assessment...")
    result = compute_novelty_assessment(
        proposal=proposal,
        ranked_papers=ranked_papers,
        gap_items=gap_items,
        gap_score=gap_score,
        citation_metrics=citation_metrics,
        rag_critique=rag_critique,
        is_live_retrieval=not ret_status.get("is_fallback", False),
        embedding_engine_desc=emb_engine_desc,
        llm_source_used=llm_source_used,
        retrieval_stats=ret_status,
        retrieval_diagnostics=ret_status.get("diagnostics")
    )

    return result, G, fig_network
