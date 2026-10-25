"""
Explainable Report Generator and Exporter.
Produces comprehensive Markdown, JSON, and structured assessment reports
incorporating multi-source literature metrics, RAG evidence passages, and clickable citations.
"""

import json
from datetime import datetime
from typing import Dict, Any
from app.config import NoveltyAssessmentResult


def export_markdown_report(result: NoveltyAssessmentResult) -> str:
    """Generates a complete, publication-grade Markdown assessment report."""
    prop = result.proposal
    p_papers = result.retrieved_papers
    c_metrics = result.citation_metrics
    stats = result.retrieval_stats

    # Retrieval stats overview
    oa_c = stats.get("openalex_count", 0)
    s2_c = stats.get("semantic_scholar_count", 0)
    cr_c = stats.get("crossref_count", 0)
    dedup_c = stats.get("unique_deduplicated", len(p_papers))

    # Papers table with source badges
    papers_rows = []
    for item in p_papers:
        p = item.paper
        link = f"[{p.title}]({p.url})" if p.url else p.title
        authors_str = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        overlap = ", ".join(item.key_overlap_terms[:3]) if item.key_overlap_terms else "General"
        src = getattr(p, "source_api", "Academic Index")
        papers_rows.append(
            f"| #{item.rank} | {link} | {authors_str} | {p.year or 'N/A'} | {item.similarity_score:.1%} | {p.citation_count:,} | {src} | {overlap} |"
        )
    papers_table = "\n".join(papers_rows)

    # Comparative matrix
    comp_rows = []
    for comp in result.methodological_comparisons:
        comp_rows.append(
            f"### Dimension: {comp.get('aspect', 'Aspect')}\n"
            f"- **Proposed Formulation:** {comp.get('proposed_state', 'N/A')}\n"
            f"- **Identified Literature State:** {comp.get('prior_art_state', 'N/A')}\n"
            f"- **Overlap Verdict:** `{comp.get('overlap_verdict', 'N/A')}`\n"
            f"- **Traceable Evidence:** *{comp.get('supporting_evidence', 'N/A')}*\n"
        )
    comp_matrix = "\n".join(comp_rows)

    # Gap table
    gap_rows = []
    for g in result.gap_analysis:
        badge = "🟢 `POTENTIAL GAP (Inferred)`" if "POTENTIAL GAP" in g.classification else "🔴 `VERIFIED OVERLAP`"
        cites = ", ".join(g.supporting_citations[:2]) if g.supporting_citations else "Empirical Analysis"
        gap_rows.append(
            f"| **{g.dimension}** | {badge} | {g.proposed_aspect} | {g.rationale} | {cites} |"
        )
    gap_table = "\n".join(gap_rows)

    # Distinctive contributions list
    contrib_items = "\n".join([f"- **{i+1}.** {c}" for i, c in enumerate(result.distinctive_contributions)])

    # Limitations & recommendations list
    limits_items = "\n".join([f"- {l}" for l in result.limitations_and_recommendations])

    report = f"""# Scientific Research Novelty Assessment Report
**Project Title:** {prop.title}  
**Date of Assessment:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Mode:** {'Multi-Source Live Literature Search' if result.is_live_retrieval else 'Curated Benchmark Mode'}  
**Synthesis Model:** {result.llm_source_used}  

---

## 1. Executive Summary & Novelty Metrics

| Metric | Computed Value | Description |
| :--- | :--- | :--- |
| **Potential Novelty Score** | **{result.overall_novelty_score} / 100** | Documented heuristic originality assessment |
| **Novelty Classification** | **{result.novelty_tier}** | Categorization based on literature proximity |
| **Evidence Coverage & Confidence** | **{result.confidence_coverage_score}%** | Indexing depth and abstract availability index |
| **Max Semantic Proximity** | **{result.max_similarity:.1%}** | Proximity to single closest identified paper |
| **Top-5 Mean Proximity** | **{result.mean_similarity:.1%}** | Density of immediate academic neighborhood |
| **Citation Orthogonality** | **{c_metrics.citation_orthogonality_score:.2f} / 1.0** | Network bridge indicator across subfields |

> **Executive Synthesis:**  
> {result.executive_summary}

---

## 2. Multi-Source Literature Retrieval Statistics

- **OpenAlex Works Retrieved:** {oa_c}
- **Semantic Scholar Papers:** {s2_c}
- **Crossref Works:** {cr_c}
- **Unique Papers (Post-Deduplication):** {dedup_c}
- **Papers Analyzed:** {len(p_papers)}

---

## 3. Documented Mathematical Scoring Breakdown

The Potential Novelty Score ($S$) is calculated using the bounded equation:

$$\\text{{Novelty Score}} = 100 \\times \\left[ 0.35 \\cdot (1 - \\text{{MaxSim}}) + 0.20 \\cdot (1 - \\text{{MeanTopKSim}}) + 0.30 \\cdot \\text{{GapScore}} + 0.15 \\cdot \\text{{CitationScore}} \\right]$$

### Component Factor Breakdown:
- **Semantic Divergence ($w_s = 0.35, w_m = 0.20$):** Contributes **{result.semantic_penalty_score} / 55.0 pts**
- **Research Gap Identification ($w_g = 0.30$):** Contributes **{result.gap_bonus_score} / 30.0 pts**
- **Citation Graph Orthogonality ($w_c = 0.15$):** Contributes **{result.citation_bonus_score} / 15.0 pts**
- **Total Composite Novelty Score:** **{result.overall_novelty_score} / 100.0 pts**

---

## 4. Retrieved Literature & Semantic Similarity Matrix

Total candidate papers analyzed: **{len(p_papers)}**  

| Rank | Title | Authors | Year | Similarity | Citations | Source | Overlapping Terms |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
{papers_table}

---

## 5. RAG Evidence Grounding & Comparative Analysis

{comp_matrix}

---

## 6. Multi-Dimensional Research Gap Analysis

| Dimension | Classification | Proposed Scope | Evidence Rationale | Benchmark References |
| :--- | :---: | :--- | :--- | :--- |
{gap_table}

---

## 7. Distinctive Contributions Claimed
{contrib_items}

---

## 8. Citation Graph Topology Metrics
- **Sub-network Nodes:** {c_metrics.total_nodes}
- **Sub-network Edges:** {c_metrics.total_edges}
- **Verified Citation Links:** {getattr(c_metrics, 'verified_citation_edges', 0)}
- **Graph Density:** {c_metrics.graph_density}
- **Cluster Modularity Count:** {c_metrics.modularity_cluster_count}
- **Citation Coverage Note:** {getattr(c_metrics, 'citation_coverage_note', '')}

---

## 9. Limitations & Recommendations for Peer Review
{limits_items}

> **Disclaimer**: This assessment report is an automated decision-support tool generated via semantic similarity, retrieval-augmented grounding, and citation network heuristics. It does not constitute formal legal verification of patentability or academic priority. Authors are advised to conduct exhaustive domain-specific manual literature searches.
"""
    return report


def export_json_report(result: NoveltyAssessmentResult) -> str:
    """Exports structured assessment result as JSON."""
    return result.model_dump_json(indent=2)
