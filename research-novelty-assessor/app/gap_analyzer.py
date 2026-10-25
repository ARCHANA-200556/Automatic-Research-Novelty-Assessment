"""
Dynamic Multi-Dimensional Research Gap Analyzer.
Analyzes differences between proposed research and retrieved literature across 5 core dimensions:
1. Research Objectives
2. Methodology & Architecture
3. Datasets & Evaluation Scale
4. Target Domain & Applications
5. Core Contributions

Zero hardcoded domain assumptions - dynamically adapts to ANY scientific discipline.
Clearly distinguishes [POTENTIAL GAP (Inferred)] from [VERIFIED OVERLAP].
"""

import re
from typing import List, Tuple, Dict, Any, Set
from app.config import ResearchInput, SimilarityItem, GapDimensionItem
from app.rag_engine import extract_salient_technical_terms


def analyze_multidimensional_research_gaps(
    proposal: ResearchInput,
    ranked_papers: List[SimilarityItem]
) -> Tuple[List[GapDimensionItem], float]:
    """
    Dynamically identifies potential research gaps across 5 dimensions
    based on the user's actual research proposal and all retrieved candidate literature.
    Returns: (list_of_gap_items, composite_gap_score [0.0 - 1.0])
    """
    gaps: List[GapDimensionItem] = []
    
    top_papers = ranked_papers[:min(5, len(ranked_papers))] if ranked_papers else []
    max_sim = ranked_papers[0].similarity_score if ranked_papers else 0.0
    
    closest_p = ranked_papers[0].paper if ranked_papers else None
    closest_title = closest_p.title if closest_p else "Identified Prior Art"
    closest_author = (closest_p.authors[0] if closest_p and closest_p.authors else "Prior researchers")
    closest_year = closest_p.year if closest_p and closest_p.year else "recent work"

    prop_full_text = f"{proposal.title} {proposal.abstract} " + " ".join(proposal.objectives)
    prop_tech_terms = extract_salient_technical_terms(prop_full_text, max_terms=8)
    
    all_prior_text = " ".join([p.paper.title + " " + p.paper.abstract for p in top_papers])
    prior_tech_terms = extract_salient_technical_terms(all_prior_text, max_terms=8)

    # --- 1. Research Objectives Dimension ---
    prop_aim = proposal.objectives[0] if proposal.objectives else proposal.title
    if max_sim > 0.70:
        gaps.append(GapDimensionItem(
            dimension="Research Objectives",
            proposed_aspect=f"Proposed primary objective: '{prop_aim[:120]}...'",
            prior_art_state=f"Identified close prior art ({closest_author} et al., {closest_year}) directly investigates a highly convergent research question.",
            classification="VERIFIED OVERLAP",
            rationale=f"High semantic proximity ({max_sim:.1%}) to '{closest_title}'. The core research question has been substantially explored.",
            confidence=0.88,
            supporting_citations=[p.paper.title for p in top_papers[:2]]
        ))
    elif max_sim > 0.45:
        gaps.append(GapDimensionItem(
            dimension="Research Objectives",
            proposed_aspect=f"Specific objective formulation: '{prop_aim[:120]}...'",
            prior_art_state=f"Literature in this domain ({closest_author} et al.) addresses related themes, but with differing operational scope.",
            classification="POTENTIAL GAP (Inferred)",
            rationale=f"Moderate similarity ({max_sim:.1%}) suggests the proposed research extends the inquiry into an underexplored niche.",
            confidence=0.76,
            supporting_citations=[p.paper.title for p in top_papers[:2]]
        ))
    else:
        gaps.append(GapDimensionItem(
            dimension="Research Objectives",
            proposed_aspect=f"Distinct problem focus: '{prop_aim[:120]}...'",
            prior_art_state="Retrieved literature exhibits low thematic convergence with the proposed problem statement.",
            classification="POTENTIAL GAP (Inferred)",
            rationale=f"Low semantic similarity ({max_sim:.1%}) indicates the problem occupies an uncrowded space in the retrieved corpus.",
            confidence=0.82,
            supporting_citations=[]
        ))

    # --- 2. Methodology & Architecture Dimension ---
    unique_methods = [t for t in prop_tech_terms if t.lower() not in [p.lower() for p in prior_tech_terms]]
    if unique_methods and len(unique_methods) >= 2:
        gaps.append(GapDimensionItem(
            dimension="Methodology & Architecture",
            proposed_aspect=f"Methodological formulation incorporating: {', '.join(unique_methods[:3])}.",
            prior_art_state=f"Retrieved literature primarily relies on alternative techniques: {', '.join(prior_tech_terms[:3]) if prior_tech_terms else 'standard domain approaches'}.",
            classification="POTENTIAL GAP (Inferred)",
            rationale="Identified distinct methodological keywords not prominently featured in the top retrieved prior publications.",
            confidence=0.82,
            supporting_citations=[p.paper.title for p in top_papers[:2]]
        ))
    else:
        gaps.append(GapDimensionItem(
            dimension="Methodology & Architecture",
            proposed_aspect=f"Algorithmic and procedural techniques ({', '.join(prop_tech_terms[:3]) if prop_tech_terms else 'standard methods'}).",
            prior_art_state=f"Core technical mechanisms share established precedents with existing literature ({', '.join(prior_tech_terms[:3]) if prior_tech_terms else 'prior art'}).",
            classification="VERIFIED OVERLAP",
            rationale="Substantial overlap between proposed methodology primitives and techniques utilized in retrieved literature.",
            confidence=0.78,
            supporting_citations=[p.paper.title for p in top_papers[:2]]
        ))

    # --- 3. Datasets & Evaluation Scale Dimension ---
    prop_eval_words = set(re.findall(r'(?i)\b(?:dataset|benchmark|cohort|trial|experimental|empirical|clinical|simulation|in vitro|in vivo|in silico|field trial|scale)\b', prop_full_text))
    prior_eval_words = set(re.findall(r'(?i)\b(?:dataset|benchmark|cohort|trial|experimental|empirical|clinical|simulation|in vitro|in vivo|in silico|field trial|scale)\b', all_prior_text))
    
    unique_eval = prop_eval_words - prior_eval_words
    if unique_eval:
        gaps.append(GapDimensionItem(
            dimension="Datasets & Evaluation Scale",
            proposed_aspect=f"Evaluation targeting specific empirical contexts: {', '.join(unique_eval)}.",
            prior_art_state=f"Existing studies emphasize alternative evaluation regimes: {', '.join(prior_eval_words) if prior_eval_words else 'standard benchmarks'}.",
            classification="POTENTIAL GAP (Inferred)",
            rationale="Proposed evaluation regime introduces distinct empirical conditions not observed in retrieved literature.",
            confidence=0.78,
            supporting_citations=[p.paper.title for p in top_papers[:1]]
        ))
    else:
        gaps.append(GapDimensionItem(
            dimension="Datasets & Evaluation Scale",
            proposed_aspect="Benchmark and evaluation protocols.",
            prior_art_state="Prior literature utilizes congruent testing datasets and validation baselines.",
            classification="VERIFIED OVERLAP",
            rationale="Evaluation baselines and data paradigms correspond closely with accepted protocols in this topic area.",
            confidence=0.75,
            supporting_citations=[p.paper.title for p in top_papers[:2]]
        ))

    # --- 4. Target Domain & Application Dimension ---
    # Extract domain nouns from proposal title
    title_tokens = [w for w in re.findall(r'[a-zA-Z]{4,}', proposal.title) if w.lower() not in {'using', 'based', 'with', 'from', 'study', 'novel', 'approach'}]
    domain_topic = " ".join(title_tokens[:3]) if title_tokens else "specialized application domain"
    
    if max_sim < 0.60:
        gaps.append(GapDimensionItem(
            dimension="Target Domain & Applications",
            proposed_aspect=f"Application to: {domain_topic}.",
            prior_art_state=f"Retrieved publications address adjacent disciplines or generic theoretical foundations.",
            classification="POTENTIAL GAP (Inferred)",
            rationale=f"Application of proposed techniques to this specific target domain appears underexplored in retrieved literature.",
            confidence=0.75,
            supporting_citations=[]
        ))
    else:
        gaps.append(GapDimensionItem(
            dimension="Target Domain & Applications",
            proposed_aspect=f"Target domain application: {domain_topic}.",
            prior_art_state=f"Literature has actively targeted this domain (e.g., '{closest_title[:60]}...').",
            classification="VERIFIED OVERLAP",
            rationale="Target application context is an established area of active inquiry in the retrieved corpus.",
            confidence=0.82,
            supporting_citations=[p.paper.title for p in top_papers[:2]]
        ))

    # --- 5. Core Contributions Dimension ---
    gaps.append(GapDimensionItem(
        dimension="Core Contributions Claimed",
        proposed_aspect=f"Synthesis of stated objectives: {proposal.objectives[0] if proposal.objectives else proposal.title}.",
        prior_art_state=f"Prior works provide foundational components ({closest_author} et al.), but may lack this exact multi-stage integration.",
        classification="POTENTIAL GAP (Inferred)",
        rationale="Specific synthesis and combination of objectives represents an exploratory contribution pending peer review verification.",
        confidence=0.78,
        supporting_citations=[]
    ))

    # Calculate overall Gap Score (0.0 to 1.0)
    potential_count = sum(1 for g in gaps if "POTENTIAL GAP" in g.classification)
    gap_score = round(potential_count / max(1, len(gaps)), 3)

    return gaps, gap_score
