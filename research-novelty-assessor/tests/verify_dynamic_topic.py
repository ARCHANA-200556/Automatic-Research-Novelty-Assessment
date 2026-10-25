"""
Verification script for dynamic multi-source retrieval on an arbitrary scientific topic (Solar Cells).
"""

from app.pipeline import run_full_novelty_assessment
from app.config import ResearchInput

def verify_solar_cell_topic():
    proposal = ResearchInput(
        title="Perovskite Silicon Tandem Solar Cells Degradation and Stability Under Environmental Stress",
        abstract="We analyze halide phase segregation and interfacial recombination losses in monolithic perovskite silicon tandem solar cells under accelerated operational damp heat.",
        keywords=["Perovskite", "Silicon Tandem", "Photovoltaics", "Stability"],
        objectives=[
            "Synthesize self-assembled monolayers for defect passivation in wide-bandgap perovskites.",
            "Measure operational power conversion efficiency retention under 85C damp heat protocols."
        ]
    )

    print("Executing live multi-source retrieval & novelty assessment...")
    result, G, fig = run_full_novelty_assessment(proposal, limit_papers=8)

    print("\n--- RESULTS FOR DYNAMIC SOLAR CELL TOPIC ---")
    print("Score:", result.overall_novelty_score, "/ 100")
    print("Tier:", result.novelty_tier)
    print("Retrieved papers count:", len(result.retrieved_papers))
    print("Sources breakdown:", result.retrieval_stats)
    
    assert len(result.retrieved_papers) > 0, "Should have retrieved literature"
    
    top_p = result.retrieved_papers[0].paper
    print(f"Top Paper: '{top_p.title}'")
    print(f"Authors: {top_p.authors[:2]}")
    print(f"Source API: {top_p.source_api}")
    print(f"DOI: {top_p.doi}")
    print(f"Citations: {top_p.citation_count}")
    print(f"Similarity: {result.retrieved_papers[0].similarity_score:.1%}")
    print(f"Overlapping terms: {result.retrieved_papers[0].key_overlap_terms}")

    print("\n--- VERIFYING GAP ANALYSIS IS TOPIC-SPECIFIC (NO HARDCODING) ---")
    for g in result.gap_analysis:
        print(f"[{g.classification}] {g.dimension}: {g.proposed_aspect[:70]}...")
        assert "scibert" not in g.rationale.lower()
        assert "citegnn" not in g.rationale.lower()
        assert "novelty detection" not in g.proposed_aspect.lower()

    print("\n--- VERIFYING RAG EVIDENCE COMPARISONS ---")
    for comp in result.methodological_comparisons:
        print(f"{comp['aspect']}: {comp['overlap_verdict']}")
        assert "scibert" not in str(comp).lower()
        assert "citegnn" not in str(comp).lower()

    print("\n--- VERIFYING CITATION GRAPH ---")
    print(f"Nodes: {result.citation_metrics.total_nodes}, Edges: {result.citation_metrics.total_edges}")
    print(f"Coverage note: {result.citation_metrics.citation_coverage_note}")

    print("\n[SUCCESS] Dynamic topic test PASSED flawlessly!")

if __name__ == "__main__":
    verify_solar_cell_topic()
