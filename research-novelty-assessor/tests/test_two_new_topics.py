"""
Automated validation of dynamic novelty assessment on TWO COMPLETELY NEW research topics
not included in the example dropdown:
1. Quantum Key Distribution for Satellite Optical Communications (Physics / Quantum Tech)
2. Deep Learning for Early Alzheimer's Detection via Multi-Modal MRI and PET (Neurology / Medical Imaging)
"""

import pytest
from app.config import ResearchInput
from app.input_parser import generate_academic_search_queries
from app.pipeline import run_full_novelty_assessment


def test_topic_1_quantum_key_distribution():
    """Validates dynamic search and assessment for Satellite Quantum Key Distribution."""
    proposal_quantum = ResearchInput(
        title="Quantum Key Distribution Protocols for Satellite-to-Ground Optical Communications",
        abstract="We evaluate decoy-state BB84 quantum key distribution performance over atmospheric turbulence channels in low-Earth orbit satellite downlinks, optimizing adaptive optics wavefront correction.",
        keywords=["Quantum Key Distribution", "QKD", "Satellite Communication", "Free-Space Optics", "Quantum Cryptography"],
        objectives=[
            "Quantify asymptotic secret key rates under atmospheric turbulence channels.",
            "Model adaptive optics wavefront correction for low-Earth orbit satellite downlinks."
        ]
    )

    # 1. Verify dynamic query generation
    queries = generate_academic_search_queries(proposal_quantum)
    print("\n[Topic 1 - Quantum] Generated Queries:", queries)
    assert len(queries) >= 2
    assert any("quantum" in q.lower() for q in queries)
    assert any("satellite" in q.lower() or "qkd" in q.lower() for q in queries)
    # Ensure zero leak from presets
    assert not any("novelty assessment" in q.lower() for q in queries)
    assert not any("drug repurposing" in q.lower() for q in queries)

    # 2. Run full novelty assessment
    result, G, fig = run_full_novelty_assessment(proposal_quantum, limit_papers=6)
    
    print(f"[Topic 1 - Quantum] Overall Score: {result.overall_novelty_score} ({result.novelty_tier})")
    print(f"[Topic 1 - Quantum] Papers Retrieved: {len(result.retrieved_papers)}")
    print(f"[Topic 1 - Quantum] Sources Breakdown: {result.retrieval_stats}")
    
    assert len(result.retrieved_papers) > 0, "Must retrieve real academic literature for Quantum topic"
    
    # 3. Verify retrieved papers match Quantum domain
    top_p = result.retrieved_papers[0].paper
    print(f"[Topic 1 - Quantum] Top Paper Title: '{top_p.title}' (Source: {top_p.source_api}, Citations: {top_p.citation_count})")
    assert any(term in (top_p.title + " " + top_p.abstract).lower() for term in ["quantum", "key", "satellite", "optical", "communication", "channel"])
    
    # 4. Verify gap analysis is specific to Quantum domain
    for g in result.gap_analysis:
        assert "scibert" not in g.rationale.lower()
        assert "citegnn" not in g.rationale.lower()
        assert "novelty assessment" not in g.proposed_aspect.lower()


def test_topic_2_alzheimers_neuroimaging():
    """Validates dynamic search and assessment for Alzheimer's Multi-Modal Neuroimaging."""
    proposal_alzheimers = ResearchInput(
        title="Deep Learning for Early Detection of Alzheimer's Disease Using Multi-Modal MRI and PET Neuroimaging",
        abstract="We formulate a multi-modal 3D convolutional neural network combining structural T1-weighted MRI and amyloid PET neuroimaging for predicting mild cognitive impairment conversion to Alzheimer's dementia.",
        keywords=["Alzheimer's Disease", "Multi-Modal Neuroimaging", "Deep Learning", "PET Scan", "Structural MRI", "Biomarkers"],
        objectives=[
            "Fuse structural MRI cortical thickness with fluorodeoxyglucose PET metabolic maps.",
            "Predict clinical progression to Alzheimer's dementia within a 36-month conversion window."
        ]
    )

    # 1. Verify dynamic query generation
    queries = generate_academic_search_queries(proposal_alzheimers)
    print("\n[Topic 2 - Alzheimer's] Generated Queries:", queries)
    assert len(queries) >= 2
    assert any("alzheimer" in q.lower() for q in queries)
    assert any("mri" in q.lower() or "neuroimaging" in q.lower() or "pet" in q.lower() for q in queries)
    # Ensure zero leak from presets or topic 1
    assert not any("quantum" in q.lower() for q in queries)
    assert not any("novelty assessment" in q.lower() for q in queries)

    # 2. Run full novelty assessment
    result, G, fig = run_full_novelty_assessment(proposal_alzheimers, limit_papers=6)
    
    print(f"[Topic 2 - Alzheimer's] Overall Score: {result.overall_novelty_score} ({result.novelty_tier})")
    print(f"[Topic 2 - Alzheimer's] Papers Retrieved: {len(result.retrieved_papers)}")
    print(f"[Topic 2 - Alzheimer's] Sources Breakdown: {result.retrieval_stats}")
    
    assert len(result.retrieved_papers) > 0, "Must retrieve real academic literature for Alzheimer's topic"
    
    # 3. Verify retrieved papers match Alzheimer's / Neuroimaging domain
    top_p = result.retrieved_papers[0].paper
    print(f"[Topic 2 - Alzheimer's] Top Paper Title: '{top_p.title}' (Source: {top_p.source_api}, Citations: {top_p.citation_count})")
    assert any(term in (top_p.title + " " + top_p.abstract).lower() for term in ["alzheimer", "mri", "pet", "neuroimaging", "dementia", "cognitive", "brain"])

    # 4. Verify gap analysis is specific to Alzheimer's domain
    for g in result.gap_analysis:
        assert "scibert" not in g.rationale.lower()
        assert "citegnn" not in g.rationale.lower()
        assert "quantum" not in g.proposed_aspect.lower()
        assert "novelty assessment" not in g.proposed_aspect.lower()


if __name__ == "__main__":
    test_topic_1_quantum_key_distribution()
    test_topic_2_alzheimers_neuroimaging()
    print("\n[SUCCESS] Both dynamic topics passed end-to-end verification flawlessly!")
