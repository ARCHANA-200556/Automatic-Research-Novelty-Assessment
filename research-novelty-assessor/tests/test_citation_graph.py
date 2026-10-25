"""
Unit tests for citation graph analysis and visualization.
"""

import pytest
import networkx as nx
from app.config import ResearchInput, PaperMetadata, SimilarityItem
from app.citation_graph import build_citation_network, create_plotly_network_figure


def test_build_citation_network():
    proposal = ResearchInput(
        title="Novel Drug Discovery using Graph Transformers",
        abstract="Graph transformers for kinase inhibitor discovery."
    )

    ranked_papers = [
        SimilarityItem(
            paper=PaperMetadata(
                id="W100",
                title="Molecular Graph Representation",
                citation_count=150,
                referenced_works=["W200"]
            ),
            similarity_score=0.72,
            rank=1
        ),
        SimilarityItem(
            paper=PaperMetadata(
                id="W200",
                title="Foundations of Graph Neural Networks",
                citation_count=1200,
                referenced_works=[]
            ),
            similarity_score=0.60,
            rank=2
        )
    ]

    G, metrics = build_citation_network(proposal, ranked_papers)
    assert isinstance(G, nx.DiGraph)
    assert "PROPOSED_WORK" in G.nodes
    assert "W100" in G.nodes
    assert "W200" in G.nodes
    assert G.has_edge("W100", "W200")  # W100 cites W200
    assert metrics.total_nodes >= 3
    assert 0.0 <= metrics.citation_orthogonality_score <= 1.0


def test_create_plotly_network_figure():
    G = nx.DiGraph()
    G.add_node("P", title="Proposed", node_type="proposed", citations=0, similarity=1.0)
    G.add_node("A", title="Paper A", node_type="retrieved", citations=50, similarity=0.8)
    G.add_edge("P", "A")

    fig = create_plotly_network_figure(G)
    assert fig is not None
    assert len(fig.data) >= 2  # edge_traces and node_trace
