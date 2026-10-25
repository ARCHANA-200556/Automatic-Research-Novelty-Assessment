"""
Citation Graph Analysis and Interactive Network Visualization Module.
Constructs NetworkX citation graphs strictly from verified reference relationships
(OpenAlex referenced_works, Semantic Scholar references) and computes
centrality, graph density, clustering, and structural bridge orthogonality.
Renders interactive Plotly graph diagrams with clear coverage indicators.
"""

import math
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
import networkx as nx
import plotly.graph_objects as go
from app.config import ResearchInput, SimilarityItem, CitationGraphMetrics, PaperMetadata

logger = logging.getLogger(__name__)


def build_citation_network(
    proposal: ResearchInput,
    ranked_papers: List[SimilarityItem]
) -> Tuple[nx.DiGraph, CitationGraphMetrics]:
    """
    Constructs a directed NetworkX graph representing citation flows
    between the proposed work, retrieved literature, and referenced landmark papers.
    Builds citation links only when real citation metadata is available.
    """
    G = nx.DiGraph()

    # 1. Add Proposed Research node
    prop_node_id = "PROPOSED_WORK"
    G.add_node(
        prop_node_id,
        title=proposal.title,
        label="★ Proposed Work",
        node_type="proposed",
        year=2026,
        citations=0,
        similarity=1.0,
        authors=["Proposed Project"],
        source="User Proposal",
        size=36
    )

    paper_lookup: Dict[str, PaperMetadata] = {}
    papers_with_refs = 0
    total_papers = len(ranked_papers)

    for item in ranked_papers:
        p = item.paper
        paper_lookup[p.id] = p
        if p.doi:
            paper_lookup[p.doi] = p
        if p.referenced_works:
            papers_with_refs += 1

    # 2. Add retrieved paper nodes
    for item in ranked_papers:
        p = item.paper
        short_title = (p.title[:28] + "...") if len(p.title) > 28 else p.title
        G.add_node(
            p.id,
            title=p.title,
            label=short_title,
            node_type="retrieved",
            year=p.year or 2023,
            citations=p.citation_count,
            similarity=item.similarity_score,
            authors=p.authors[:2],
            source=p.source_api,
            size=min(42, max(18, int(16 + math.log(max(1, p.citation_count) + 1) * 2.8)))
        )

    # 3. Add VERIFIED citation edges between retrieved papers
    verified_citation_count = 0
    for item in ranked_papers:
        p = item.paper
        for ref_id in p.referenced_works:
            # Check if referenced work is in our retrieved set
            target_paper = paper_lookup.get(ref_id)
            if target_paper and target_paper.id != p.id:
                G.add_edge(p.id, target_paper.id, relationship="verified_citation", label="cites")
                verified_citation_count += 1
            elif len(G.nodes) < 35 and len(ref_id) > 2:
                # Add influential external landmark reference if cited by our papers
                landmark_id = f"REF_{ref_id[:12]}"
                if landmark_id not in G:
                    G.add_node(
                        landmark_id,
                        title=f"Referenced Authority ({ref_id})",
                        label=landmark_id,
                        node_type="landmark",
                        year=2020,
                        citations=100,
                        similarity=0.25,
                        authors=["Landmark Authority"],
                        source="Cross-reference",
                        size=14
                    )
                G.add_edge(p.id, landmark_id, relationship="verified_citation", label="cites")
                verified_citation_count += 1

    # 4. Connect Proposed Work to top related prior art papers (conceptual prior-art links)
    for item in ranked_papers[:min(5, len(ranked_papers))]:
        G.add_edge(prop_node_id, item.paper.id, relationship="conceptual_prior_art", label="synthesizes")

    # 5. Compute network metrics
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    # In-degree (who is cited most in this subgraph)
    in_degrees = dict(G.in_degree())
    max_in_deg = max(in_degrees.values()) if in_degrees else 0
    most_central_id = max(in_degrees, key=in_degrees.get) if in_degrees else None
    most_central_title = G.nodes[most_central_id].get("title", "") if most_central_id else "N/A"

    prop_in = G.in_degree(prop_node_id) if prop_node_id in G else 0
    prop_out = G.out_degree(prop_node_id) if prop_node_id in G else 0

    density = round(nx.density(G), 4)

    # Connected components
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    cluster_count = len(components)

    # Citation Orthogonality Score
    if cluster_count > 1 and prop_out > 1:
        bridged_components = 0
        for comp in components:
            if any(neighbor in comp for neighbor in G.neighbors(prop_node_id)):
                bridged_components += 1
        orthogonality = round(min(1.0, 0.45 + (bridged_components * 0.18)), 3)
    else:
        orthogonality = round(max(0.25, 1.0 - (density * 2.2)), 3)

    if total_papers > 0:
        coverage_pct = round((papers_with_refs / total_papers) * 100, 1)
        cov_note = f"Verified citation metadata available for {papers_with_refs} of {total_papers} papers ({coverage_pct}%). Graph displays verified citation flows and top conceptual prior-art connections."
    else:
        cov_note = "No papers available for citation analysis."

    metrics = CitationGraphMetrics(
        total_nodes=total_nodes,
        total_edges=total_edges,
        verified_citation_edges=verified_citation_count,
        proposed_in_degree=prop_in,
        proposed_out_degree=prop_out,
        max_in_degree=max_in_deg,
        most_central_paper=most_central_title,
        graph_density=density,
        modularity_cluster_count=cluster_count,
        citation_orthogonality_score=orthogonality,
        citation_coverage_note=cov_note
    )

    return G, metrics


def create_plotly_network_figure(G: nx.DiGraph) -> go.Figure:
    """Generates an interactive 2D Plotly network visualization."""
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No citation network data available", showarrow=False)
        return fig

    # Compute spring layout
    pos = nx.spring_layout(G, k=0.65, iterations=60, seed=42)

    # Prepare edge traces: separate into verified citations and conceptual links
    edge_x_verified = []
    edge_y_verified = []
    edge_x_conceptual = []
    edge_y_conceptual = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        rel = data.get("relationship", "verified_citation")
        if rel == "verified_citation":
            edge_x_verified.extend([x0, x1, None])
            edge_y_verified.extend([y0, y1, None])
        else:
            edge_x_conceptual.extend([x0, x1, None])
            edge_y_conceptual.extend([y0, y1, None])

    traces = []
    if edge_x_verified:
        traces.append(go.Scatter(
            x=edge_x_verified, y=edge_y_verified,
            line=dict(width=1.8, color='#64748B'),
            hoverinfo='none',
            mode='lines',
            name="Verified Citation Flow"
        ))

    if edge_x_conceptual:
        traces.append(go.Scatter(
            x=edge_x_conceptual, y=edge_y_conceptual,
            line=dict(width=1.5, color='#93C5FD', dash='dot'),
            hoverinfo='none',
            mode='lines',
            name="Conceptual Prior-Art Link"
        ))

    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_symbols = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        attrs = G.nodes[node]
        n_type = attrs.get("node_type", "retrieved")
        title = attrs.get("title", node)
        cites = attrs.get("citations", 0)
        sim = attrs.get("similarity", 0.0)
        year = attrs.get("year", "N/A")
        authors = ", ".join(attrs.get("authors", []))
        source = attrs.get("source", "Academic Index")

        if n_type == "proposed":
            node_color.append('#2563EB')
            node_size.append(38)
            node_symbols.append('diamond')
            tooltip = f"<b>★ PROPOSED WORK</b><br>Title: {title}<br>Status: Evaluated Proposal"
        elif n_type == "landmark":
            node_color.append('#94A3B8')
            node_size.append(16)
            node_symbols.append('circle')
            tooltip = f"<b>Landmark Referenced Authority</b><br>ID: {node}"
        else:
            if sim > 0.65:
                node_color.append('#EF4444')  # Red (High similarity)
            elif sim > 0.40:
                node_color.append('#F59E0B')  # Amber (Moderate)
            else:
                node_color.append('#10B981')  # Emerald (Low/Distinct)

            node_size.append(max(20, min(42, int(18 + math.log(max(1, cites) + 1) * 2.8))))
            node_symbols.append('circle')
            tooltip = (
                f"<b>{title}</b><br>"
                f"Authors: {authors or 'N/A'}<br>"
                f"Year: {year} | Source: {source}<br>"
                f"Citations: {cites:,} | Similarity: {sim:.1%}"
            )

        node_text.append(tooltip)

    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[n].get("label", "")[:18] for n in G.nodes()],
        textposition="bottom center",
        textfont=dict(size=10, color="#1E293B", family="sans-serif"),
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            symbol=node_symbols,
            line=dict(width=2, color='#FFFFFF')
        ),
        name="Literature Nodes"
    ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text="<b>Citation & Prior-Art Network Topology</b>",
                font=dict(size=16, color="#0F172A")
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='closest',
            margin=dict(b=20, l=10, r=10, t=55),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(248, 250, 252, 0.8)',
            paper_bgcolor='white',
            height=540
        )
    )

    return fig
