"""
Streamlit Web Application: Automatic Research Novelty Assessment.
Interactive dashboard with dynamic multi-source retrieval (OpenAlex, Semantic Scholar, Crossref),
sentence-transformers embeddings, RAG evidence grounding, NetworkX citation graphs, and report export.
Supports ANY research domain or topic.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time

from app.config import ResearchInput, NoveltyAssessmentResult
from app.input_parser import parse_pdf_document, build_manual_research_input, load_presets
from app.pipeline import run_full_novelty_assessment
from app.report_generator import export_markdown_report, export_json_report

# Page layout configuration
st.set_page_config(
    page_title="Research Novelty Assessor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .badge-live {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fallback {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stat-pill {
        display: inline-block;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes session state variables without forcing presets."""
    if "assessment_result" not in st.session_state:
        st.session_state.assessment_result = None
    if "citation_graph" not in st.session_state:
        st.session_state.citation_graph = None
    if "network_fig" not in st.session_state:
        st.session_state.network_fig = None
    # Initialize with clean empty inputs ready for any topic
    if "form_title" not in st.session_state:
        st.session_state["form_title"] = ""
    if "form_abstract" not in st.session_state:
        st.session_state["form_abstract"] = ""
    if "form_keywords" not in st.session_state:
        st.session_state["form_keywords"] = ""
    if "form_objectives" not in st.session_state:
        st.session_state["form_objectives"] = ""
    if "input_mode_selector" not in st.session_state:
        st.session_state["input_mode_selector"] = "✍️ Enter Your Own Research Topic (Default - Any Domain)"


init_session_state()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=56)
    st.markdown("### 🔬 Novelty Assessor")
    st.caption("Evidence-Grounded Scientific Literature Evaluation")
    
    st.divider()
    st.markdown("#### ⚙️ Literature Search Depth")
    max_papers = st.slider(
        "Target Literature Depth:",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="Target number of unique peer-reviewed publications to retrieve and analyze across OpenAlex, Semantic Scholar, and Crossref."
    )

    st.markdown("#### 🌐 Live Academic Sources")
    st.markdown("- 🟢 **OpenAlex API**: Connected")
    st.markdown("- 🟢 **Semantic Scholar API**: Connected")
    st.markdown("- 🟢 **Crossref Works API**: Connected")
    
    st.divider()
    with st.expander("💡 Optional: Load an Example Proposal", expanded=False):
        st.caption("Autofill the form with a sample topic to test the UI. You can freely edit any field or clear the form.")
        presets = load_presets()
        preset_names = ["-- Select a Sample Proposal --"] + [f"{p['title'][:36]}..." for p in presets]
        selected_preset_idx = st.selectbox(
            "Choose sample topic:",
            range(len(preset_names)),
            format_func=lambda i: preset_names[i],
            key="sidebar_preset_select"
        )
        
        if selected_preset_idx > 0:
            if st.button(
                "📋 Autofill Form with Selected Example",
                key="btn_autofill_sidebar"
            ):
                chosen = presets[selected_preset_idx - 1]

                st.session_state["form_title"] = chosen["title"]
                st.session_state["form_abstract"] = chosen["abstract"]
                st.session_state["form_keywords"] = chosen["keywords"]
                st.session_state["form_objectives"] = chosen["objectives"]

                st.success(
                    "Form autofilled! You can edit any field "
                    "or run the assessment."
                )
                st.rerun()

    force_fallback = st.checkbox(
        "Force Offline Benchmark Cache",
        value=False,
        help="Use cached offline peer-reviewed papers for benchmark testing when no internet connection is available."
    )

    st.divider()
    st.markdown("#### 📐 Mathematical Formula")
    st.latex(r"S = 100 \times [0.35(1 - \text{MaxSim}) + 0.20(1 - \text{Mean}) + 0.30\text{Gap} + 0.15\text{Cit}]")
    st.caption("Bounded within [0, 100]. Penalty for high semantic overlap; bonus for grounded gaps and citation bridge orthogonality.")


# --- MAIN HEADER ---
st.markdown("<div class='main-title'>Automatic Research Novelty Assessment</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Evidence-Grounded Novelty Evaluation for ANY Research Topic Using "
    "Multi-Source Academic APIs (OpenAlex, Semantic Scholar, Crossref), Dense Vector Embeddings, RAG, and Citation Graphs</div>",
    unsafe_allow_html=True
)

# Tabs
tab_input, tab_score, tab_literature, tab_rag, tab_gaps, tab_graph, tab_report = st.tabs([
    "📝 1. Research Input",
    "📊 2. Novelty Scorecard",
    "📚 3. Literature Similarity",
    "🔍 4. RAG Evidence",
    "🎯 5. Gap & Contributions",
    "🕸️ 6. Citation Network",
    "📄 7. Report & Export"
])


# ==========================================
# TAB 1: RESEARCH INPUT & INGESTION
# ==========================================
with tab_input:
    st.markdown("### 📥 Submit Research Work for Assessment")

    input_mode = st.radio(
        "Choose Input Method:",
        [
            "✍️ Enter Your Own Research Topic (Default - Any Domain)",
            "📄 Upload Research Paper (PDF)",
            "💡 Browse Example Presets"
        ],
        horizontal=True,
        key="input_mode_selector",
        help="The application accepts any research topic from any academic discipline."
    )

    if input_mode == "✍️ Enter Your Own Research Topic (Default - Any Domain)":
        st.info("💡 **Enter any research topic below.** You can type any research title, abstract, keywords, and objectives from any field (Medicine, AI, Renewable Energy, Physics, Robotics, Materials Science, Humanities, etc.). Real literature will be dynamically retrieved from OpenAlex, Semantic Scholar, and Crossref.")
        
        col_rst, col_sp = st.columns([1, 4])
        with col_rst:
            if st.button("🧹 Clear Form / Reset"):
                st.session_state["form_title"] = ""
                st.session_state["form_abstract"] = ""
                st.session_state["form_keywords"] = ""
                st.session_state["form_objectives"] = ""
                st.session_state["assessment_result"] = None
                st.rerun()

    elif input_mode == "📄 Upload Research Paper (PDF)":
        st.info("📄 Upload an existing research paper PDF. Section headers (Title, Abstract, Keywords, Objectives) will be parsed and populated into the editable fields below for your review.")
        uploaded_pdf = st.file_uploader("Upload Research Paper PDF", type=["pdf"], help="Extracts Title, Abstract, Keywords, and Objectives using pypdf.")
        if uploaded_pdf is not None:
            with st.spinner("Extracting text and identifying paper sections..."):
                parsed_proposal, status_msg = parse_pdf_document(uploaded_pdf)
                if parsed_proposal:
                    st.success(status_msg)
                    st.session_state["form_title"] = parsed_proposal.title
                    st.session_state["form_abstract"] = parsed_proposal.abstract
                    st.session_state["form_keywords"] = ", ".join(parsed_proposal.keywords)
                    st.session_state["form_objectives"] = "\n".join(parsed_proposal.objectives)
                else:
                    st.error(status_msg)

    elif input_mode == "💡 Browse Example Presets":
        st.info("💡 Selecting an example will populate the editable form with sample text. You can freely edit any field before running the assessment.")
        presets = load_presets()
        preset_names = [f"Example {i+1}: {p['title']}" for i, p in enumerate(presets)]
        selected_idx = st.selectbox("Select a sample proposal:", range(len(preset_names)), format_func=lambda i: preset_names[i])
        if selected_idx is not None:
            chosen = presets[selected_idx]
            st.markdown(f"**Title:** {chosen['title']}")
            st.markdown(f"**Keywords:** {chosen['keywords']}")
            st.caption(f"**Abstract:** {chosen['abstract'][:250]}...")
            if st.button(
                "📥 Copy this Example into Editable Form",
                key="btn_load_preset"
            ):
                st.session_state["form_title"] = chosen["title"]
                st.session_state["form_abstract"] = chosen["abstract"]
                st.session_state["form_keywords"] = chosen["keywords"]
                st.session_state["form_objectives"] = chosen["objectives"]

                st.success(
                    "Example copied into the editable form! "
                    "Review your inputs and run the assessment."
                )
                st.rerun()

    # Editable form fields - always visible and editable!
    st.markdown("---")
    st.markdown("#### 📝 Research Proposal Details (Editable)")

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        in_title = st.text_input(
            "Research Title *",
            value=st.session_state.get("form_title", ""),
            placeholder="e.g. Deep Reinforcement Learning for Autonomous Drone Swarm Navigation in GPS-Denied Environments",
            key="form_title",
            help="Enter the specific title of the proposed research."
        )
    with col_t2:
        in_keywords = st.text_input(
            "Keywords (comma-separated)",
            value=st.session_state.get("form_keywords", ""),
            placeholder="e.g. Drone swarms, Reinforcement learning, Collision avoidance",
            key="form_keywords",
            help="Comma-separated keywords highlighting key concepts."
        )

    in_abstract = st.text_area(
        "Abstract / Research Summary *",
        value=st.session_state.get("form_abstract", ""),
        height=170,
        placeholder="Enter a comprehensive abstract describing the research problem, methodology, target application, and expected contributions...",
        key="form_abstract",
        help="Core summary describing the problem and proposed approach."
    )

    in_objectives = st.text_area(
        "Research Objectives / Hypotheses (1 per line)",
        value=st.session_state.get("form_objectives", ""),
        height=110,
        placeholder="1. Formulate decentralized policy gradient algorithm for inter-agent collision avoidance...\n2. Benchmark against Voronoi-based coordination protocols in high-density obstacle fields...",
        key="form_objectives",
        help="List specific research aims or hypotheses (1 per line)."
    )

    st.markdown("---")
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        start_analysis = st.button("🚀 Run Comprehensive Novelty Assessment", type="primary", use_container_width=True)
    with col_info:
        st.caption(f"Searches OpenAlex, Semantic Scholar, and Crossref for up to {max_papers} unique papers matching the current form inputs, computes sentence-transformers similarity, extracts RAG evidence, and evaluates novelty.")

    if start_analysis:
        if not in_title.strip() or not in_abstract.strip():
            st.error("⚠️ Please provide both a Research Title and Abstract to perform the novelty assessment.")
        else:
            proposal_obj = build_manual_research_input(
                title=in_title,
                abstract=in_abstract,
                keywords_str=in_keywords,
                objectives_str=in_objectives
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current_step, total_steps, message):
                pct = int((current_step / total_steps) * 100)
                progress_bar.progress(pct)
                status_text.markdown(f"**Step {current_step}/{total_steps}:** {message}")

            try:
                result, G, fig_net = run_full_novelty_assessment(
                    proposal=proposal_obj,
                    limit_papers=max_papers,
                    force_fallback=force_fallback,
                    progress_callback=progress_callback
                )
                time.sleep(0.3)
                progress_bar.progress(100)
                status_text.success("Assessment Complete! Review your results in the dashboard tabs.")

                st.session_state.assessment_result = result
                st.session_state.citation_graph = G
                st.session_state.network_fig = fig_net

            except ValueError as ve:
                progress_bar.empty()
                status_text.empty()
                st.error(f"🔍 {str(ve)}")
                st.info("💡 **Tips to get better search results:**\n"
                        "- Ensure your Research Title contains key scientific nouns (e.g., 'Drone swarms', 'CRISPR delivery', 'Perovskite solar').\n"
                        "- Add 2-3 specific, comma-separated keywords.\n"
                        "- Avoid conversational phrasing like 'In this project we plan to look at'.")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error during assessment pipeline: {str(e)}")


# Guard: Check if assessment results are available
res: NoveltyAssessmentResult = st.session_state.assessment_result

if res is None:
    st.info("👈 Enter your research details and click **'🚀 Run Comprehensive Novelty Assessment'** above to evaluate your proposal.")
else:
    # ==========================================
    # TAB 2: NOVELTY SCORE & EXECUTIVE DASHBOARD
    # ==========================================
    with tab_score:
        st.markdown("### 📊 Potential Novelty Assessment Summary")

        # Multi-source retrieval stats banner
        stats = getattr(res, 'retrieval_stats', {})
        if stats:
            oa_c = stats.get('openalex_count', 0)
            s2_c = stats.get('semantic_scholar_count', 0)
            cr_c = stats.get('crossref_count', 0)
            dedup_c = stats.get('unique_deduplicated', len(res.retrieved_papers))
            analyzed_c = stats.get('analyzed_count', len(res.retrieved_papers))
            
            st.markdown(
                f"<div style='margin-bottom:14px;'>"
                f"<span class='stat-pill'>🌐 OpenAlex: <b>{oa_c}</b> papers</span>"
                f"<span class='stat-pill'>🔬 Semantic Scholar: <b>{s2_c}</b> papers</span>"
                f"<span class='stat-pill'>📖 Crossref: <b>{cr_c}</b> papers</span>"
                f"<span class='stat-pill'>🧹 Deduplicated: <b>{dedup_c}</b> unique</span>"
                f"<span class='stat-pill' style='background:#DCFCE7; border-color:#86EFAC;'>⚡ Analyzed: <b>{analyzed_c}</b> papers</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Top Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(
                label="Overall Novelty Score",
                value=f"{res.overall_novelty_score} / 100",
                delta=res.novelty_tier.split(" (")[0],
                delta_color="normal"
            )
        with col_m2:
            st.metric(
                label="Evidence Coverage & Confidence",
                value=f"{res.confidence_coverage_score}%",
                help="Coverage based on number of relevant retrieved papers and abstract completeness across sources."
            )
        with col_m3:
            st.metric(
                label="Max Semantic Proximity",
                value=f"{res.max_similarity:.1%}",
                delta=f"-{res.max_similarity:.1%} Overlap",
                delta_color="inverse"
            )
        with col_m4:
            st.metric(
                label="Top-5 Mean Proximity",
                value=f"{res.mean_similarity:.1%}",
                help="Neighborhood density of top 5 closest papers."
            )

        st.markdown("---")

        # Gauge Chart & Executive Summary
        col_g1, col_g2 = st.columns([1, 1.3])
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res.overall_novelty_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"<b>{res.novelty_tier}</b>", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#2563EB", 'thickness': 0.28},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 40], 'color': '#FEE2E2'},
                        {'range': [40, 60], 'color': '#FEF3C7'},
                        {'range': [60, 80], 'color': '#E0E7FF'},
                        {'range': [80, 100], 'color': '#DCFCE7'}
                    ],
                    'threshold': {
                        'line': {'color': "#1E293B", 'width': 3},
                        'thickness': 0.75,
                        'value': res.overall_novelty_score
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            st.markdown("#### 📑 Executive Assessment Narrative")
            st.markdown(f"> {res.executive_summary}")
            
            badge_html = "<span class='badge-live'>● Multi-Source Live Search</span>" if res.is_live_retrieval else "<span class='badge-fallback'>▲ Offline Benchmark Mode</span>"
            st.markdown(f"**Data Sources:** {badge_html} &nbsp;&nbsp;|&nbsp;&nbsp; **Synthesis Engine:** `{res.llm_source_used}`", unsafe_allow_html=True)
            st.caption("Note: The novelty score is a calculated heuristic indicator of literature proximity and gap coverage. It does not constitute legal proof of patentability or academic priority.")

        st.markdown("---")
        st.markdown("#### 📐 Mathematical Sub-Score Breakdown")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<b>1. Semantic Divergence</b><br>"
                f"<h3 style='color:#2563EB;'>{res.semantic_penalty_score} <span style='font-size:0.9rem; color:#64748B;'>/ 55.0 pts</span></h3>"
                f"<small>Weights: 35% MaxSim + 20% Top-5 Mean<br>Max overlap: {res.max_similarity:.1%}</small>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_f2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<b>2. Research Gap Evidence</b><br>"
                f"<h3 style='color:#16A34A;'>{res.gap_bonus_score} <span style='font-size:0.9rem; color:#64748B;'>/ 30.0 pts</span></h3>"
                f"<small>Weight: 30% Multi-dimensional gap coverage<br>5 dimensions evaluated</small>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_f3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<b>3. Citation Orthogonality</b><br>"
                f"<h3 style='color:#7C3AED;'>{res.citation_bonus_score} <span style='font-size:0.9rem; color:#64748B;'>/ 15.0 pts</span></h3>"
                f"<small>Weight: 15% Network topology<br>Orthogonality: {res.citation_metrics.citation_orthogonality_score:.2f} / 1.0</small>"
                f"</div>",
                unsafe_allow_html=True
            )


    # ==========================================
    # TAB 3: LITERATURE SIMILARITY MATRIX
    # ==========================================
    with tab_literature:
        st.markdown("### 📚 Retrieved Peer-Reviewed Literature & Similarity Scores")
        st.markdown(f"Analyzed **{len(res.retrieved_papers)}** candidate papers retrieved for your topic. Cosine similarities are computed directly using `sentence-transformers/all-MiniLM-L6-v2`.")

        # Similarity Bar Chart for all retrieved papers
        paper_names = [p.paper.title[:38] + "..." for p in res.retrieved_papers]
        scores = [p.similarity_score for p in res.retrieved_papers]
        cites = [p.paper.citation_count for p in res.retrieved_papers]
        sources = [getattr(p.paper, 'source_api', 'Academic Index') for p in res.retrieved_papers]

        df_sim = pd.DataFrame({
            "Paper Title": paper_names,
            "Cosine Similarity": scores,
            "Citations": cites,
            "Source API": sources,
            "Full Title": [p.paper.title for p in res.retrieved_papers]
        })

        fig_bar = px.bar(
            df_sim,
            x="Cosine Similarity",
            y="Paper Title",
            orientation="h",
            color="Cosine Similarity",
            color_continuous_scale="Viridis",
            hover_data=["Full Title", "Citations", "Source API"],
            title="Literature Semantic Proximity Ranking (All Retrieved Papers)"
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=max(360, len(res.retrieved_papers) * 26))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed Table
        st.markdown("#### Literature Details & Source References")
        for item in res.retrieved_papers:
            p = item.paper
            src_tag = getattr(p, 'source_api', 'Academic Index')
            with st.expander(f"#{item.rank} | {p.title} ({p.year or 'Recent'}) — Similarity: {item.similarity_score:.1%} [{src_tag}]"):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
                    st.markdown(f"**Venue:** {p.venue or 'Academic Journal / Conference'}")
                    st.markdown(f"**Source Index:** `{src_tag}`")
                    st.markdown(f"**Abstract:** {p.abstract if p.abstract else 'No abstract indexed.'}")
                with c2:
                    st.metric("Citations", f"{p.citation_count:,}")
                    if p.url:
                        st.markdown(f"[🔗 View Original Paper]({p.url})")
                    if p.doi:
                        st.caption(f"DOI: {p.doi}")
                    if item.key_overlap_terms:
                        st.markdown(f"**Overlapping Terms:**\n`{'`, `'.join(item.key_overlap_terms)}`")


    # ==========================================
    # TAB 4: RAG EVIDENCE COMPARATIVE ANALYSIS
    # ==========================================
    with tab_rag:
        st.markdown("### 🔍 Grounded RAG Evidence Analysis")
        st.caption(f"Synthesized via: {res.llm_source_used} grounded in retrieved academic literature passages.")

        for comp in res.methodological_comparisons:
            st.markdown(f"#### 📌 {comp.get('aspect', 'Comparative Aspect')}")
            col_p, col_a = st.columns(2)
            with col_p:
                st.markdown(
                    f"<div style='background-color:#EFF6FF; border-left:4px solid #3B82F6; padding:12px; border-radius:6px;'>"
                    f"<b>Proposed Approach:</b><br>{comp.get('proposed_state', 'N/A')}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_a:
                st.markdown(
                    f"<div style='background-color:#F8FAFC; border-left:4px solid #94A3B8; padding:12px; border-radius:6px;'>"
                    f"<b>Identified Literature State:</b><br>{comp.get('prior_art_state', 'N/A')}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown(f"**Verdict:** `{comp.get('overlap_verdict', 'N/A')}` &nbsp;|&nbsp; **Evidence Grounding:** *{comp.get('supporting_evidence', 'N/A')}*")
            st.divider()


    # ==========================================
    # TAB 5: GAP & CONTRIBUTIONS BREAKDOWN
    # ==========================================
    with tab_gaps:
        st.markdown("### 🎯 Multi-Dimensional Research Gap Analysis")
        st.markdown("Distinguishes potential unexplored research gaps from verified prior art overlap across 5 key dimensions.")

        for g in res.gap_analysis:
            is_potential = "POTENTIAL GAP" in g.classification
            badge_color = "#16A34A" if is_potential else "#DC2626"
            badge_bg = "#DCFCE7" if is_potential else "#FEE2E2"
            
            st.markdown(
                f"<div style='border:1px solid #E2E8F0; border-radius:8px; padding:14px; margin-bottom:12px; background:white;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='font-size:1.05rem; font-weight:700; color:#1E293B;'>{g.dimension}</span>"
                f"<span style='background-color:{badge_bg}; color:{badge_color}; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.85rem;'>{g.classification}</span>"
                f"</div>"
                f"<p style='margin:8px 0 4px 0;'><b>Proposed Aspect:</b> {g.proposed_aspect}</p>"
                f"<p style='margin:4px 0 4px 0;'><b>Prior Art Context:</b> {g.prior_art_state}</p>"
                f"<p style='margin:4px 0 0 0; color:#475569;'><i>Evidence Rationale:</i> {g.rationale}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### 🌟 Distinctive Contributions Claimed")
        for i, c in enumerate(res.distinctive_contributions):
            st.markdown(f"**{i+1}.** {c}")


    # ==========================================
    # TAB 6: CITATION NETWORK GRAPH
    # ==========================================
    with tab_graph:
        st.markdown("### 🕸️ Citation & Prior-Art Network Topology")
        c_met = res.citation_metrics
        cov_note = getattr(c_met, 'citation_coverage_note', '')
        if cov_note:
            st.caption(f"💡 {cov_note}")

        col_g_metric1, col_g_metric2, col_g_metric3, col_g_metric4 = st.columns(4)
        with col_g_metric1:
            st.metric("Total Network Nodes", c_met.total_nodes)
        with col_g_metric2:
            st.metric("Total Edges", c_met.total_edges)
        with col_g_metric3:
            st.metric("Graph Density", f"{c_met.graph_density:.3f}")
        with col_g_metric4:
            st.metric("Orthogonality Rating", f"{c_met.citation_orthogonality_score:.2f} / 1.0")

        # Render Plotly interactive figure
        if st.session_state.network_fig is not None:
            st.plotly_chart(st.session_state.network_fig, use_container_width=True)

        if c_met.most_central_paper and c_met.most_central_paper != "N/A":
            st.info(f"🏆 **Most Influential Landmark Node:** *{c_met.most_central_paper}*")


    # ==========================================
    # TAB 7: REPORT & EXPORT
    # ==========================================
    with tab_report:
        st.markdown("### 📄 Assessment Report & Data Export")
        st.markdown("Export publication-ready documentation of this novelty assessment.")

        md_report = export_markdown_report(res)
        json_report = export_json_report(res)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Full Markdown Report (.md)",
                data=md_report,
                file_name=f"novelty_assessment_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                label="📥 Download Structured JSON Data (.json)",
                data=json_report,
                file_name=f"novelty_assessment_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("#### ⚠️ Limitations & Recommendations for Peer Review")
        for lim in res.limitations_and_recommendations:
            st.warning(lim)

        st.markdown("---")
        with st.expander("👁️ View Full Markdown Report Preview"):
            st.markdown(md_report)
