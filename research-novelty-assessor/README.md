# Automatic Research Novelty Assessment System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAlex API](https://img.shields.io/badge/retrieval-OpenAlex%20API-orange.svg)](https://openalex.org/)
[![NetworkX](https://img.shields.io/badge/graphs-NetworkX-blueviolet.svg)](https://networkx.org/)

An evidence-grounded web application designed to evaluate research paper originality and detect potential research gaps using **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, **Dense Vector Embeddings**, and **NetworkX Citation Graph Analysis**.

---

## Key Features

1. **Multi-Modal Research Input**:
   - Ingest research proposals via structured manual forms (Title, Abstract, Keywords, Objectives).
   - Ingest complete research papers in PDF format with automated section parsing (`pypdf`).
   - One-click demo presets representing diverse academic domains (AI, Medicine, TinyML).

2. **Real Academic Paper Retrieval**:
   - Queries real academic databases via the **OpenAlex API** (hundreds of millions of indexed papers, DOIs, abstracts, and reference connections).
   - Zero hallucination guarantee: Never invents papers or citations.
   - Built-in transparent offline fallback with curated peer-reviewed benchmarks when network is unavailable.

3. **Semantic Similarity Matrix**:
   - Contextual vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and cosine similarity.
   - Dual-mode architecture with automatic N-gram Vector Space Model fallback.
   - Identifies conceptual proximity even when authors use differing vocabulary.

4. **RAG-Grounded Comparative Analysis**:
   - Chunks retrieved literature into passage evidence units.
   - Compares proposed objectives and methodologies against existing literature using strict evidence grounding (via Google Gemini API or local deterministic synthesis).

5. **5-Dimensional Research Gap Identification**:
   - Evaluates: **Objectives**, **Methodologies**, **Datasets & Scale**, **Target Domain**, and **Core Contributions**.
   - Explicitly classifies findings into `[POTENTIAL GAP]` vs `[VERIFIED OVERLAP]`.

6. **NetworkX Citation Graph Analysis**:
   - Constructs directed citation networks representing reference flows and landmark authorities.
   - Computes graph density, in-degree centrality, and citation bridge orthogonality.
   - Renders interactive 2D network visualizations via **Plotly**.

7. **Documented Mathematical Novelty Scoring (0–100)**:
   - Evaluates a mathematically bounded composite equation:
     $$\text{Novelty Score} = 100 \times \left[ 0.35(1 - \text{MaxSim}) + 0.20(1 - \text{MeanTopK}) + 0.30\text{GapScore} + 0.15\text{CitationScore} \right]$$
   - Computes an **Evidence Coverage & Confidence Index (%)**.
   - Clear disclaimer: The calculated score is an exploratory decision-support metric, not formal proof of originality.

8. **Exportable Explainable Reports**:
   - Download complete assessment reports in **Markdown (`.md`)** and structured **JSON (`.json`)**.

---

## Directory Structure

```
research-novelty-assessor/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic schemas, scoring weights, API configs
│   ├── input_parser.py           # PDF extraction, text cleaning, section parser
│   ├── retrieval.py              # OpenAlex and Semantic Scholar APIs + fallback
│   ├── embeddings.py             # SentenceTransformers & Cosine similarity
│   ├── rag_engine.py             # Grounded passage retrieval & comparative reasoning
│   ├── gap_analyzer.py           # 5-dimension gap detection (potential vs verified)
│   ├── citation_graph.py         # NetworkX citation graph & Plotly visualization
│   ├── novelty_scorer.py         # 0-100 mathematical scoring & confidence metrics
│   ├── report_generator.py      # Markdown and JSON report exporters
│   ├── pipeline.py               # Unified end-to-end evaluation orchestrator
│   ├── api.py                    # FastAPI REST backend endpoints
│   └── main_app.py               # Streamlit interactive web dashboard
├── tests/
│   ├── test_input_parser.py      # PDF & manual input tests
│   ├── test_retrieval.py         # OpenAlex parsing & fallback tests
│   ├── test_similarity.py        # Embedding & ranking tests
│   ├── test_gap_analyzer.py      # Gap classification tests
│   ├── test_citation_graph.py    # NetworkX & Plotly tests
│   └── test_novelty_scorer.py    # Scoring formula invariants
├── sample_data/
│   └── benchmarks.json           # Benchmark papers and demo proposals
├── requirements.txt              # Dependency specifications
├── .env.example                  # Environment configuration template
└── README.md                     # Documentation
```

---

## Quickstart Installation

### 1. Clone or Navigate to the Workspace
```bash
cd C:\Users\archa\.gemini\antigravity\scratch\research-novelty-assessor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Add your optional API keys for Gemini, OpenAI, or Semantic Scholar if desired. If left blank, the application will automatically use the public OpenAlex API and local evidence synthesis engine.

---

## Running the Application

### 1. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app/main_app.py
```
Open your browser at `http://localhost:8501`.

### 2. (Optional) Launch the FastAPI Backend
```bash
uvicorn app.api:app --reload --port 8000
```
API Documentation will be live at `http://localhost:8000/docs`.

---

## Running Automated Tests

Run the test suite to verify all modules:
```bash
python -m pytest -v
```

---

## Academic Disclaimer
The Potential Novelty Score and associated research gap analyses generated by this system are automated heuristic assessments intended for academic exploration and preliminary screening. They do not constitute conclusive legal proof of intellectual property rights, patentability, or academic priority.
