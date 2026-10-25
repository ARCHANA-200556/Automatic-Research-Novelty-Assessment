"""
Configuration and Data Models for Research Novelty Assessor.
Defines Pydantic schemas, scoring formula weights, and API settings.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if .env exists
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- API Keys & Endpoints ---
OPENALEX_API_BASE = "https://api.openalex.org"
SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
CROSSREF_API_BASE = "https://api.crossref.org"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# --- Scoring Weights ---
# Novelty Score = 100 * [w_s * (1 - MaxSim) + w_m * (1 - MeanTopK) + w_g * GapScore + w_c * CitationScore]
WEIGHT_MAX_SIMILARITY = 0.35
WEIGHT_MEAN_SIMILARITY = 0.20
WEIGHT_GAP_EVIDENCE = 0.30
WEIGHT_CITATION_ORTHOGONALITY = 0.15

# Default thresholds
TOP_K_PAPERS_DEFAULT = 10
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --- Data Models ---
class ResearchInput(BaseModel):
    title: str = Field(..., description="Title of the proposed research")
    abstract: str = Field(..., description="Abstract or executive summary")
    keywords: List[str] = Field(default_factory=list, description="Keywords or topic tags")
    objectives: List[str] = Field(default_factory=list, description="Specific research aims or objectives")
    full_text: Optional[str] = Field(None, description="Extracted full paper text if PDF uploaded")
    source_filename: Optional[str] = Field(None, description="Filename if uploaded from PDF")


class PaperMetadata(BaseModel):
    id: str = Field(..., description="Unique paper identifier (OpenAlex ID, S2 ID, or DOI)")
    title: str = Field(..., description="Title of the retrieved paper")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Journal, conference, or repository venue")
    abstract: str = Field("", description="Paper abstract text")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    url: Optional[str] = Field(None, description="Direct URL or OpenAlex/DOI link")
    citation_count: int = Field(0, description="Number of times cited")
    referenced_works: List[str] = Field(default_factory=list, description="IDs of referenced works")
    source_api: str = Field("OpenAlex", description="Source API where paper was retrieved")
    is_fallback: bool = Field(False, description="Flag indicating offline benchmark fallback")


class SimilarityItem(BaseModel):
    paper: PaperMetadata
    similarity_score: float = Field(..., description="Cosine similarity score [0.0 - 1.0]")
    rank: int = Field(..., description="Rank in similarity order (1 = closest)")
    key_overlap_terms: List[str] = Field(default_factory=list, description="Salient overlapping concepts")


class PassageChunk(BaseModel):
    paper_id: str
    paper_title: str
    section_type: str = Field("abstract", description="Section (e.g., abstract, methodology, results)")
    text: str
    relevance_score: float = 0.0


class GapDimensionItem(BaseModel):
    dimension: str = Field(..., description="Objectives, Methods, Datasets, Domain, or Contributions")
    proposed_aspect: str = Field(..., description="Proposed approach in this dimension")
    prior_art_state: str = Field(..., description="Existing literature approaches")
    classification: str = Field(..., description="'POTENTIAL GAP' or 'VERIFIED OVERLAP'")
    rationale: str = Field(..., description="Evidence-grounded explanation")
    confidence: float = Field(..., description="Confidence score [0.0 - 1.0]")
    supporting_citations: List[str] = Field(default_factory=list, description="Citations supporting this finding")


class CitationGraphMetrics(BaseModel):
    total_nodes: int = 0
    total_edges: int = 0
    verified_citation_edges: int = 0
    proposed_in_degree: int = 0
    proposed_out_degree: int = 0
    max_in_degree: int = 0
    most_central_paper: Optional[str] = None
    graph_density: float = 0.0
    modularity_cluster_count: int = 1
    citation_orthogonality_score: float = 0.5  # 0.0 to 1.0
    citation_coverage_note: str = ""


class NoveltyAssessmentResult(BaseModel):
    proposal: ResearchInput
    overall_novelty_score: float = Field(..., description="Overall novelty score from 0.0 to 100.0")
    novelty_tier: str = Field(..., description="Categorization: Incremental, Moderate Novelty, High Novelty, Disruptive")
    confidence_coverage_score: float = Field(..., description="Coverage & confidence metric [0.0 - 100.0]")
    
    # Sub-scores
    semantic_penalty_score: float
    gap_bonus_score: float
    citation_bonus_score: float
    
    # Detailed components
    max_similarity: float
    mean_similarity: float
    retrieved_papers: List[SimilarityItem]
    gap_analysis: List[GapDimensionItem]
    citation_metrics: CitationGraphMetrics
    
    # Narrative explanations
    executive_summary: str
    methodological_comparisons: List[Dict[str, Any]]
    distinctive_contributions: List[str]
    limitations_and_recommendations: List[str]
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict, description="Counts of papers per source and after deduplication")
    retrieval_diagnostics: Optional[str] = Field(None, description="Diagnostic messages or warnings from retrieval")
    is_live_retrieval: bool = True
    llm_source_used: str = "Rule-Grounded Heuristics"
