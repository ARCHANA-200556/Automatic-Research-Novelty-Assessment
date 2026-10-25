"""
FastAPI Backend Application for Research Novelty Assessment.
Exposes REST endpoints for paper retrieval, similarity calculation,
citation graph metrics, and end-to-end assessment.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import io

from app.config import ResearchInput, NoveltyAssessmentResult
from app.pipeline import run_full_novelty_assessment
from app.input_parser import parse_pdf_document, build_manual_research_input, load_presets
from app.retrieval import retrieve_relevant_literature

app = FastAPI(
    title="Automatic Research Novelty Assessment API",
    description="REST API evaluating research originality using LLMs, RAG, and Citation Graphs",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Research Novelty Assessor"}


@app.get("/api/presets")
def get_sample_presets():
    """Returns curated demo research proposals."""
    return {"presets": load_presets()}


@app.post("/api/assess", response_model=NoveltyAssessmentResult)
def assess_novelty(
    proposal: ResearchInput,
    limit: int = 10,
    force_fallback: bool = False
):
    """Executes full novelty assessment workflow on the provided research proposal."""
    try:
        result, _, _ = run_full_novelty_assessment(
            proposal=proposal,
            limit_papers=limit,
            force_fallback=force_fallback
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload_pdf")
async def upload_pdf_document(file: UploadFile = File(...)):
    """Parses an uploaded PDF research paper and extracts structured fields."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
    
    contents = await file.read()
    pdf_stream = io.BytesIO(contents)
    pdf_stream.name = file.filename
    
    parsed, msg = parse_pdf_document(pdf_stream)
    if not parsed:
        raise HTTPException(status_code=422, detail=msg)
    
    return {"message": msg, "proposal": parsed}
