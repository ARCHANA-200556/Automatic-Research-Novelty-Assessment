"""
Research Input Parser and Document Extractor.
Handles manual text ingestion, PDF document extraction (pypdf),
academic search query generation, section segmentation, and demo sample presets.
"""

import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pypdf
from app.config import ResearchInput, BASE_DIR


def load_presets() -> List[Dict[str, Any]]:
    """Loads sample proposals for 1-click testing."""
    benchmark_file = BASE_DIR / "sample_data" / "benchmarks.json"
    if benchmark_file.exists():
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("sample_proposals", [])
        except Exception:
            pass
    return []


def clean_text(text: str) -> str:
    """Removes excessive whitespace, hyphenation at line breaks, and strange chars."""
    if not text:
        return ""
    # Merge hyphenated words across line breaks (e.g., "algo-\nrithm" -> "algorithm")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace multiple spaces/newlines with single whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def generate_academic_search_queries(proposal: ResearchInput) -> List[str]:
    """
    Formulates adaptive, prioritized search queries for OpenAlex, Semantic Scholar,
    and Crossref based on any research proposal title, abstract, keywords, and objectives.
    Removes conversational boilerplate to ensure high recall and precision.
    """
    queries = []
    stopwords = {
        'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'and', 'or', 'as', 'into', 'towards', 'through', 'under', 'using', 'based',
        'via', 'novel', 'study', 'approach', 'investigation', 'method', 'paper', 'system'
    }
    
    # 1. Clean title of academic framing boilerplates
    raw_title = proposal.title.strip()
    clean_title = re.sub(
        r'^(?i:\b(?:a\s+study\s+(?:on|of)|investigating|an\s+investigation\s+(?:into|of)|towards|on\s+the|a\s+novel\s+approach\s+(?:for|to)|novel|automatic|automated|analysis\s+of|evaluation\s+of|development\s+of)\b[:\s\-]*)',
        '',
        raw_title
    ).strip()
    
    # If clean_title has >= 2 words, it's our primary candidate
    if len(clean_title.split()) >= 2:
        queries.append(clean_title)
    
    # 2. Add concise 3-5 key content words from title (essential for multi-word titles)
    content_words = [w for w in re.findall(r'[a-zA-Z0-9\-]+', clean_title) if w.lower() not in stopwords and len(w) > 2]
    if len(content_words) >= 2:
        queries.append(" ".join(content_words[:4]))
    
    # 3. Add raw title if different and not too excessively long
    if raw_title and raw_title != clean_title and len(raw_title.split()) <= 12:
        queries.append(raw_title)

    # 4. Keyword-driven query
    if proposal.keywords:
        valid_kw = [k.strip() for k in proposal.keywords if len(k.strip()) > 2]
        if valid_kw:
            # Query with top 3 keywords
            queries.append(" ".join(valid_kw[:3]))
            # Query combining first keyword and core title term
            first_kw = valid_kw[0]
            if content_words:
                queries.append(f"{first_kw} {' '.join(content_words[:2])}")

    # 5. Extract salient scientific acronyms or hyphenated terms from abstract / objectives
    extra_text = proposal.abstract + " " + " ".join(proposal.objectives)
    if extra_text.strip():
        sci_terms = re.findall(r'\b[A-Z0-9]{2,}(?:-[a-zA-Z0-9]+)*\b|\b[a-zA-Z]+-[a-zA-Z]+\b', extra_text)
        unique_sci = list(dict.fromkeys([t for t in sci_terms if len(t) > 2 and t.lower() not in stopwords]))
        if len(unique_sci) >= 2:
            queries.append(" ".join(unique_sci[:3]))

    # Deduplicate queries while preserving order
    deduped = []
    seen = set()
    for q in queries:
        norm = " ".join(q.lower().split())
        if norm and norm not in seen and len(norm) > 3:
            seen.add(norm)
            deduped.append(q)

    # Fallback default if title was very short
    if not deduped:
        deduped = [raw_title if raw_title else "novel research methodology"]

    return deduped


def parse_pdf_document(pdf_file_or_path) -> Tuple[Optional[ResearchInput], str]:
    """
    Extracts text and key sections (Title, Abstract, Objectives/Methods) from a PDF.
    Returns (ResearchInput, status_message).
    """
    try:
        reader = pypdf.PdfReader(pdf_file_or_path)
        num_pages = len(reader.pages)
        if num_pages == 0:
            return None, "The uploaded PDF has 0 pages."

        full_text_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            full_text_pages.append(page_text)

        full_raw_text = "\n\n".join(full_text_pages)
        cleaned = clean_text(full_raw_text)

        if len(cleaned) < 50:
            return None, "The PDF contained very little extractable text. It may be a scanned image."

        # Parse sections from the first 2 pages
        first_section = "\n".join(full_text_pages[:min(2, num_pages)])
        
        # 1. Title Heuristics
        title = "Extracted Research Document"
        lines = [line.strip() for line in first_section.split("\n") if line.strip()]
        candidate_title_lines = []
        for line in lines:
            if re.search(r'^(abstract|introduction|keywords|author|department)', line, re.IGNORECASE):
                break
            if len(line) > 5 and len(candidate_title_lines) < 3:
                candidate_title_lines.append(line)
        if candidate_title_lines:
            title = " ".join(candidate_title_lines)

        # 2. Abstract extraction
        abstract = ""
        abstract_match = re.search(r'(?i)\babstract\b[:\s\-\—]*(.*?)(?=(?i:\b(?:keywords?|introduction|1\.\s+introduction|index terms)\b|\n\n\n|$))', cleaned, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            if len(abstract) > 3000:
                abstract = abstract[:3000]
        else:
            abstract = cleaned[:1200]

        # 3. Keywords extraction
        keywords = []
        kw_match = re.search(r'(?i)\b(?:keywords?|index terms)\b[:\s\-\—]*(.*?)(?=(?i:\b(?:introduction|1\.\s+introduction)\b|\n\n|$))', cleaned, re.DOTALL)
        if kw_match:
            kw_text = kw_match.group(1).strip()
            parts = re.split(r'[,;•\n]', kw_text)
            keywords = [p.strip() for p in parts if len(p.strip()) > 2][:10]

        # 4. Objectives or Research Questions extraction
        objectives = []
        obj_matches = re.findall(r'(?i)(?:objective[s]?|aim[s]?|goal[s]?|we propose to|we investigate|we aim to|in this paper we)\s*[:\-\—]?\s*(.*?)(?=\.|\n\n)', cleaned)
        if obj_matches:
            for m in obj_matches[:5]:
                m_clean = m.strip()
                if len(m_clean) > 20 and m_clean not in objectives:
                    objectives.append(m_clean)

        if not objectives:
            objectives = [
                "Investigate novel methodologies proposed in the manuscript.",
                "Benchmark empirical performance against established baselines.",
                "Demonstrate conceptual advancements over related prior literature."
            ]

        filename = getattr(pdf_file_or_path, 'name', 'Uploaded_Document.pdf')

        parsed = ResearchInput(
            title=title,
            abstract=abstract,
            keywords=keywords,
            objectives=objectives,
            full_text=cleaned[:50000],
            source_filename=filename
        )
        return parsed, f"Successfully parsed {num_pages} pages from '{filename}'."

    except Exception as e:
        return None, f"PDF Parsing Error: {str(e)}"


def build_manual_research_input(
    title: str,
    abstract: str,
    keywords_str: str = "",
    objectives_str: str = ""
) -> ResearchInput:
    """Builds a verified ResearchInput object from manual form fields."""
    kw_list = []
    if keywords_str:
        kw_list = [k.strip() for k in re.split(r'[,;\n]', keywords_str) if k.strip()]

    obj_list = []
    if objectives_str:
        for line in objectives_str.split("\n"):
            line = re.sub(r'^\s*[\d\.\-\*•\)]+\s*', '', line).strip()
            if line:
                obj_list.append(line)

    return ResearchInput(
        title=clean_text(title),
        abstract=clean_text(abstract),
        keywords=kw_list,
        objectives=obj_list if obj_list else [
            "Investigate the primary hypotheses formulated in the research title and abstract.",
            "Compare proposed architectural and empirical methodologies against relevant prior literature."
        ],
        full_text=f"{clean_text(title)}\n\n{clean_text(abstract)}"
    )
