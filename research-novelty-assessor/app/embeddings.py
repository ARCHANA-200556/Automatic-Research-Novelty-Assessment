"""
Semantic Embeddings and Vector Similarity Analysis Module.
Uses sentence-transformers (all-MiniLM-L6-v2) to generate dense contextual embeddings,
with a transparent Scikit-Learn Vector Space fallback.
Computes cosine similarity, ranks literature, and extracts overlapping concepts.
"""

import logging
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import PaperMetadata, SimilarityItem, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Global model cache to avoid repeated disk reads
_ST_MODEL = None
_ST_AVAILABLE = None


def get_sentence_transformer_model():
    """Lazily loads the sentence-transformers model if available."""
    global _ST_MODEL, _ST_AVAILABLE
    if _ST_AVAILABLE is False:
        return None
    if _ST_MODEL is not None:
        return _ST_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer: {EMBEDDING_MODEL_NAME}...")
        _ST_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _ST_AVAILABLE = True
        return _ST_MODEL
    except Exception as e:
        logger.warning(f"SentenceTransformer could not be loaded: {e}. Using TF-IDF vector space.")
        _ST_AVAILABLE = False
        return None


def extract_key_overlapping_terms(query_text: str, doc_text: str, top_n: int = 5) -> List[str]:
    """Identifies the most salient overlapping academic terms between query and paper."""
    stop_words = {
        'the', 'and', 'for', 'with', 'this', 'that', 'our', 'from', 'which', 'using', 'used',
        'paper', 'propose', 'proposed', 'approach', 'model', 'method', 'results', 'based',
        'study', 'novel', 'new', 'present', 'presents', 'show', 'shows', 'can', 'were', 'have'
    }
    # Tokenize words of length >= 4
    q_words = set(w.lower() for w in re.findall(r'[a-zA-Z]{4,}', query_text) if w.lower() not in stop_words)
    d_words = set(w.lower() for w in re.findall(r'[a-zA-Z]{4,}', doc_text) if w.lower() not in stop_words)
    common = list(q_words.intersection(d_words))
    # Sort by length and specificity
    common.sort(key=lambda x: len(x), reverse=True)
    return common[:top_n]


def compute_semantic_similarities(
    proposal_text: str,
    papers: List[PaperMetadata]
) -> Tuple[List[SimilarityItem], str]:
    """
    Computes pairwise semantic similarities between the proposed research text
    and all candidate paper abstracts.
    Returns: (ranked_similarity_items, engine_description)
    """
    if not papers:
        return [], "No papers available"

    paper_texts = []
    for p in papers:
        text = f"{p.title}. {p.abstract}"
        paper_texts.append(text.strip())

    st_model = get_sentence_transformer_model()

    if st_model is not None:
        try:
            # 1. Encode with SentenceTransformers
            query_emb = st_model.encode([proposal_text], convert_to_numpy=True)
            paper_embs = st_model.encode(paper_texts, convert_to_numpy=True)

            # Cosine similarity
            scores = cosine_similarity(query_emb, paper_embs)[0]
            engine_desc = f"Dense Vector Embeddings ({EMBEDDING_MODEL_NAME})"
        except Exception as e:
            logger.error(f"Error during SentenceTransformer inference: {e}")
            scores = _compute_tfidf_similarities(proposal_text, paper_texts)
            engine_desc = "TF-IDF N-Gram Vector Space [Fallback]"
    else:
        # Fallback to TF-IDF with sublinear scaling and ngram_range (1, 2)
        scores = _compute_tfidf_similarities(proposal_text, paper_texts)
        engine_desc = "TF-IDF N-Gram Vector Space [Dense model offline]"

    # Build ranked items
    ranked_indices = np.argsort(scores)[::-1]
    items: List[SimilarityItem] = []

    for rank, idx in enumerate(ranked_indices, start=1):
        raw_score = float(scores[idx])
        # Bound score between 0.0 and 1.0
        clamped_score = max(0.0, min(1.0, round(raw_score, 4)))
        paper = papers[idx]
        overlap_terms = extract_key_overlapping_terms(proposal_text, paper_texts[idx])

        items.append(SimilarityItem(
            paper=paper,
            similarity_score=clamped_score,
            rank=rank,
            key_overlap_terms=overlap_terms
        ))

    return items, engine_desc


def _compute_tfidf_similarities(query_text: str, docs: List[str]) -> np.ndarray:
    """Calculates n-gram TF-IDF cosine similarities."""
    corpus = [query_text] + docs
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english',
        max_features=5000
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = tfidf_matrix[0:1]
        doc_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(query_vec, doc_vecs)[0]
        return sims
    except Exception:
        # If corpus vocabulary is empty or degenerate
        return np.zeros(len(docs))
