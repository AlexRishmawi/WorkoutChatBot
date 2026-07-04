"""
reranker.py
-----------
Cross-encoder re-ranking step inserted between retrieval and the LLM.

Why this exists
----------------
BM25 + dense retrieval (the EnsembleRetriever) are both *bi-encoder* style
scorers: the query and each document are scored independently, then
compared. This is fast and scales well, but it's an approximation —
neither retriever ever looks at the query and a candidate document
*together*.

Alias query expansion (see exercise_aliases.py) widens the candidate pool
on purpose to improve recall — e.g. a "squat" query now also retrieves
Front Squat, Goblet Squat, Bulgarian Split Squat documents. That's good
for recall, but it means BM25/dense scores no longer reliably rank the
single best match first, which is why MRR can drop even as recall rises.

A cross-encoder fixes this final-mile ranking problem: it takes the
(query, document) pair *together* as a single input and outputs a
relevance score with full attention across both. It's too slow to run
over your whole corpus, but it's cheap to run over the ~10-20 candidates
the ensemble retriever already narrowed things down to.

Pipeline position
------------------
BM25 ─┐
       ├─→ Ensemble merge ─→ [THIS MODULE] re-rank ─→ top N ─→ Gemini
Dense ─┘
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

from huggingface_hub import scan_cache_dir
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 16

# Lazy-loaded singleton — loading a cross-encoder takes ~1-2s and we don't
# want that cost paid at import time (e.g. during eval_pipeline startup
# before any query has actually been issued).
_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    cached = any(
        repo.repo_id == CROSS_ENCODER_MODEL
        for repo in scan_cache_dir().repos
    )
    if cached:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
 
    print(f"[Reranker] Loading cross-encoder '{CROSS_ENCODER_MODEL}'"
          f"{' (offline, from cache)' if cached else ' (downloading...)'}")
 
    try:
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception as e:
        raise RuntimeError(
            f"[Reranker] Failed to load '{CROSS_ENCODER_MODEL}'. "
            f"Original error: {e}"
        ) from e
 
    print("[Reranker] Model loaded.")
    return _model



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    documents: list[Document],
    top_n: int = DEFAULT_TOP_N,
) -> list[Document]:
    """
    Re-score `documents` against `query` with a cross-encoder and return
    the top_n highest-scoring documents, ordered best-first.

    If `documents` is empty, returns it unchanged (no-op).
    If fewer than top_n documents are passed in, returns all of them,
    re-ordered by score.
    """
    if not documents:
        return documents

    model = _get_model()

    # Cross-encoders score a single (query, passage) pair at a time, but
    # sentence-transformers' .predict() batches this efficiently under
    # the hood — pass everything in one call rather than looping.
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda pair: pair[1], reverse=True)

    ranked_docs = [doc for doc, _score in scored_docs[:top_n]]
    return ranked_docs


def rerank_with_scores(
    query: str,
    documents: list[Document],
    top_n: int = DEFAULT_TOP_N,
) -> list[tuple[Document, float]]:
    """
    Same as rerank(), but also returns the raw cross-encoder score for
    each document. Useful for debugging / logging why a doc ranked
    where it did (e.g. via --sources in main.py).
    """
    if not documents:
        return []

    model = _get_model()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda pair: pair[1], reverse=True)

    return [(doc, float(score)) for doc, score in scored_docs[:top_n]]