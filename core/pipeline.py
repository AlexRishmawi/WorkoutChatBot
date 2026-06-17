"""
Hybrid Ensemble Retriever (BM25 + Chroma dense)
"""

import os
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from exercise_aliases import get_all_aliases, expand_query_aliases, normalize_exercise_name
from ingest import build_alias_filter
from reranker import rerank

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "workout_sessions"
EMBEDDING_MODEL = "gemini-embedding-001"
ENSEMBLE_WEIGHTS = [0.4, 0.6] # BM25 gets 40% weight, Chroma gets 60%
TOP_K = 10
RERANK_TOP_N = 10



def build_retriever(documents: list[Document], api_key: str, force_rebuild: bool = False) -> tuple[EnsembleRetriever, Chroma]:
    """
    Build or reload a hybrid EnsembleRetriever from pre-built Documents.
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model = EMBEDDING_MODEL,
        google_api_key = api_key
    )

    vectorstore, dense_retriever = _build_dense_retriever(documents, embeddings, force_rebuild)
    sparse_retreiver = _build_sparse_retriever(documents)

    ensemble = EnsembleRetriever(
        retrievers=[sparse_retreiver, dense_retriever],
        weights= ENSEMBLE_WEIGHTS
    )
    print(f"[retriever] Hybrid retriever ready (BM25 {ENSEMBLE_WEIGHTS[0]}) / Dense {ENSEMBLE_WEIGHTS[1]}")
    return ensemble, vectorstore


# Retrieval Utility Functions

def search_sessions(retriever: EnsembleRetriever, query: str, use_reranker: bool = True, top_n: int = RERANK_TOP_N) -> list[Document]:
    """
    Semantic + keyword search with alias expansion
    """
    expansions = expand_query_aliases(query)
    if expansions:
        extra = " ".join(t for t in expansions if t.lower() not in query.lower())
        expanded_query = f"{query} {extra}".strip()
    else:
        expanded_query = query
    
    candidates = retriever.invoke(expanded_query)
    if not use_reranker:
        return candidates
    
    return rerank(query, candidates, top_n)

def get_sessions_by_week(documents: list[Document], week: str) -> list[Document]:
    """
    Return all sessions for a given week
    """
    week_lower = str(week).lower()
    return [d for d in documents if week_lower in str(d.metadata.get("week", "")).lower()]

def get_exercise_history(documents: list[Document], exercise_name: str, use_reranker: bool = False, top_n: int = RERANK_TOP_N) -> list[Document]:
    """
    All sessions containing an exercise across all weeks
    """
    aliases = {a.lower() for a in get_all_aliases(exercise_name)}
    docs = [
        d for d in documents
        if aliases & {ex.lower() for ex in d.metadata.get("exercise_names", [])}
    ]
    if use_reranker and docs:
        docs = rerank(exercise_name, docs, top_n)
    return docs

def _build_dense_retriever(documents, embeddings, force_rebuild):
    chroma_exists = os.path.isdir(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR)

    if chroma_exists and not force_rebuild:
        print(f"[Retriever] Loading existing Chroma store from '{CHROMA_PERSIST_DIR}'")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
    else:
        print(f"[Retriever] Embedding {len(documents)} documents into Chroma...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        print(f"[Retriever] Chroma store persisted to '{CHROMA_PERSIST_DIR}'")

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return vectorstore, retriever

def _build_sparse_retriever(documents):
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = TOP_K
    print(f"[Retriever] BM25 index built over {len(documents)} documents")
    return retriever

def search_by_exercise(vectorstore: Chroma, exercise_name: str, use_reranker: bool = True, top_n: int = RERANK_TOP_N,) -> list[Document]:
    meta_filter = build_alias_filter(exercise_name)
    canonical = normalize_exercise_name(exercise_name)
    candidates = vectorstore.similarity_search(
        canonical,
        k=TOP_K,
        filter=meta_filter
    )
    if not use_reranker:
        return candidates
    
    return rerank(exercise_name, candidates, top_n)