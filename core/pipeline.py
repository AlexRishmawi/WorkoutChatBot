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
DEFAULT_TOP_K = 32
DEFAULT_RERANK_TOP_N = 16

def get_program_size(documents: list[Document]) -> dict:
    """
    Analyzes parsed training log on startup.
    Determines optimal K retrieve and N rerank.
    """
    #Safety Margin M
    M = 4
    #Padding
    P = 3

    if not documents:
        return {
            "total_docs": 0,
            "weeks": 0,
            "max_daily_exercises": 6,
            "t_max": 6,
            "k_retrieve": 12,
            "n_rerank": 8
        }
    
    weeks = set()
    for doc in documents:
        w = doc.metadata.get("week")
        if w:
            weeks.add(str(w).strip())
    num_weeks = max(1, len(weeks))

    sessions = {}
    for doc in documents:
        w = doc.metadata.get("week")
        d = doc.metadata.get("day")
        if w and d:
            key = (str(w).strip(), str(d).strip())
            sessions[key] = sessions.get(key, 0) + 1
    
    sessions_sizes = list(sessions.values())
    max_daily_exercises = max(sessions_sizes) if sessions_sizes else 6

    if max_daily_exercises < 4:
        max_daily_exercises = 6

    t_lookup = max_daily_exercises + P
    t_trend = int(num_weeks * 1.5)
    t_max = max(t_lookup, t_trend)

    k_retrieve = (t_max * 2) + M
    n_rerank = t_max + 2

    total_docs = len(documents)
    k_retrieve = min(total_docs, k_retrieve)
    n_rerank = min(n_rerank, k_retrieve)
    
    return {
        "total_docs": total_docs,
        "weeks": num_weeks,
        "max_daily_exercises": max_daily_exercises,
        "t_max": t_max,
        "k_retrieve": k_retrieve,
        "n_rerank": n_rerank
    }

def _get_sizing(obj, fallback_k = DEFAULT_TOP_K, fallback_n = DEFAULT_RERANK_TOP_N) -> tuple[int,int]:
    """
    Helper function for pulling k_retrieve and n_rerank off retriever or vectorstore.
    Falls back to default values if object isn't tagged.
    """
    k = getattr(obj, "_k_retrieve", fallback_k)
    n = getattr(obj, "n_rerank", fallback_n)
    return k, n



def build_retriever(documents: list[Document], api_key: str, force_rebuild: bool = False) -> tuple[EnsembleRetriever, Chroma]:
    """
    Build or reload a hybrid EnsembleRetriever from pre-built Documents.
    """

    program_size = get_program_size(documents)
    k_retrieve = program_size["k_retrieve"]
    n_rerank = program_size["n_rerank"]

    print("\n" + "─"*60)
    print("🧠 DYNAMIC SIZING REPORT")
    print("─"*60)
    print(f"  ├─ Total documents parsed: {program_size['total_docs']}")
    print(f"  ├─ Program span detected: {program_size['weeks']} weeks")
    print(f"  ├─ Max exercises per day: {program_size['max_daily_exercises']}")
    print(f"  ├─ Target footprint (T):  {program_size['t_max']} documents")
    print(f"  └─ Configured Parameters:")
    print(f"     ├─ Initial Retrieval K (Chroma & BM25): \033[92m{k_retrieve}\033[0m")
    print(f"     └─ Suggested Rerank Cutoff (n):        \033[94m{n_rerank}\033[0m")
    print("─"*60 + "\n")

    embeddings = GoogleGenerativeAIEmbeddings(
        model = EMBEDDING_MODEL,
        google_api_key = api_key
    )

    vectorstore, dense_retriever = _build_dense_retriever(documents, embeddings, force_rebuild, k_retrieve)
    sparse_retreiver = _build_sparse_retriever(documents, k_retrieve)

    ensemble = EnsembleRetriever(
        retrievers=[sparse_retreiver, dense_retriever],
        weights= ENSEMBLE_WEIGHTS
    )

    ensemble._k_retrieve = k_retrieve
    ensemble._n_rerank  = n_rerank
    vectorstore._k_retrieve = k_retrieve
    vectorstore._n_rerank = n_rerank
    print(f"[retriever] Hybrid retriever ready (BM25 {ENSEMBLE_WEIGHTS[0]}) / Dense {ENSEMBLE_WEIGHTS[1]}")
    return ensemble, vectorstore


# Retrieval Utility Functions

def search_sessions(retriever: EnsembleRetriever, query: str, use_reranker: bool = True, top_n: int | None = None) -> list[Document]:
    """
    Semantic + keyword search with alias expansion
    """
    #k_retrieve is used at construction time, no need to save value. Only n_rerank is relavent for top_n
    if top_n is None:
        _, top_n = _get_sizing(retriever)
    
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

def get_exercise_history(documents: list[Document], exercise_name: str, use_reranker: bool = False, top_n: int | None = None) -> list[Document]:
    """
    All sessions containing an exercise across all weeks
    """
    if top_n is None:
        top_n = DEFAULT_RERANK_TOP_N
    aliases = {a.lower() for a in get_all_aliases(exercise_name)}
    docs = [
        d for d in documents
        if aliases & {ex.lower() for ex in d.metadata.get("exercise_names", [])}
    ]
    if use_reranker and docs:
        docs = rerank(exercise_name, docs, top_n)
    return docs

def _build_dense_retriever(documents, embeddings, force_rebuild, k_retrieve):
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieve})
    return vectorstore, retriever

def _build_sparse_retriever(documents, k_retrieve):
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k_retrieve
    print(f"[Retriever] BM25 index built over {len(documents)} documents")
    return retriever

def search_by_exercise(vectorstore: Chroma, exercise_name: str, use_reranker: bool = True, top_n: int | None = None) -> list[Document]:
    k_retrieve = n_rerank = _get_sizing(vectorstore)
    if top_n is None:
        top_n = n_rerank

    meta_filter = build_alias_filter(exercise_name)
    canonical = normalize_exercise_name(exercise_name)
    candidates = vectorstore.similarity_search(
        canonical,
        k=k_retrieve,
        filter=meta_filter
    )
    if not use_reranker:
        return candidates
    
    return rerank(exercise_name, candidates, top_n)