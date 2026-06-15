"""
Hybrid Ensemble Retriever (BM25 + Chroma dense)
"""

import os
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "workout_sessions"
EMBEDDING_MODEL = "gemini-embedding-001"
ENSEMBLE_WEIGHTS = [0.4, 0.6] # BM25 gets 40% weight, Chroma gets 60%
TOP_K = 8



def build_retriever(documents: list[Document], api_key: str, force_rebuild: bool = False) -> EnsembleRetriever:
    """
    Build or reload a hybrid EnsembleRetriever from pre-built Documents.
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model = EMBEDDING_MODEL,
        google_api_key = api_key
    )

    dense_retriever = _build_dense_retriever(documents, embeddings, force_rebuild)
    sparse_retreiver = _build_sparse_retriever(documents)

    ensemble = EnsembleRetriever(
        retrievers=[sparse_retreiver, dense_retriever],
        weights= ENSEMBLE_WEIGHTS
    )
    print(f"[retriever] Hybrid retriever ready (BM25 {ENSEMBLE_WEIGHTS[0]}) / Dense {ENSEMBLE_WEIGHTS[1]}")
    return ensemble


# Retrieval Utility Functions

def search_sessions(retriever: EnsembleRetriever, query: str) -> list[Document]:
    """
    Semantic + keyword search
    """
    return retriever.invoke(query)

def get_sessions_by_week(documents: list[Document], week: str) -> list[Document]:
    """
    Return all sessions for a given week
    """
    week_lower = str(week).lower()
    return [d for d in documents if week_lower in str(d.metadata.get("week", "")).lower()]

def get_exercise_history(documents: list[Document], exercise_name: str) -> list[Document]:
    """
    All sessions containing an exercise across all weeks
    """
    name_lower = exercise_name.lower()
    return [d for d in documents if any(name_lower in ex.lower() for ex in d.metadata.get("exercise_names", []))]

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

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

def _build_sparse_retriever(documents):
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = TOP_K
    print(f"[Retriever] BM25 index built over {len(documents)} documents")
    return retriever