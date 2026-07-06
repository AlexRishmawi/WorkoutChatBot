"""
app.py
------
FastAPI backend service for Workout RAG v2.
Exposes REST endpoints for file upload, RAG Q&A, and topology diagnostics.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "core")))

import shutil
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core imports from local modules
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from backend.core.ingest import load_workout_documents
from backend.core.pipeline import build_retriever, get_program_size
from backend.core.chain import build_rag_chain_with_sources, _condense_question, _format_docs

# Configure local workspace paths
CACHE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
CACHE_PATH = os.path.join(CACHE_DIR, "extracted_docs.json")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")

# Ensure necessary system directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# API Schema Definition
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the chat turn")


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's latest input question")
    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list, 
        description="Sequential list of preceding messages for condensation"
    )
    use_reranker: bool = Field(
        default=True, 
        description="Whether to run the secondary cross-encoder reranking pass"
    )
    model: str = Field(
        default="gemini-2.5-flash", 
        description="Gemini chat model version to run"
    )


class ExerciseSource(BaseModel):
    content: str
    week: Optional[str] = None
    day: Optional[str] = None
    exercise_name: Optional[str] = None
    session_name: Optional[str] = None


class QueryResponse(BaseModel):
    standalone_query: str = Field(..., description="The context-condensed query sent to the retriever")
    answer: str = Field(..., description="Synthesized analytical response from the LLM")
    sources: List[ExerciseSource] = Field(..., description="Granular document chunks used to compile the answer")


class ProgramStatusResponse(BaseModel):
    is_indexed: bool = Field(..., description="Whether a program has been successfully compiled and cached")
    total_documents: int = Field(0, description="Total exercise-level chunks indexed")
    weeks: int = Field(0, description="Number of unique weeks detected in the training block")
    max_daily_exercises: int = Field(0, description="Highest concentration of exercises observed in a single session")
    suggested_k_retrieve: int = Field(12, description="Retriever fetch boundary optimized by sizing engine")
    suggested_n_rerank: int = Field(8, description="Reranker output cutoff optimized by sizing engine")


# ---------------------------------------------------------------------------
# State Management & Lifespan Context
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup/shutdown cycles.
    Pre-loads the document cache and initializes the Retriever into memory.
    """
    app.state.documents = []
    app.state.retriever = None
    app.state.vectorstore = None

    # Load from extracted_docs.json if present on startup
    if os.path.exists(CACHE_PATH):
        try:
            print(f"[Startup] Loading pre-extracted program cache from {CACHE_PATH}...")
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            app.state.documents = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in cached_data
            ]
            print(f"[Startup] Loaded {len(app.state.documents)} exercise records.")

            # Resolve API Key
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                print("[Startup] Initializing retriever structures...")
                app.state.retriever, app.state.vectorstore = build_retriever(
                    documents=app.state.documents, 
                    api_key=api_key, 
                    force_rebuild=False
                )
                print("[Startup] Retriever initialization complete.")
            else:
                print("[Startup] WARNING: GEMINI_API_KEY environment variable missing. Retriever startup delayed.")
        except Exception as e:
            print(f"[Startup] Error loading cached data: {e}")

    yield

    # Shutdown / cleanup
    print("[Shutdown] Cleaning up API processes.")


# Initialize FastAPI app
app = FastAPI(
    title="Workout RAG API",
    description="FastAPI service serving structured exercise-level workout logs using dynamic hybrid RAG.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin client integration (Web interfaces, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Internal Helper Subroutines
# ---------------------------------------------------------------------------

def _get_api_key(header_key: Optional[str]) -> str:
    """
    Resolves the Google API Key checking both the HTTP headers and environment.
    """
    api_key = header_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: GEMINI_API_KEY must be provided via 'X-API-Key' header or env variable."
        )
    return api_key


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/status", response_model=ProgramStatusResponse)
async def get_system_status():
    """
    Returns diagnostic analysis about the currently loaded workout program topology.
    """
    if not app.state.documents:
        return ProgramStatusResponse(is_indexed=False)

    topology = get_program_size(app.state.documents)
    return ProgramStatusResponse(
        is_indexed=True,
        total_documents=topology["total_docs"],
        weeks=topology["weeks"],
        max_daily_exercises=topology["max_daily_exercises"],
        suggested_k_retrieve=topology["k_retrieve"],
        suggested_n_rerank=topology["n_rerank"]
    )


@app.post("/upload", response_model=ProgramStatusResponse)
async def upload_workout_program(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Uploads a training spreadsheet (.xlsx), processes it using exercise-level
    ingestion chunks, caches the structure, and indexes the results.
    """
    api_key = _get_api_key(x_api_key)

    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Unsupported file format. Only .xlsx spreadsheets are permitted.")

    # Save uploaded file to a temporary route
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file write: {str(e)}")

    try:
        # 1. Parse Excel cells into exercise-level Document structures via LLM
        print(f"[API Upload] Parsing '{file.filename}' into exercise-level chunks...")
        documents = load_workout_documents(temp_file_path, api_key)
        
        if not documents:
            raise HTTPException(status_code=422, detail="Extraction failed: No workout logs could be resolved from spreadsheet.")

        # 2. Cache parsed output locally to extracted_docs.json
        print(f"[API Upload] Caching {len(documents)} document definitions to disk...")
        serializable_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

        # 3. Synchronize App States
        app.state.documents = documents

        # 4. Rebuild retrieval indices
        print("[API Upload] Rebuilding hybrid retrieval indexes...")
        retriever, vectorstore = build_retriever(documents, api_key, force_rebuild=True)
        app.state.retriever = retriever
        app.state.vectorstore = vectorstore

        topology = get_program_size(documents)
        return ProgramStatusResponse(
            is_indexed=True,
            total_documents=topology["total_docs"],
            weeks=topology["weeks"],
            max_daily_exercises=topology["max_daily_exercises"],
            suggested_k_retrieve=topology["k_retrieve"],
            suggested_n_rerank=topology["n_rerank"]
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spreadsheet parser failed during execution: {str(e)}")
    finally:
        # Cleanup temp file securely
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/query", response_model=QueryResponse)
async def query_workout_logs(
    payload: QueryRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Chat endpoint for multi-turn conversational analysis.
    Accepts question and history, resolves rephrasing, retrieves, and synthesizes answers.
    """
    api_key = _get_api_key(x_api_key)

    if not app.state.documents or not app.state.retriever or not app.state.vectorstore:
        raise HTTPException(
            status_code=400, 
            detail="Retriever is uninitialized. Upload a spreadsheet program via /upload first to set context."
        )

    try:
        # Step 1: Map raw conversational history into LangChain Message structures
        history_messages = []
        for msg in payload.chat_history:
            if msg.role == "assistant":
                history_messages.append(AIMessage(content=msg.content))
            else:
                history_messages.append(HumanMessage(content=msg.content))

        # Instantiate LLM matching the requested model parameter
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=payload.model, google_api_key=api_key, temperature=0.2)

        # Step 2: Resolve standalone condensed query strictly using core chain logic
        standalone_query = _condense_question(llm, payload.question, history_messages)
        print(f"[RAG API Query] Raw input: '{payload.question}' -> Standalone resolved: '{standalone_query}'")

        # Step 3: Build and invoke your production intent-routing RAG chain
        chain = build_rag_chain_with_sources(
            retriever=app.state.retriever,
            vectorstore=app.state.vectorstore,
            documents=app.state.documents,
            api_key=api_key,
            model=payload.model
        )

        # Standard inputs expected by chain.py runnables
        chain_inputs = {
            "question": payload.question,
            "history": history_messages
        }

        result = chain.invoke(chain_inputs)
        answer_string = result["answer"]
        retrieved_docs = result["source_documents"]

        # Step 4: Format source elements for schema response
        formatted_sources = []
        for d in retrieved_docs:
            formatted_sources.append(ExerciseSource(
                content=d.page_content,
                week=d.metadata.get("week"),
                day=d.metadata.get("day"),
                exercise_name=d.metadata.get("exercise_name"),
                session_name=d.metadata.get("session_name")
            ))

        return QueryResponse(
            standalone_query=standalone_query,
            answer=answer_string,
            sources=formatted_sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed during execution: {str(e)}")


@app.post("/reset", response_model=Dict[str, str])
async def reset_database():
    """
    Flushes the memory state, removes the extracted JSON cache,
    and resets the persistent vector storage directory.
    """
    if getattr(app.state, "vectorstore", None) is not None:
        try:
            print("[Reset] Programmatically deleting Chroma collection...")
            app.state.vectorstore.delete_collection()
            print("[Reset] Chroma collection programmatically wiped.")
        except Exception as e:
            print(f"[Reset] Programmatic wipe skipped or failed: {e}")
    app.state.documents = []
    app.state.retriever = None
    app.state.vectorstore = None

    import gc
    gc.collect()

    # Delete Cached JSON
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    # Delete Chroma Persistent DB Directory
    chroma_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "core", "chroma_db"))
    if os.path.exists(chroma_dir):
        try:
            shutil.rmtree(chroma_dir)
            print("[Reset] Successfully deleted physical chroma_db directory.")
        except PermissionError as pe:
            print(
                f"[Reset] Windows File Lock Warning: Could not delete physical folder {chroma_dir} directly ({pe}). "
                "However, the database collection has been programmatically wiped. "
                "Please restart your app.py process if you want to cleanly delete empty directory structures."
            )
        except Exception as e:
            print(f"[Reset] Warning: Failed to clean directory {chroma_dir}: {e}")

    return {"message": "Memory logs, cached extractions, and persistent vector stores successfully flushed."}


if __name__ == "__main__":
    import uvicorn
    # Start web server automatically if run directly
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)