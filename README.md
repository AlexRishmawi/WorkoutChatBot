# Workout RAG Chatbot

A Retrieval-Augmented Generation (RAG) backend chatbot designed to parse, index, and search workout spreadsheets (`.xlsx`) using Gemini 2.5 Flash, Hybrid Ensemble Retrieval, Cross-Encoder Reranking, and Dynamic Spreadsheet Sizing.


## System Architecture

```text
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Ingestion Layer                                                                        │
│                                                                                           │
│  [ Raw Spreadsheet (.xlsx) ] ──► [ OpenPyXL Dump ] ──► [ LLM Parser ]                     │
│                                                              │                            │
│  [ Metadata Normalization ] ◄── [ Extracted Exercise JSON ] ◄┘                            │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Document Pipeline                                                                      │
│                                                                                           │
│                   ┌──► Dynamic Sizing Engine (K_retrieve, N_rerank)                       │
│                   │                                                                       │
│  [ Document Pool ] ──► Sparse Vector Index (BM25 Keyword Search) ─┐                       │
│                   │                                               ├─► EnsembleRetriever   │
│                   └──► Dense Vector Index (ChromaDB)         ─────┘    (40/60 Weighted)   │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Query Chain                                                                            │
│                                                                                           │
│  [ User Question + History ] ──► [ Standalone Query Condensation ]                        │
│                                                       │                                   │
│                                                       ▼                                   │
│                                        [ Intent Classifier Routing ]                      │
│                                         /          |          \                           │
│                         ┌──────────────┘           │           └──────────────┐           │
│                         ▼                          ▼                          ▼           │
│              [ History Intent ]          [ Detail/Search Intent ]       [ Session Search ]│
│             (Exhaustive Scan)            (Alias Filtered Dense)          (Ensemble Search)│
└─────────────────────────┬──────────────────────────┬──────────────────────────┬───────────┘
                          │                          │                          │
                          └──────────────────────────┼──────────────────────────┘
                                                     ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Reranking Layer                                                                        │
│                                                                                           │
│  [ Candidate Documents ] ──► Cross-Encoder (MS-Marco MiniLM) ──► Top N Cutoff Candidate   │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Response                                                                               │
│                                                                                           │
│  [ Reranked Context + Condensed Query ] ──► [ Gemini 2.5 Flash LLM ] ──► [ User Response ]│
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

## Core Features

### 1. LLM-Powered Unstructured Ingestion

Traditional RAG text splitters fail on spreadsheet grids because row/column boundaries disconnect exercises from their set/rep headers.

- **Cell Dumps:** Raw `.xlsx` sheets are converted into cell coordinate tuples (`row R | col C | value`) and processed via **Gemini 2.5 Flash** with temperature `0`.
- **Chunking:** Sessions are broken down into **exercise-level Documents** (each containing approximately 100 tokens of structured metadata).

### 2. Dynamic Sizing Engine

Static retrieval limits (`K`) cause context window bloat on small queries and recall drops on large macrocycles. The system automatically inspects program metadata on startup:

- Calculates the total program span in weeks (`W`) and maximum daily exercise volume (`V`).
- Determines the **Target Footprint** (`T_max`) required to satisfy both daily lookups and multi-week trend queries:

$$
T_{\text{lookup}} = V + P
$$

$$
T_{\text{trend}} = \lfloor W \times 1.5 \rfloor
$$

$$
T_{\text{max}} = \max(T_{\text{lookup}}, T_{\text{trend}})
$$

- Dynamically sets initial retrieval depth (`K_retrieve`) and reranker cutoff (`N_rerank`) using safety margins (`M = 4`, `P = 3`):

$$
K_{\text{retrieve}} = \min(\text{Total Docs}, (T_{\text{max}} \times 2) + M)
$$

$$
N_{\text{rerank}} = \min(K_{\text{retrieve}}, T_{\text{max}} + P)
$$

### 3. Multi-Turn Query Condensation

In conversational turns (e.g., User: *"What did I squat in Week 1?"* → *"How about Week 2?"*), the query condensation module rephrases follow-up questions into context-aware search queries before triggering the retriever.

### 4. Intent-Based Query Routing

Queries pass through an intent classifier that evaluates user intent and applies query expansion (`exercise_aliases.py`):

1. **History/Trend Intent** (e.g., *"Show my bench progression over time"*): Triggers `get_exercise_history`, performing an exhaustive scan over all matching exercise records across the entire dataset.
2. **Detail/Qualitative Intent** (e.g., *"What notes did I leave on deadlifts?"*): Triggers `search_by_exercise`, executing an alias-filtered dense search combined with Cross-Encoder reranking.
3. **General Intent** (e.g., *"What exercises did I perform on Wednesday?"*): Executes full hybrid sparse/dense search (`search_sessions`).

### 5. Hybrid Retrieval & Cross-Encoder Reranking

- **Bi-Encoder Layer:** Combines sparse **BM25** (weight `0.4`) and dense **ChromaDB** embeddings (weight `0.6`, model `models/text-embedding-004`).
- **Cross-Encoder Layer:** Candidate chunks pass through a sentence-transformer cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). This scores query-document attention pairs directly, resolving bi-encoder ranking limitations and boosting Mean Reciprocal Rank (MRR).

## Prerequisites & Installation

### Prerequisites

- **Python 3.10+** (Python 3.13 supported)
- **Google Gemini API Key** (`GEMINI_API_KEY` or `GOOGLE_API_KEY`)

### Installation

1. **Clone the repository and enter the directory:**

```bash
git clone https://github.com/your-username/WorkoutChatBot.git
cd WorkoutChatBot
```

2. **Create and activate a virtual environment:**

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install fastapi uvicorn openpyxl langchain-core langchain-google-genai langchain-community langchain-chroma sentence-transformers huggingface_hub requests python-dotenv
```

4. **Configure your Environment Variables:**

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

## Running the Application

### FastAPI Web Service

To start the REST API backend server with live reload:

```bash
python app.py
```

The server will initialize at:

```text
http://127.0.0.1:8000
```

### CLI Mode

To run the RAG system directly from the terminal without starting the web server:

```bash
python backend/core/main.py --xlsx "path/to/your_program.xlsx" --sources
```

## REST API Reference

### 1. Diagnostic System Status

- **URL:** `GET /status`
- **Description:** Returns metadata on the indexed workout dataset and dynamic sizing bounds.
- **Response (HTTP 200):**

```json
{
  "is_indexed": true,
  "total_documents": 96,
  "weeks": 3,
  "max_daily_exercises": 8,
  "suggested_k_retrieve": 26,
  "suggested_n_rerank": 13
}
```

### 2. Upload Spreadsheet Program

- **URL:** `POST /upload`
- **Headers:** `X-API-Key: <OPTIONAL_KEY>`
- **Body:** `multipart/form-data` (file: `.xlsx` file)
- **Description:** Parses spreadsheet sheets via LLM, updates local JSON cache, and rebuilds vector and sparse indices.

### 3. Multi-Turn Conversational Query

- **URL:** `POST /query`
- **Headers:** `Content-Type: application/json`
- **Body Example:**

```json
{
  "question": "How did that compare to Week 2?",
  "chat_history": [
    {
      "role": "user",
      "content": "What did I do for Bench Press in Week 1?"
    },
    {
      "role": "assistant",
      "content": "On Week 1, you benched 185 lbs for 3 sets of 8 reps."
    }
  ],
  "use_reranker": true,
  "model": "gemini-2.5-flash"
}
```

- **Response Example:**

```json
{
  "standalone_query": "What did I do for Bench Press on Week 2?",
  "answer": "On Week 2, you increased your weight on Bench Press to 195 lbs for 3 sets of 8 reps.",
  "sources": [
    {
      "content": "Week: Week 2\nDay: Day 1\nWorkout: Upper Push\nExercise: Barbell Bench Press...",
      "week": "Week 2",
      "day": "Day 1",
      "exercise_name": "barbell bench press",
      "session_name": "Upper Push"
    }
  ]
}
```

### 4. Database Reset

- **URL:** `POST /reset`
- **Description:** Programmatically wipes ChromaDB collections, clears in-memory states, triggers garbage collection to release Windows SQLite handles, and removes local JSON caches.

## System Evaluation & Benchmarking

The project includes an automated evaluation harness (`eval_pipeline.py`) that tests retrieval quality against a 20-scenario golden evaluation dataset (`eval_queries.py`).

### Key Metrics Tracked

- **Mean Reciprocal Rank (MRR):** Measures how quickly the first relevant chunk appears in retrieved candidate results.
- **Precision@K & Recall@K:** Evaluates hit ratios across variable cutoff points.
- **F1-Score Curve:** Mathematically identifies the optimal value of `K`.

### Running the Evaluation

> **Note:** Close any running `app.py` server instances before executing evaluations to release file locks on ChromaDB.

```bash
python backend/core/eval_pipeline.py
```

### Sample Output Comparison

```text
======================================================================
COMPARISON — RERANKING IMPACT
======================================================================
Metric                      No Rerank      With Rerank    Δ
Mean Reciprocal Rank (MRR)  0.8125         0.9643         +0.1518
Recall@1                    62.50%         87.50%         +25.00%
Recall@3                    81.25%         93.75%         +12.50%
Recall@5                    87.50%         100.00%        +12.50%
======================================================================
```

## Future Changes Roadmap
- [ ] **PostgreSQL + pgvector Cloud Database Migration (Neon)**
- [ ] **Upstash Redis Caching**
- [ ] **Backend Hosting (Railway)**
- [ ] **Frontend Dashboard (React)**
- [ ] **Frontend Hosting (Vercel)**
