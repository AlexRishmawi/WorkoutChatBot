# Workout Chatbot — Production Implementation Plan

A RAG-powered chatbot that lets users upload a workout spreadsheet and ask questions about their plan. Built on FastAPI, Next.js, Neon, Cloudflare R2, Clerk, Upstash Redis, Railway, and Vercel.

---

## Phase 1 — Core RAG Pipeline (Local Environment)

**Goal:** Prove the AI pipeline works end-to-end before touching any cloud service.

### What you're building

- A FastAPI backend that can ingest a spreadsheet, chunk its content, embed it, store the vectors, and answer questions against them.
- A local PostgreSQL instance with the `pgvector` extension for testing vector search.

### Steps

1. **Set up FastAPI** — scaffold your project with a virtual environment and install core dependencies: `fastapi`, `uvicorn`, `openai`, `psycopg2-binary`, `pgvector`, `openpyxl` or `pandas` for spreadsheet parsing.

2. **Spreadsheet ingestion** — write a Python function that reads an uploaded `.xlsx` or `.csv`, extracts rows into text chunks (e.g. one chunk per exercise or per day), and attaches metadata like sheet name or row index.

3. **Embedding** — call OpenAI's `text-embedding-3-small` model on each chunk. Store the resulting vectors in your local `pgvector` table alongside the raw chunk text and metadata.

4. **Retrieval** — write a search function that takes a user question, embeds it, and runs a cosine similarity query against your vector table to retrieve the top-k relevant chunks.

5. **Generation** — pass the retrieved chunks as context to `gpt-4o` (or your chosen model) with a system prompt that grounds it in the spreadsheet data. Stream the response back via a FastAPI `StreamingResponse`.

6. **Test locally** — upload a sample workout spreadsheet and verify that questions like "How many sets of squats do I do on Wednesday?" return accurate answers.

### Key files

```
backend/
  main.py          # FastAPI app and route definitions
  ingest.py        # Spreadsheet parsing, chunking, embedding, upsert
  search.py        # Vector similarity query
  generate.py      # Context assembly and LLM call
  db.py            # pgvector connection and schema
```

### Environment variables (local)

```
OPENAI_API_KEY=
DATABASE_URL=postgresql://localhost:5432/workout_chatbot
```

---

## Phase 2 — Cloud Infrastructure (Neon + Cloudflare R2)

**Goal:** Replace local storage with production cloud services without changing application logic.

### What you're building

- A **Neon** serverless Postgres database with `pgvector` as the production vector store.
- A **Cloudflare R2** bucket as the file store for raw uploaded spreadsheets.

### Steps

1. **Provision Neon** — create a project at [neon.tech](https://neon.tech), copy the connection string, and run your schema migration to enable `pgvector` and create the embeddings table.

2. **Provision Cloudflare R2** — create a bucket in the Cloudflare dashboard. Generate an API token with R2 read/write permissions. Note your account ID, bucket name, access key, and secret key.

3. **Add the R2 upload step** — before chunking and embedding, upload the raw file to R2 using the `boto3` client (R2 is S3-compatible). Store the R2 object key in your database row so you can retrieve the original file later if needed.

4. **Update environment variables** — swap your local `DATABASE_URL` for the Neon connection string. Add R2 credentials.

5. **End-to-end test** — upload a file, confirm it lands in R2, confirm vectors land in Neon, confirm a query returns the right answer.

### Updated environment variables

```
OPENAI_API_KEY=
DATABASE_URL=postgresql://<user>:<pass>@<neon-host>/workout_chatbot?sslmode=require
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
```

### Ingestion flow (updated)

```
Upload spreadsheet
  → Upload raw file to R2 (store object key)
  → Parse spreadsheet into chunks
  → Embed chunks via OpenAI
  → Upsert vectors + metadata into Neon
```

---

## Phase 3 — Next.js UI and Authentication (Frontend Setup)

**Goal:** Build the user-facing interface with auth so only logged-in users can access their data.

### What you're building

- A **Next.js 14** app with the App Router.
- **Clerk** for sign-up, sign-in, and session management.
- A chat interface with a scrolling message history, an input bar, and a file upload button.

### Steps

1. **Scaffold Next.js** — run `npx create-next-app@latest` with TypeScript and Tailwind. Install Clerk: `npm install @clerk/nextjs`.

2. **Configure Clerk** — wrap your app in `<ClerkProvider>` in `layout.tsx`. Add the Clerk middleware to protect routes. Create sign-in and sign-up pages at `/sign-in` and `/sign-up`.

3. **Build the chat layout** — a two-panel layout: a sidebar for navigation/file upload and a main panel for the conversation. The main panel has a scrollable message list and a sticky input bar at the bottom.

4. **Build the file upload component** — a drag-and-drop zone that accepts `.xlsx` and `.csv` files, displays upload progress, and shows a confirmation once the file has been processed.

5. **Build the message components** — a `UserMessage` component and an `AssistantMessage` component. The assistant message should support streaming (characters appear as they arrive).

### Key components

```
app/
  layout.tsx            # ClerkProvider wrapper
  page.tsx              # Redirect to /chat
  chat/
    page.tsx            # Main chat page (protected)
components/
  ChatWindow.tsx        # Scrolling message history
  MessageInput.tsx      # Input bar with send button
  FileUpload.tsx        # Drag-and-drop upload
  UserMessage.tsx
  AssistantMessage.tsx
```

### Environment variables (frontend)

```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Phase 4 — API Integration (Bridge Frontend and Backend)

**Goal:** Connect the Next.js frontend to the FastAPI backend securely, with real-time streaming.

### What you're building

- FastAPI endpoints that validate the Clerk JWT on every request.
- A streaming chat endpoint that the frontend subscribes to with `EventSource` or `fetch` + `ReadableStream`.
- A file upload endpoint that accepts the spreadsheet, runs ingestion, and returns a confirmation.

### Steps

1. **Validate Clerk JWTs in FastAPI** — install `PyJWT` and `httpx`. Write a FastAPI dependency (`get_current_user`) that reads the `Authorization: Bearer <token>` header, fetches Clerk's JWKS endpoint, and verifies the token. Attach the decoded user ID to every request.

2. **Scope data by user** — add a `user_id` column to your embeddings table. On ingest, tag every chunk with the authenticated user's ID. On search, filter by `user_id` so users only see their own data.

3. **Streaming endpoint** — create `POST /chat` that accepts `{ message: string, session_id: string }`, runs retrieval, streams the LLM response as Server-Sent Events (SSE), and closes the stream when done.

4. **Upload endpoint** — create `POST /upload` that accepts a `multipart/form-data` request, runs the full ingestion pipeline, and returns `{ status: "ok", chunks_indexed: N }`.

5. **Wire the frontend** — in `MessageInput.tsx`, on submit, call `POST /chat` with the Clerk session token in the `Authorization` header. Read the SSE stream and append tokens to the assistant message in real time. In `FileUpload.tsx`, call `POST /upload` similarly.

### FastAPI route summary

```
POST /upload          # Ingest a spreadsheet (auth required)
POST /chat            # Streaming chat query (auth required)
GET  /health          # Public health check
```

### JWT validation dependency (sketch)

```python
from fastapi import Depends, HTTPException, Header
import jwt, httpx

async def get_current_user(authorization: str = Header(...)):
    token = authorization.removeprefix("Bearer ")
    # Fetch JWKS from Clerk and verify signature
    # Raise HTTPException(401) if invalid
    return decoded_payload["sub"]  # Clerk user ID
```

---

## Phase 5 — Caching and Deployment (Optimization and Launch)

**Goal:** Add performance caching, deploy the backend to Railway, and deploy the frontend to Vercel.

### What you're building

- **Upstash Redis** for chat history caching and session state (so conversations persist across page refreshes).
- A production Railway deployment for FastAPI.
- A production Vercel deployment for Next.js.

### Steps

#### Upstash Redis

1. Create a Redis database at [upstash.com](https://upstash.com). Copy the REST URL and token.

2. In FastAPI, install `upstash-redis`. On each chat request, load the session's message history from Redis (keyed by `user_id:session_id`), append the new exchange, and save it back. Pass the history to the LLM for multi-turn context.

3. Optionally cache embeddings for recently uploaded files to avoid re-embedding on repeated ingestion of the same content.

#### Railway (FastAPI backend)

1. Push your backend to a GitHub repo. Connect it to Railway and create a new service pointing at the repo.

2. Add all environment variables in the Railway dashboard: `OPENAI_API_KEY`, `DATABASE_URL`, `R2_*`, `CLERK_*`, `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`.

3. Set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.

4. Deploy and note the public Railway URL (e.g. `https://workout-chatbot-api.up.railway.app`).

#### Vercel (Next.js frontend)

1. Push your frontend to a GitHub repo. Import it into Vercel.

2. Add all environment variables in the Vercel dashboard. Update `NEXT_PUBLIC_API_URL` to your Railway backend URL.

3. Deploy. Vercel auto-detects Next.js and handles the build.

### Final environment variables (production)

```
# Backend (Railway)
OPENAI_API_KEY=
DATABASE_URL=                          # Neon connection string
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
CLERK_SECRET_KEY=
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=

# Frontend (Vercel)
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=
NEXT_PUBLIC_API_URL=                   # Railway backend URL
```

---

## Full Architecture Summary

```
User browser (Vercel — Next.js)
  ↕ HTTPS + Clerk JWT
FastAPI backend (Railway)
  ├── Cloudflare R2         (raw file storage)
  ├── Neon + pgvector       (vector embeddings + metadata)
  ├── OpenAI API            (embeddings + chat completions)
  └── Upstash Redis         (chat history + session cache)
```

---

## Recommended Libraries

| Purpose | Library |
|---|---|
| Spreadsheet parsing | `openpyxl`, `pandas` |
| Vector embeddings | `openai` (`text-embedding-3-small`) |
| Vector store client | `psycopg2-binary`, `pgvector` |
| R2 file upload | `boto3` (S3-compatible) |
| JWT validation | `PyJWT`, `httpx` |
| Redis caching | `upstash-redis` |
| Frontend auth | `@clerk/nextjs` |
| Streaming UI | Native `fetch` + `ReadableStream` |