-- ============================================================
-- Workout RAG Database Schema
-- Neon PostgreSQL + pgvector
-- ============================================================

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- Users
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    email TEXT UNIQUE NOT NULL,

    name TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Workout Programs
-- ============================================================

CREATE TABLE IF NOT EXISTS workout_programs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    program_name TEXT NOT NULL,

    original_filename TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Exercise Chunks
-- ============================================================

CREATE TABLE IF NOT EXISTS exercise_chunks (

    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    program_id UUID
        REFERENCES workout_programs(id)
        ON DELETE CASCADE,

    page_content TEXT NOT NULL,

    week TEXT,

    day TEXT,

    session_name TEXT,

    exercise_name TEXT,

    metadata JSONB,

    embedding VECTOR(768)

);

-- ============================================================
-- Chat Sessions
-- ============================================================

CREATE TABLE IF NOT EXISTS chat_sessions (

    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID
        REFERENCES users(id)
        ON DELETE CASCADE,

    program_id UUID
        REFERENCES workout_programs(id)
        ON DELETE CASCADE,

    created_at TIMESTAMPTZ DEFAULT NOW()

);

-- ============================================================
-- Messages
-- ============================================================

CREATE TABLE IF NOT EXISTS messages (

    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    session_id UUID
        REFERENCES chat_sessions(id)
        ON DELETE CASCADE,

    role TEXT NOT NULL,

    content TEXT NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW()

);

-- ============================================================
-- Helpful Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_program_user
ON workout_programs(user_id);

CREATE INDEX IF NOT EXISTS idx_chunks_program
ON exercise_chunks(program_id);

CREATE INDEX IF NOT EXISTS idx_chat_user
ON chat_sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_messages_session
ON messages(session_id);

-- ============================================================
-- pgvector Index
-- ============================================================

CREATE INDEX IF NOT EXISTS exercise_embedding_idx
ON exercise_chunks
USING hnsw (embedding vector_cosine_ops);