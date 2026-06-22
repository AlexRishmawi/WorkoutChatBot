"""
main.py
-------
Entry point for Workout RAG v2.

Startup has two phases:
  1. EXTRACT  — LLM reads each sheet's raw cells and produces structured Documents
                (only runs when no cached documents exist, or --rebuild is passed)
  2. INDEX    — Documents are embedded into Chroma + indexed in BM25

Then the app enters interactive Q&A mode (or answers a single --question).

Usage
-----
  python main.py --xlsx path/to/program.xlsx --api-key YOUR_KEY
  python main.py --xlsx path/to/program.xlsx --api-key YOUR_KEY --question "What did I squat in Week 1?"
  python main.py --xlsx path/to/program.xlsx --api-key YOUR_KEY --sources   # show retrieved docs
  python main.py --xlsx path/to/program.xlsx --api-key YOUR_KEY --rebuild   # re-extract + re-embed
"""

import argparse
import json
import os
import sys
from ingest import load_workout_documents
from pipeline import build_retriever
from chain import build_rag_chain, build_rag_chain_with_sources, get_session_history
from langchain_core.runnables import RunnableWithMessageHistory


CACHE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore", "extracted_docs.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Workout RAG v2 — LLM-powered xlsx ingestion")
    p.add_argument("--xlsx",      required=True,  help="Path to workout .xlsx file")
    p.add_argument("--api-key",   required=False, help="Google GenAI API key (or set GOOGLE_API_KEY)")
    p.add_argument("--model",     default="gemini-2.5-flash")
    p.add_argument("--question",  default=None,   help="Single question (non-interactive)")
    p.add_argument("--sources",   action="store_true", help="Show retrieved source sessions")
    p.add_argument("--rebuild",   action="store_true", help="Force re-extraction and re-embedding")
    return p.parse_args()


def resolve_api_key(args_key):
    key = args_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Error: provide --api-key or set GOOGLE_API_KEY.")
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# Document caching
# (Extraction is the expensive LLM step — cache to avoid re-calling on restart)
# ---------------------------------------------------------------------------

def _save_doc_cache(documents):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[main] Extraction cache saved to '{CACHE_PATH}'")


def _load_doc_cache():
    if not os.path.exists(CACHE_PATH):
        return None
    from langchain_core.documents import Document
    with open(CACHE_PATH) as f:
        data = json.load(f)
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    print(f"[main] Loaded {len(docs)} documents from extraction cache.")
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    api_key = resolve_api_key(args.api_key)

    print(f"\n{'='*55}")
    print(f"  Workout RAG v2 — LLM-Powered xlsx Ingestion")
    print(f"{'='*55}\n")

    # Phase 1: Extract (or load cache)
    cached = _load_doc_cache() if not args.rebuild else None
    if cached:
        documents = cached
    else:
        print("[main] Phase 1: Extracting workout data via LLM...")
        documents = load_workout_documents(args.xlsx, api_key)
        _save_doc_cache(documents)

    if not documents:
        print("No workout sessions could be extracted. Check your xlsx file.")
        sys.exit(1)

    # Phase 2: Build retriever
    print("\n[main] Phase 2: Building hybrid retriever...")
    retriever, vectorstore = build_retriever(
        documents=documents,
        api_key=api_key,
        force_rebuild=args.rebuild,
    )

    # Phase 3: Build RAG chain
    if args.sources:
        chain = build_rag_chain_with_sources(retriever, vectorstore, documents, api_key, model=args.model)
    else:
        chain = build_rag_chain(retriever, vectorstore, documents, api_key, model=args.model)


    chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer" if args.sources else None
    )

    print("\nReady. Ask anything about your workout program. Type 'quit' to exit.\n")

    # Single question mode
    if args.question:
        _ask(chain, args.question, show_sources=args.sources)
        return

    # Interactive REPL
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        _ask(chain, question, show_sources=args.sources)
        print()


def _ask(chain, question, session_id="cli-session", show_sources=False):
    print("\nAssistant: ", end="", flush=True)
    result = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}})
    if isinstance(result, dict):
        print(result["answer"])
        if show_sources:
            _print_sources(result["source_documents"])
    else:
        print(result)


def _print_sources(docs):
    if not docs:
        return
    print("\n── Retrieved Sessions ──────────────────────────────")
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        print(f"\n[{i}] {m.get('week')} | {m.get('day')} — {m.get('session_name', '')}")
        print(f"    Exercises: {m.get('exercise_names_str', 'N/A')}")
        print(f"    Original:  {m.get('original_exercise_name', 'N/A')}")
        print(f"    Strategy:  routed by chain intent detection")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()