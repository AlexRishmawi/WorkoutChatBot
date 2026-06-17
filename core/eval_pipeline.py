import os
import json
from langchain_core.documents import Document
from ingest import load_workout_documents
from pipeline import build_retriever, CHROMA_PERSIST_DIR
from ingest import normalize_exercise_name
from ingest import get_all_aliases
from pipeline import search_sessions

API_KEY = os.getenv("GEMINI_API_KEY")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore", "extracted_docs.json")

def _exercise_match(doc: Document, exercise_query: str) -> bool:
    canonical = normalize_exercise_name(exercise_query)
    all_aliases = set(a.lower() for a in get_all_aliases(canonical))

    raw = doc.metadata.get("exercise_names", [])
    if isinstance(raw, str):
        exercise_names = [ex.strip() for ex in raw.split(",") if ex.strip()]
    else:
        exercise_names = list(raw)

    doc_exercises = set(ex.lower() for ex in exercise_names)

    if all_aliases & doc_exercises:
        return True

    return any(exercise_query.lower() in ex for ex in doc_exercises)


EVAL_DATASET = [
    {
        "query": "What did I squat in Week 1?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 1"
            and _exercise_match(doc, "squat")
        ),
    },
    {
        "query": "Show me my primary bench logs for week 2",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 2"
            and _exercise_match(doc, "bench press")
        ),
    },
    {
        "query": "Did I leave any notes on deadlifts?",
        "condition": lambda doc: _exercise_match(doc, "deadlift"),
    },
    {
        "query": "What exercises did I do on Day 1 of Week 3?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 3"
            and "day 1" in str(doc.metadata.get("day", "")).lower()
        ),
    },
]



def run_retriever_eval():
    if not os.path.exists(CACHE_PATH):
        print(f"Error: Cache file missing at '{CACHE_PATH}'.")
        return
 
    print("Loading document pool from cache...")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cached_data = json.load(f)
 
    all_documents = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in cached_data
    ]
    print(f"Pool contains {len(all_documents)} total docs.")
 
    print("Initializing EnsembleRetriever...")
    # build_retriever returns (ensemble, vectorstore) — only the ensemble
    # is needed here since search_sessions wraps retriever.invoke().
    retriever, _ = build_retriever(all_documents, API_KEY, force_rebuild=False)
 
    # Run the full eval twice: once with the raw ensemble (no rerank),
    # once through search_sessions with cross-encoder reranking enabled.
    # This makes the MRR/recall delta from reranking directly visible.
    print("\n" + "═" * 70)
    print("PASS 1 — BM25 + Dense ENSEMBLE ONLY (no reranking)")
    print("═" * 70)
    baseline_mrr, baseline_recall = _run_eval_pass(
        all_documents,
        retrieve_fn=lambda query: search_sessions(retriever, query, use_reranker=False),
    )
 
    print("\n" + "═" * 70)
    print("PASS 2 — ENSEMBLE + CROSS-ENCODER RERANK (top 10)")
    print("═" * 70)
    reranked_mrr, reranked_recall = _run_eval_pass(
        all_documents,
        retrieve_fn=lambda query: search_sessions(retriever, query, use_reranker=True, top_n=10),
    )
 
    print("\n" + "═" * 70)
    print("📊 COMPARISON — RERANKING IMPACT")
    print("═" * 70)
    print(f"{'Metric':<28}{'No Rerank':<15}{'With Rerank':<15}{'Δ'}")
    print(f"{'Mean Reciprocal Rank (MRR)':<28}{baseline_mrr:<15.4f}{reranked_mrr:<15.4f}{reranked_mrr - baseline_mrr:+.4f}")
    print(f"{'Average Recall':<28}{baseline_recall*100:<14.2f}%{reranked_recall*100:<14.2f}%{(reranked_recall - baseline_recall)*100:+.2f}%")
    print("═" * 70 + "\n")
 
 
def _run_eval_pass(all_documents: list[Document], retrieve_fn) -> tuple[float, float]:
    """
    Runs EVAL_DATASET through `retrieve_fn(query) -> list[Document]` and
    returns (mrr, average_recall) for that pass.
    """
    total_queries = len(EVAL_DATASET)
    sum_recip_rank = 0.0
    sum_recall = 0.0
 
    for idx, test_case in enumerate(EVAL_DATASET, start=1):
        query = test_case["query"]
        condition = test_case["condition"]
 
        # Count how many docs in the pool should match this query
        relevant_docs = [doc for doc in all_documents if condition(doc)]
        relevant_count = len(relevant_docs)
 
        if relevant_count == 0:
            print(
                f"\nTest Query #{idx}: '{query}'\n"
                f"  └─ SKIPPED: No docs in database match validation criteria."
            )
            total_queries -= 1
            continue
 
        # Run retrieval (caller decides whether reranking is applied)
        retrieved_docs = retrieve_fn(query)
 
        # Find which retrieved docs satisfy the condition and at what rank
        match_ranks = [
            rank
            for rank, doc in enumerate(retrieved_docs, start=1)
            if condition(doc)
        ]
 
        # Reciprocal Rank
        if match_ranks:
            first_match_rank = match_ranks[0]
            rr = 1.0 / first_match_rank
        else:
            first_match_rank = "Not Found"
            rr = 0.0
 
        sum_recip_rank += rr
 
        # Recall
        relevant_retrieved_count = len(match_ranks)
        recall = relevant_retrieved_count / relevant_count
        sum_recall += recall
 
        print(f"\nTest Query #{idx}: '{query}'")
        print(f"  └─ Total Relevant in DB : {relevant_count}")
        print(f"  └─ Relevant Recovered   : {relevant_retrieved_count} / {len(retrieved_docs)} returned")
        print(f"  └─ First Match Rank     : {first_match_rank}")
        print(f"  └─ Reciprocal Rank (RR) : {rr:.4f}")
        print(f"  └─ Recall Score         : {recall * 100:.1f}%")
 
    final_mrr = sum_recip_rank / total_queries if total_queries > 0 else 0.0
    final_recall = sum_recall / total_queries if total_queries > 0 else 0.0
 
    print(f"\n  Pass MRR: {final_mrr:.4f}  |  Pass Recall: {final_recall * 100:.2f}%")
 
    return final_mrr, final_recall

if __name__ == "__main__":
    run_retriever_eval()
    