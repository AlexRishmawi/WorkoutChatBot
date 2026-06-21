import os
import json
from langchain_core.documents import Document
from ingest import load_workout_documents
from pipeline import build_retriever, search_sessions, get_program_size, CHROMA_PERSIST_DIR
from collections import defaultdict
from eval_queries import EVAL_DATASET

API_KEY = os.getenv("GEMINI_API_KEY")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore", "extracted_docs.json")

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
    program_size = get_program_size(all_documents)
    print(f"[eval] Using dynamic sizing: k_retrieve={program_size['k_retrieve']}, n_rerank={program_size['n_rerank']}")
 
    # Run the full eval twice: once with the raw ensemble (no rerank),
    # once through search_sessions with cross-encoder reranking enabled.
    # This makes the MRR/recall delta from reranking directly visible.
    print("\n" + "═" * 70)
    print("PASS 1 — BM25 + Dense ENSEMBLE ONLY (no reranking)")
    print("═" * 70)
    baseline_mrr, baseline_recall_at_k = _run_eval_pass(
        all_documents,
        retrieve_fn=lambda query: search_sessions(retriever, query, use_reranker=False),
    )
 
    print("\n" + "═" * 70)
    print("PASS 2 — ENSEMBLE + CROSS-ENCODER RERANK")
    print("═" * 70)
    reranked_mrr, reranked_recall_at_k = _run_eval_pass(
        all_documents,
        retrieve_fn=lambda query: search_sessions(retriever, query, use_reranker=True),
    )
 
    print("\n" + "═" * 70)
    print("COMPARISON — RERANKING IMPACT")
    print("═" * 70)
    print(f"{'Metric':<28}{'No Rerank':<15}{'With Rerank':<15}{'Δ'}")
    print(f"{'Mean Reciprocal Rank (MRR)':<28}{baseline_mrr:<15.4f}{reranked_mrr:<15.4f}{reranked_mrr - baseline_mrr:+.4f}")
    
    shared_ks = sorted(set(baseline_recall_at_k) & set(reranked_recall_at_k))
    for k in shared_ks:
        b = baseline_recall_at_k[k]
        r = reranked_recall_at_k[k]
        print(f"{f'Recall@{k}':<28}{b*100:<14.2f}%{r*100:<14.2f}%{(r - b)*100:+.2f}%")
    print("═" * 70 + "\n")
 
 
def _run_eval_pass(all_documents: list[Document], retrieve_fn, fixed_ks: list[int] = (1,3,5,10,15,20)) -> tuple[float, dict[int, float]]:
    """
    Runs EVAL_DATASET through `retrieve_fn(query) -> list[Document]` and
    returns (mrr, average_recall) for that pass.
    """
    total_queries = len(EVAL_DATASET)
    sum_recip_rank = 0.0

    global_k_metrics = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})

 
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
        num_retrieved = len(retrieved_docs)
 
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

        """
        print(f"\nTest Query #{idx}: '{query}'")
        print(f"  ├─ Total Relevant in DB : {relevant_count}")
        print(f"  ├─ Total Returned       : {num_retrieved}")
        print(f"  ├─ First Match Rank     : {first_match_rank} (Reciprocal Rank: {rr:.4f})")
        print(f"  └─ Step-by-Step Metric Curve (K=1 to K={num_retrieved}):")
        print(f"     ┌──────┬───────────┬───────────┬───────────┐")
        print(f"     │  K   │ Precision │  Recall   │ F1-Score  │")
        print(f"     ├──────┼───────────┼───────────┼───────────┤")
        """
        for k in range(1, num_retrieved + 1):
            retrieved_at_k = retrieved_docs[:k]
            matches_at_k = sum(1 for doc in retrieved_at_k if condition(doc))
            precision_at_k = matches_at_k / k
            recall_at_k = matches_at_k / relevant_count if relevant_count > 0 else 0.0

            if(precision_at_k + recall_at_k) > 0:
                f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
            else:
                f1_at_k = 0.0
            global_k_metrics[k]["precision"].append(precision_at_k)
            global_k_metrics[k]["recall"].append(recall_at_k)
            global_k_metrics[k]["f1"].append(f1_at_k)
            # print(f"     │  {k:<3} │   {precision_at_k*100:5.1f}%  │   {recall_at_k*100:5.1f}%  │   {f1_at_k*100:5.1f}%  │")

        # print(f"     └──────┴───────────┴───────────┴───────────┘")

 
    final_mrr = sum_recip_rank / total_queries if total_queries > 0 else 0.0
    print("\n" + "═"*90)
    print("SYSTEM PERFORMANCE SUMMARY DASHBOARD (MACRO AVERAGES)")
    print("═"*90)
    print(f"  Total Queries Evaluated : {total_queries}")
    print(f"  Mean Reciprocal Rank (MRR): {final_mrr:.4f}\n")
    print(f"  Performance Curve Across Cutoffs:")
    print(f"  ┌──────┬───────────────────┬───────────────────┬───────────────────┐")
    print(f"  │  K   │ Average Precision │  Average Recall   │ Average F1-Score  │")
    print(f"  ├──────┼───────────────────┼───────────────────┼───────────────────┤")
 
    for k in sorted(global_k_metrics.keys()):
        p_avg = sum(global_k_metrics[k]["precision"]) / len(global_k_metrics[k]["precision"])
        r_avg = sum(global_k_metrics[k]["recall"]) / len(global_k_metrics[k]["recall"])
        f1_avg = sum(global_k_metrics[k]["f1"]) / len(global_k_metrics[k]["f1"])
        print(f"  │  {k:<2}  │       {p_avg*100:5.1f}%       │       {r_avg*100:5.1f}%       │       {f1_avg*100:5.1f}%       │")

    print(f"  └──────┴───────────────────┴───────────────────┴───────────────────┘")
    print("  Note: Watch where the F1-Score peaks. This is your mathematically optimal K.")
    print("═"*90 + "\n")

    recall_at_k = {
        k: sum(global_k_metrics[k]["recall"]) / len(global_k_metrics[k]["recall"])
        for k in fixed_ks
        if k in global_k_metrics
    }

    return final_mrr, recall_at_k

if __name__ == "__main__":
    run_retriever_eval()
    