import os
import json
from langchain_core.documents import Document
from ingest import load_workout_documents
from pipeline import build_retriever, CHROMA_PERSIST_DIR

API_KEY = os.getenv("GEMINI_API_KEY")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore", "extracted_docs.json")

EVAL_DATASET = [
    {
        "query": "What did I squat in Week 1?",
        "condition": lambda doc: doc.metadata.get("week") == "Week 1" and any("squat" in ex.lower() for ex in doc.metadata.get("exercise_names", []))
    },
    {
        "query": "Show me my primary bench logs for week 2",
        "condition": lambda doc: doc.metadata.get("week") == "Week 2" and any("bench" in ex.lower() for ex in doc.metadata.get("exercise_names", []))
    },
    {
        "query": "Did I leave any notes on deadlifts?",
        "condition": lambda doc: any("deadlift" in ex.lower() for ex in doc.metadata.get("exercise_names", []))
    },
    {
        "query": "What exercises did I do on Day 1 of Week 3?",
        "condition": lambda doc: doc.metadata.get("week") == "Week 3" and "day 1" in str(doc.metadata.get("day", "")).lower()
    }
]

def run_retriever_eval():
    if not os.path.exists(CACHE_PATH):
        print(f"Error: Cache file missing at '{CACHE_PATH}'.")
        return
    
    print(f"Loading Document pool from cache...")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cached_data = json.load(f)

    all_documents = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in cached_data
    ]

    print(f"Pool contains {len(all_documents)} total docs.")

    print("Initializing EnsembleRetriever")
    retriever = build_retriever(all_documents, API_KEY, force_rebuild=False)
    total_queries = len(EVAL_DATASET)
    sum_recip_rank = 0.0
    sum_recall = 0.0

    print("\n" + "═"*70)
    print("RUNNING RETRIEVER EVALUATION ENGINE")
    print("═"*70)

    for idx, test_case in enumerate(EVAL_DATASET, start=1):
        query = test_case["query"]
        condition = test_case["condition"]

        #Scan pool to see how many matching docs
        relevant_docs = [doc for doc in all_documents if condition(doc)]
        relevant_count = len(relevant_docs)

        if relevant_count == 0:
            print(f"Skipping Test #{idx}: No docs in database match validation criteria.")
            total_queries -= 1
            continue
        #Trigger retrieval pipeline
        retrieved_docs = retriever.invoke(query)

        #Track matches landed in ranked results
        match_ranks = []
        for rank, doc in enumerate(retrieved_docs, start=1):
            if condition(doc):
                match_ranks.append(rank)
        
        #Calculate Reciprocal Rank (RR) for this query
        if match_ranks:
            first_match_rank = match_ranks[0]
            rr = 1.0 / first_match_rank
        else:
            first_match_rank = "Not Found"
            rr = 0.0
        
        sum_recip_rank += rr

        #Calculate Recall for query
        relevant_retrieved_count = len(match_ranks)
        recall = relevant_retrieved_count / relevant_count
        sum_recall += recall

        print(f"\nTest Query #{idx}: '{query}'")
        print(f"  └─ Total Relevant in DB : {relevant_count}")
        print(f"  └─ Relevant Recovered   : {relevant_retrieved_count} out of {len(retrieved_docs)} returned")
        print(f"  └─ First Match Rank     : {first_match_rank}")
        print(f"  └─ Reciprocal Rank (RR) : {rr:.4f}")
        print(f"  └─ Recall Score         : {recall * 100:.1f}%")
    
    final_mrr = sum_recip_rank / total_queries if total_queries > 0 else 0.0
    final_recall = sum_recall / total_queries if total_queries > 0 else 0.0

    print("\n" + "═"*70)
    print("📊 FINAL RETRIEVER PERFORMANCE SCOREBOARD")
    print("═"*70)
    print(f"Total Evaluated Queries : {total_queries}")
    print(f"Mean Reciprocal Rank (MRR) : {final_mrr:.4f}  (Goal: closer to 1.0)")
    print(f"Average System Recall      : {final_recall * 100:.2f}% (Goal: closer to 100%)")
    print("═"*70 + "\n")

if __name__ == "__main__":
    run_retriever_eval()
    