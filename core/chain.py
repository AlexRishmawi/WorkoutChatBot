"""
RAG Chain: hybrid retriever -> LLM -> answer
"""
from collections import defaultdict
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.retrievers import EnsembleRetriever
from exercise_aliases import expand_query_aliases
from pipeline import get_exercise_history, search_by_exercise, search_sessions
from langchain_chroma import Chroma

# Inside chain.py

SYSTEM_PROMPT = """You are an elite, highly analytical strength and conditioning coach assistant. 
Your objective is to answer the athlete's question using strictly the historical training logs provided in the Context blocks.

You will receive multiple separate exercise log documents representing different weeks, days, and training sessions. 

Strict Synthesis Rules:
1. COMPREHENSIVE SCANNING: You must scan every single retrieved context record provided. Never ignore relevant context simply because it appears later or in the middle of the text payload.
2. HORIZONTAL SYNTHESIS: If multiple distinct workout sessions or weeks contain data relevant to the question, you MUST combine, cross-reference, and summarize all of them. 
3. CHRONOLOGICAL ANALYSIS: When tracking an exercise over time, organize your thought process chronologically (from oldest week to newest week) to accurately identify performance trends.
4. ABSOLUTE LITERALISM: Base your answers entirely on the exact numbers (weights, sets, reps) present in the text. Never estimate, approximate, or assume progress if it is not explicitly logged.
5. NO OUTSIDE KNOWLEDGE: If the context records do not contain data for the specific week, day, or exercise requested, state verbatim: "I cannot find that specific training log entry in the system cache."

Context Data:
{context}"""

 
_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, rewrite the follow-up "
     "question to be a standalone question that includes all necessary context "
     "(exercise names, weeks, days) from the history. "
     "If the follow-up question is already standalone, return it unchanged. "
     "Do not answer the question — only rewrite it. Return ONLY the rewritten question, "
     "no preamble, no quotes."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# Keywords that signal the user wants every instance of an exercise (history)
_HISTORY_KEYWORDS = {
    "progress", "progression", "progressed", "trend", "over time",
    "every time", "all my", "all weeks", "each week", "history",
    "show me all", "how has", "how have", "throughout", "across weeks",
    "improved", "gone up", "gone down", "pr", "personal record",
}
 
# Keywords that signal the user wants qualitative insight about one exercise
_DETAIL_KEYWORDS = {
    "note", "notes", "cue", "cues", "feel", "felt", "form",
    "best session", "worst session", "hardest", "easiest",
    "technique", "what did i write", "logged about",
}

_store: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

def _detect_exercise_term(query: str) -> str | None:
    """
    Return the first recognized exercise term found in the query,
    already normalized to its canonical name.  Returns None if no
    known exercise is detected.
    """
    query_lower = query.lower()
    # expand_query_aliases returns all synonyms for any recognized term in the query
    expansions = expand_query_aliases(query)
    if not expansions:
        return None
    # expansions[0] is always the canonical name
    return expansions[0]

def _condense_question(llm: ChatGoogleGenerativeAI, question: str, history: list) -> str:
    if not history:
        return question
    
    rewritten = llm.invoke(
        _CONDENSE_PROMPT.invoke({"question": question, "history": history})
    )
    standalone = StrOutputParser().invoke(rewritten).strip()
    print(f"[Chain] Condensed query: '{question}' -> '{standalone}'")
    return standalone

def _is_history_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _HISTORY_KEYWORDS)
 
 
def _is_detail_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _DETAIL_KEYWORDS)


def route_and_retrieve(
    query: str,
    documents: list[Document],
    vectorstore: Chroma,
    retriever: EnsembleRetriever,
) -> tuple[list[Document], str]:
    """
    Decide which retrieval strategy to use based on query intent.
 
    Returns (docs, strategy_label) so callers can log or display which
    path was taken.
 
    Decision logic
    ──────────────
    1. Recognize an exercise term in the query.
       └─ Yes + history intent  → get_exercise_history  (full scan, every match)
       └─ Yes + detail intent   → search_by_exercise    (filtered dense search)
       └─ Yes + neither/both    → search_by_exercise    (scoped is safer default
                                                          for single-exercise queries)
    2. No exercise term recognized → search_sessions    (general hybrid search)
    """
    exercise_term = _detect_exercise_term(query)
 
    if exercise_term:
        if _is_history_query(query):
            docs = get_exercise_history(documents, exercise_term)
            strategy = f"history scan (no rerank/exhaustive) -> '{exercise_term}'"
        else:
            # Covers detail queries AND ambiguous single-exercise questions
            docs = search_by_exercise(vectorstore, exercise_term)
            strategy = f"filtered dense + cross-encoder rerank -> '{exercise_term}'"
 
        # Safety fallback: if the targeted strategies return nothing
        # (exercise exists in aliases but not in this user's data),
        # drop back to general hybrid search.
        if not docs:
            docs = search_sessions(retriever, query)
            strategy += " (fallback -> hybrid + rerank)"
    else:
        docs = search_sessions(retriever, query)
        strategy = "hybrid (BM25 + dense)"
 
    return docs, strategy

def build_rag_chain(retriever: EnsembleRetriever, vectorstore: Chroma, documents: list[Document], api_key: str, model: str= "gemini-2.5-flash"):
    """
    Standard RAG Chain. Returns runnable accepting {"question": str}.
    """
    llm = ChatGoogleGenerativeAI(model=model,google_api_key=api_key, temperature=0.2)

    def run(inputs: dict) -> str:
        question = inputs["question"]
        history = inputs.get("history", [])
        standalone_question = _condense_question(llm, question, history)
        docs, strategy = route_and_retrieve(standalone_question, documents, vectorstore, retriever)
        print(f"[Chain] Strategy: {strategy} -> {len(docs)} docs retrieved")
        context = _format_docs(docs)
        answer = llm.invoke(_PROMPT.invoke({"context": context, "question": question, "history": history}))
        return StrOutputParser().invoke(answer)
    
    return RunnableLambda(run)


def build_rag_chain_with_sources(retriever: EnsembleRetriever, vectorstore: Chroma, documents: list[Document], api_key: str, model: str = "gemini-2.5-flash"):
    """
    Like build_rag_chain but also returns {"answer": str, "source_documents": list[Document]}
    """

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)

    def run(inputs: dict) -> dict:
        question = inputs["question"]
        history = inputs.get("history", [])
        standalone_question = _condense_question(llm, question, history)
        docs, strategy = route_and_retrieve(standalone_question, documents, vectorstore, retriever)
        print(f"[Chain] Strategy: {strategy} → {len(docs)} docs retrieved")
        context = _format_docs(docs)
        answer = llm.invoke(_PROMPT.invoke({"context": context, "question": question, "history": history}))
        return {
            "answer": StrOutputParser().invoke(answer),
            "source_documents": docs,
        }
    return RunnableLambda(run)


def _format_docs(docs: list[Document]) -> str:
    """
    Deduplicate by (week,day) and join into context string
    """
    seen = set()
    grouped_data = defaultdict(list)
    for doc in docs:
        content = doc.page_content.strip()
        if content in seen:
            continue
        seen.add(content)
        week = doc.metadata.get("week", "unknown_week")
        day = doc.metadata.get("day", "unknown_day")
        session_name = doc.metadata.get("session_name", "unknown_session")
        grouped_data[(week, day, session_name)].append(doc.page_content)
    
    formatted_context_blocks = []
    for (week, day, session), exercise_strings in grouped_data.items():
        block = f"=== HISTORICAL LOG: {week} | {day} | WORKOUT: {session} ===\n"
        block += "\n".join(exercise_strings)
        formatted_context_blocks.append(block)
    return "\n\n---\n\n".join(formatted_context_blocks)
