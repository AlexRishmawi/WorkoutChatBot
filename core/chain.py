"""
RAG Chain: hybrid retriever -> LLM -> answer
"""
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

SYSTEM_PROMPT = """You are a knowledgeable fitness coach assistant.
You have access to a workout program extracted from a spreadsheet.
The data may include prescribed sets/reps, actual weights logged, actual reps performed,
and any notes the athlete recorded.
 
Answer the user's question using ONLY the context provided below.
Be specific with numbers when available. Distinguish between prescribed targets
(e.g. "3 x 6-8") and what was actually logged (e.g. "weight: 65 lbs, reps: 3x10").
If the context doesn't contain enough information, say so clearly.

Context:
{context}"""
 
_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def build_rag_chain(retriever: EnsembleRetriever, api_key: str, model: str= "gemini-2.5-flash"):
    """
    Standard RAG Chain. Returns runnable accepting {"question": str}.
    """
    llm = ChatGoogleGenerativeAI(model=model,google_api_key=api_key, temperature=0.2)

    return (
        {
            "context": itemgetter("question") | retriever | RunnableLambda(_format_docs),
            "question": itemgetter("question"),
        }
        | _PROMPT
        | llm
        | StrOutputParser()
    )


def build_rag_chain_with_sources(retriever: EnsembleRetriever, api_key: str, model: str = "gemini-2.5-flash"):
    """
    Like build_rag_chain but also returns {"answer": str, "source_documents": list[Document]}
    """

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)

    def retrieve_and_answer(inputs: dict) -> dict:
        question = inputs["question"]
        docs = retriever.invoke(question)
        context = _format_docs(docs)
        answer = llm.invoke(_PROMPT.invoke({"context": context, "question": question}))
        return {
            "answer":       StrOutputParser().invoke(answer),
            "source_documents":     docs,
        }
    return RunnableLambda(retrieve_and_answer)


def _format_docs(docs: list[Document]) -> str:
    """
    Deduplicate by (week,day) and join into context string
    """
    seen, unique = set(), []
    for doc in docs:
        key = (doc.metadata.get("week"), doc.metadata.get("day"))
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return "\n\n---\n\n".join(d.page_content for d in unique)
