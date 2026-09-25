import os
from pathlib import Path
from typing import TypedDict, List

import chromadb
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from ingest import DB, COLLECTION_NAME, MODEL_NAME, build_collection

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
]

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support assistant.

CONTEXT:
Use only the retrieved Zepto policy context supplied below.
Retrieved context:
{context}

TASK:
Answer the user's question using only the supplied context.

FORMAT:
Return a concise answer and identify the source document IDs.

LENGTH:
Keep the answer short and directly useful.

NEGATIVE CONSTRAINT:
Do not answer using information not present in the provided context.
Do not invent Zepto policies.

FEW-SHOT EXAMPLE:
User: "How long do refunds take?"
Context: "Approved refunds are credited to the original payment method within 3–5 business days..."
Answer: "Approved refunds reach the original payment method within 3–5 business days."
"""

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: List[dict]
    answer: str
    sources: List[str]
    confidence: float


class SupportResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class SupportRequest(BaseModel):
    query: str


model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=str(DB))

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = build_collection()


def classify_intent(state: SupportState):
    query = state["query"].lower()
    intent = "policy_question" if any(k in query for k in KEYWORDS) else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: SupportState):
    q_embedding = model.encode([state["query"]], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=q_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    docs = result["documents"][0]
    ids = result["ids"][0]
    retrieved = [
        {"id": ids[i], "text": docs[i], "distance": result["distances"][0][i]}
        for i in range(len(docs))
    ]

    top = retrieved[0]
    if MOCK_LLM:
        answer = f"Based on the retrieved context: {top['text'][:200]}"
    else:
        answer = real_llm_answer(state["query"], retrieved)

    return {
        "retrieved": retrieved,
        "answer": answer,
        "sources": [x["id"] for x in retrieved],
        "confidence": 1.0 if MOCK_LLM else 0.9
    }


def direct_answer(state: SupportState):
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = real_llm_answer(state["query"], [])
    return {"answer": answer, "sources": [], "confidence": 1.0 if MOCK_LLM else 0.9}


def real_llm_answer(query, retrieved):
    """
    Optional extension for MOCK_LLM=0.
    Uses a Groq-compatible OpenAI endpoint if GROQ_API_KEY is supplied.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "ERROR: MOCK_LLM=0 requires GROQ_API_KEY for the optional real-LLM path."

    from openai import OpenAI
    client_llm = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    )

    context = "\n\n".join(f"[{x['id']}] {x['text']}" for x in retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context) + f"\n\nUser question: {query}"

    last_error = None
    for attempt in range(3):
        try:
            response = client_llm.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = response.choices[0].message.content or ""
            # The optional branch is kept intentionally simple; final API
            # validation is still enforced by SupportResponse.
            return text
        except Exception as exc:
            last_error = exc
            prompt += "\nCorrective instruction: return only a concise grounded answer."
    return f"ERROR: real LLM generation failed after 3 attempts: {last_error}"


def route(state: SupportState):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def make_graph():
    builder = StateGraph(SupportState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)

    builder.set_entry_point("classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )
    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)
    return builder.compile()


graph = make_graph()


def ask(query: str) -> SupportResponse:
    state = graph.invoke({"query": query})
    return SupportResponse(
        answer=state["answer"],
        sources=state.get("sources", []),
        confidence=float(state.get("confidence", 1.0))
    )
