"""
LangGraph node and edge functions.

Nodes  — transform state (retrieve, grade, generate, etc.)
Edges  — return a routing string that drives graph branching
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .state import AgentState
from ..config import config

if TYPE_CHECKING:
    from ..retrieval.hybrid_search import HybridRetriever

# ── Module-level retriever (set by main.py after warm-up) ─────────────────────
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        from ..retrieval.hybrid_search import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


# ── LLM ───────────────────────────────────────────────────────────────────────
_llm = ChatOllama(model=config.ollama_model, temperature=config.temperature)


# ── Pydantic schemas for structured output ────────────────────────────────────

class RouteQuery(BaseModel):
    """Route a query to the most appropriate datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="'vectorstore' for document questions; 'web_search' for current events."
    )


class GradeDocument(BaseModel):
    """Binary relevance score for a single retrieved document."""
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise."
    )


class GradeHallucination(BaseModel):
    """Binary score: is the generation grounded in the retrieved documents?"""
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer is grounded in the facts, 'no' if it hallucinated."
    )


class GradeAnswer(BaseModel):
    """Binary score: does the generation actually answer the question?"""
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer addresses the question, 'no' otherwise."
    )


# ── Structured LLM chains ─────────────────────────────────────────────────────
_router_llm      = _llm.with_structured_output(RouteQuery)
_doc_grader_llm  = _llm.with_structured_output(GradeDocument)
_hall_grader_llm = _llm.with_structured_output(GradeHallucination)
_ans_grader_llm  = _llm.with_structured_output(GradeAnswer)


# ── Prompts ───────────────────────────────────────────────────────────────────

_ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert query router. Route to 'vectorstore' for questions about "
     "ingested documents. Route to 'web_search' for current events or topics "
     "not covered in the documents. Return only the datasource name."),
    ("human", "{question}"),
])

_DOC_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader. Assess whether the retrieved document contains "
     "information useful for answering the user question. "
     "Score 'yes' if relevant, 'no' if not."),
    ("human",
     "Retrieved document:\n\n{document}\n\nUser question: {question}"),
])

_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a hallucination detector. Check whether the LLM answer is fully "
     "grounded in the provided source documents. "
     "Score 'yes' if grounded, 'no' if the answer contains unsupported claims."),
    ("human",
     "Source documents:\n\n{documents}\n\nLLM answer: {generation}"),
])

_ANSWER_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an answer quality grader. Decide whether the generated answer "
     "actually resolves the user's question. "
     "Score 'yes' if it does, 'no' if it is off-topic or incomplete."),
    ("human",
     "User question: {question}\n\nGenerated answer: {generation}"),
])

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for question-answering tasks. "
     "Use ONLY the following retrieved context to answer the question. "
     "If the context does not contain enough information, say you don't know. "
     "Keep the answer concise (3–5 sentences).\n\nContext:\n{context}"),
    ("human", "{question}"),
])

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query optimiser. Rewrite the user question to improve "
     "vectorstore retrieval — sharpen the semantic intent without changing meaning."),
    ("human",
     "Original question: {question}\n\nProvide an improved version of the question."),
])


# ══════════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS  (each returns a partial state dict)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(state: AgentState) -> dict:
    """Hybrid retrieval: dense + BM25 → RRF → cross-encoder reranking."""
    print("\n--- NODE: RETRIEVE ---")
    documents = get_retriever().retrieve(state["question"])
    return {"documents": documents}


def grade_documents(state: AgentState) -> dict:
    """Grade each retrieved document for relevance; flag if web search needed."""
    print("\n--- NODE: GRADE DOCUMENTS ---")
    question  = state["question"]
    documents = state["documents"]

    chain = _DOC_GRADE_PROMPT | _doc_grader_llm

    filtered: list[str] = []
    web_search_needed = "No"

    for i, doc in enumerate(documents):
        try:
            score = chain.invoke({"question": question, "document": doc})
            grade = score.binary_score
        except Exception:
            grade = "yes"   # fail-safe: keep the doc if grading errors

        marker = "✓" if grade == "yes" else "✗"
        print(f"  doc_{i} → {grade} {marker}")

        if grade == "yes":
            filtered.append(doc)
        else:
            web_search_needed = "Yes"

    return {"documents": filtered, "web_search": web_search_needed}


def generate(state: AgentState) -> dict:
    """Generate a context-grounded answer and increment the iteration counter."""
    print("\n--- NODE: GENERATE ---")
    context    = "\n\n".join(state["documents"])
    chain      = _GENERATE_PROMPT | _llm
    generation = chain.invoke({"context": context, "question": state["question"]})
    return {
        "generation": generation.content,
        "iterations": state.get("iterations", 0) + 1,
    }


def transform_query(state: AgentState) -> dict:
    """Rewrite the question to improve retrieval on the next loop iteration."""
    print("\n--- NODE: TRANSFORM QUERY ---")
    chain           = _REWRITE_PROMPT | _llm
    better_question = chain.invoke({"question": state["question"]})
    print(f"  Rewritten: {better_question.content[:80]}…")
    return {"question": better_question.content}


def web_search(state: AgentState) -> dict:
    """
    Fetch live web results via Tavily and append them to the document list.

    IMPORTANT — web search is OPTIONAL and requires a Tavily API key:
        1. Get a free key at https://tavily.com
        2. Add TAVILY_API_KEY=<your-key> to your .env file

    WITHOUT a key:
        - The import or API call will raise an exception.
        - The except block catches it silently.
        - The graph continues with whatever documents were already retrieved.
        - No crash, no data loss — web search is simply skipped.

    WITH a key:
        - Tavily fetches 3 live web results for the query.
        - Results are appended to the existing document list.
        - The LLM then has both local docs AND web context to draw from.

    NOTE: The LLM (Ollama) always runs locally. Tavily is only a data-fetching
    plugin — like a library card. The AI brain never touches the internet.
    """
    print("\n--- NODE: WEB SEARCH ---")
    question  = state["question"]
    documents = list(state.get("documents", []))

    tavily_key = config.tavily_api_key
    if not tavily_key:
        # ── Placeholder mode ──────────────────────────────────────────────────
        # No API key configured. Instead of silently doing nothing, we inject a
        # placeholder document so the full agent graph still runs end-to-end and
        # the user can see the web_search → generate flow working locally.
        print("  Web search running in PLACEHOLDER mode (no TAVILY_API_KEY found)")
        print("  To enable live web search: add TAVILY_API_KEY to your .env file")
        print("  Get a free key at https://tavily.com")

        placeholder = (
            f"[WEB SEARCH PLACEHOLDER]\n"
            f"Query: {question}\n\n"
            f"Web search is not yet configured for this system. "
            f"To enable live web results, add a TAVILY_API_KEY to your .env file "
            f"(free tier available at https://tavily.com).\n\n"
            f"This placeholder is standing in for real web results. "
            f"The answer below is generated from locally available documents only."
        )
        documents.append(placeholder)
        return {"documents": documents}

    # ── Live web search via Tavily ─────────────────────────────────────────────
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool    = TavilySearchResults(k=3)
        results = tool.invoke({"query": question})
        web_doc = "\n".join(r["content"] for r in results)
        documents.append(web_doc)
        print(f"  Web search returned {len(results)} results")
    except Exception as exc:
        print(f"  Web search failed ({exc}). Proceeding with existing docs.")

    return {"documents": documents}


# ══════════════════════════════════════════════════════════════════════════════
# EDGE / ROUTING FUNCTIONS  (return a string that drives conditional edges)
# ══════════════════════════════════════════════════════════════════════════════

def route_question(state: AgentState) -> Literal["web_search", "vectorstore"]:
    """Decide whether to hit the vectorstore or the web."""
    print("\n--- EDGE: ROUTE QUESTION ---")
    chain  = _ROUTE_PROMPT | _router_llm
    source = chain.invoke({"question": state["question"]})
    print(f"  → Routing to: {source.datasource}")
    return source.datasource


def decide_to_generate(
    state: AgentState,
) -> Literal["transform_query", "generate"]:
    """After grading, decide: transform the query or proceed to generation."""
    print("\n--- EDGE: DECIDE TO GENERATE ---")
    if state.get("web_search") == "Yes" or not state["documents"]:
        print("  → Not enough relevant docs — transforming query")
        return "transform_query"
    print(f"  → {len(state['documents'])} relevant docs — generating answer")
    return "generate"


def grade_generation(
    state: AgentState,
) -> Literal["useful", "not supported", "not useful"]:
    """
    Two-stage quality gate:
      1. Hallucination check  — is the answer grounded in the docs?
      2. Answer quality check — does it actually resolve the question?
    """
    print("\n--- EDGE: GRADE GENERATION ---")

    # Guard against infinite loops
    if state.get("iterations", 0) >= config.max_iterations:
        print(f"  → Max iterations reached ({config.max_iterations}). Returning best answer.")
        return "useful"

    context    = "\n\n".join(state["documents"])
    generation = state["generation"]
    question   = state["question"]

    # ① Hallucination check
    try:
        h_chain = _HALLUCINATION_PROMPT | _hall_grader_llm
        h_score = h_chain.invoke({"documents": context, "generation": generation})
        grounded = h_score.binary_score == "yes"
    except Exception:
        grounded = True   # fail-safe

    if not grounded:
        print("  → Hallucination detected — regenerating")
        return "not supported"

    # ② Answer quality check
    try:
        a_chain = _ANSWER_GRADE_PROMPT | _ans_grader_llm
        a_score = a_chain.invoke({"question": question, "generation": generation})
        useful  = a_score.binary_score == "yes"
    except Exception:
        useful = True     # fail-safe

    if useful:
        print("  → Answer is grounded and useful ✓")
        return "useful"

    print("  → Answer is unhelpful — transforming query")
    return "not useful"
