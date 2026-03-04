"""
LangGraph workflow — wires nodes and edges into a compiled graph.

Graph topology:
                        START
                          │
                   ┌──────▼──────┐
                   │route_question│
                   └──┬───────┬──┘
             vectorstore│     │web_search
                   ┌────▼─┐  ┌▼────────┐
                   │Retrieve│  │WebSearch│
                   └────┬──┘  └────┬────┘
                        │          │
                ┌───────▼──────┐   │
                │GradeDocuments│   │
                └──┬───────┬───┘   │
         relevant  │       │not enough
                   │  ┌────▼──────────┐
                   │  │TransformQuery  │◄─────────┐
                   │  └────┬──────────┘           │
                   │       │ (loop back to Retrieve)
                   │       └──► Retrieve           │
                   │                               │
           ┌───────▼──────────────────────┐        │
           │           Generate            │        │
           └───────────────┬──────────────┘        │
                           │                        │
               ┌───────────▼────────────┐           │
               │    GradeGeneration     │           │
               └──┬──────┬──────┬──────┘           │
                  │      │      └─ not useful ──────┘
         not      │  useful
       supported  │      │
            ┌─────▼┐     └──► END
            │Regen  │
            └───────┘
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    decide_to_generate,
    generate,
    grade_documents,
    grade_generation,
    retrieve,
    route_question,
    transform_query,
    web_search,
)
from .state import AgentState


def build_graph():
    """Construct and compile the agentic RAG LangGraph."""

    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("retrieve",         retrieve)
    workflow.add_node("grade_documents",  grade_documents)
    workflow.add_node("generate",         generate)
    workflow.add_node("transform_query",  transform_query)
    workflow.add_node("web_search",       web_search)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "vectorstore": "retrieve",
            "web_search":  "web_search",
        },
    )

    # ── Retrieval path ────────────────────────────────────────────────────────
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate":       "generate",
            "transform_query": "transform_query",
        },
    )

    # ── Query rewrite loops back to retrieval ─────────────────────────────────
    workflow.add_edge("transform_query", "retrieve")

    # ── Web search goes straight to generation ────────────────────────────────
    workflow.add_edge("web_search", "generate")

    # ── Self-correction gate ──────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "useful":        END,
            "not supported": "generate",        # regenerate (hallucination)
            "not useful":    "transform_query", # rewrite query
        },
    )

    return workflow.compile()
