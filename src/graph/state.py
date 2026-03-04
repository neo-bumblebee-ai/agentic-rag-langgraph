"""
LangGraph agent state definition.
Every node reads from and writes into this shared TypedDict.
"""

from typing import TypedDict, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state passed between all nodes in the graph."""

    # The current question (may be rewritten by transform_query)
    question: str

    # Retrieved and filtered document chunks
    documents: List[str]

    # Final generated answer
    generation: str

    # "Yes" → at least one doc was graded irrelevant → consider web search
    web_search: str

    # Iteration counter — guards against infinite self-correction loops
    iterations: int

    # Multi-turn conversation history
    chat_history: List[BaseMessage]
