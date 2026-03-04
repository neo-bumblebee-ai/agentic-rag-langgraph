"""
Conversation memory with automatic summarisation.

Keeps a rolling window of Q&A turns. When the history exceeds *max_turns*,
the LLM compresses everything into a summary to preserve context without
blowing up the prompt window.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ..config import config


class ConversationMemory:
    """Rolling conversation history with LLM-driven summarisation."""

    def __init__(self, max_turns: int = 8) -> None:
        self._history: list[BaseMessage] = []
        self._max_messages = max_turns * 2  # each turn = 1 Human + 1 AI
        self._llm = ChatOllama(
            model=config.ollama_model, temperature=config.temperature
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_exchange(self, question: str, answer: str) -> None:
        """Append a Q&A pair and compress history if needed."""
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        if len(self._history) > self._max_messages:
            self._summarise()

    def get_history(self) -> list[BaseMessage]:
        return list(self._history)

    def clear(self) -> None:
        self._history = []

    # ── Internal ───────────────────────────────────────────────────────────────

    def _summarise(self) -> None:
        """Replace the full history with a one-shot LLM summary."""
        if not self._history:
            return

        history_text = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in self._history
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Summarise the conversation below in 3–5 bullet points, "
             "preserving key facts and decisions. Be concise."),
            ("human", "{history}"),
        ])

        chain   = prompt | self._llm
        summary = chain.invoke({"history": history_text})

        self._history = [
            SystemMessage(
                content=f"[Conversation summary]\n{summary.content}"
            )
        ]
        print("  [Memory] History compressed into summary.")
