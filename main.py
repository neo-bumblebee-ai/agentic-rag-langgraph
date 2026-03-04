#!/usr/bin/env python3
"""
Agentic RAG with LangGraph — Entry Point

Multi-agent Q&A system with:
  • Hybrid retrieval  (dense + BM25 → RRF)
  • Cross-encoder reranking
  • Per-document relevance grading
  • Hallucination detection + self-correction
  • Automatic query rewriting
  • Conversation memory with summarisation
"""

from src.config import config
from src.graph import nodes as _nodes
from src.graph.workflow import build_graph
from src.ingestion.loader import load_documents
from src.memory.conversation import ConversationMemory
from src.retrieval.hybrid_search import HybridRetriever


# ── Setup ──────────────────────────────────────────────────────────────────────

def setup_retriever() -> HybridRetriever:
    """
    Initialise the hybrid retriever.

    • First run  → ingest documents from data/ directories.
    • Later runs → load from persisted ChromaDB; rebuild BM25 in-memory.
    """
    retriever = HybridRetriever()

    if retriever.vector_store.count() == 0:
        print("No documents in vector store. Starting ingestion …\n")
        docs = load_documents()
        if docs:
            chunks = [d.page_content for d in docs]
            retriever.ingest(chunks)
            print(f"\n  Total chunks ingested: {len(chunks)}\n")
        else:
            print(f"  No documents found.")
            print(f"  Add PDFs  → {config.pdf_dir}/")
            print(f"  Add texts → {config.text_dir}/\n")
    else:
        all_docs = retriever.vector_store.get_all_documents()
        retriever.bm25.ingest(all_docs)
        print(
            f"Vector store ready — {retriever.vector_store.count()} chunks loaded.\n"
            f"BM25 index rebuilt from existing documents.\n"
        )

    return retriever


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 58)
    print("  Agentic RAG Pipeline  —  LangGraph Multi-Agent System")
    print("=" * 58)
    print()

    retriever = setup_retriever()

    # Inject retriever into the nodes module (avoids re-instantiation)
    _nodes._retriever = retriever

    print("Compiling agent graph …")
    graph  = build_graph()
    memory = ConversationMemory()

    print(f"  Model      : {config.ollama_model}")
    print(f"  Embeddings : {config.embedding_model}")
    print(f"  Reranker   : {config.reranker_model}")
    print(f"  Max loops  : {config.max_iterations}")
    print("\nType a question, 'clear' to reset memory, or 'quit' to exit.\n")
    print("─" * 58)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if question.lower() == "clear":
            memory.clear()
            print("  Memory cleared.")
            continue

        # ── Build initial graph state ──────────────────────────────────────────
        state = {
            "question":    question,
            "chat_history": memory.get_history(),
            "documents":   [],
            "generation":  "",
            "web_search":  "No",
            "iterations":  0,
        }

        print()
        result = graph.invoke(state)

        answer = result.get("generation") or "I could not generate an answer."
        memory.add_exchange(question, answer)

        print(f"\n{'─' * 58}")
        print(f"Answer:\n{answer}")
        print(f"{'─' * 58}")


if __name__ == "__main__":
    main()
