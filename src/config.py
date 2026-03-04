"""
Centralised configuration for the Agentic RAG pipeline.
All tuneable parameters live here — no magic strings scattered across the codebase.
"""

from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── LLM ────────────────────────────────────────────────────────────────────
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Reranker ───────────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Retrieval ──────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 10   # candidates fetched by each retriever
    top_k_final: int = 5        # chunks passed to the LLM after reranking

    # ── Paths ──────────────────────────────────────────────────────────────────
    vector_store_dir: str = "data/vector_store"
    pdf_dir: str = "data/pdf_files"
    text_dir: str = "data/text_files"

    # ── Agent ──────────────────────────────────────────────────────────────────
    max_iterations: int = 3     # self-correction loop guard

    # ── Optional integrations ──────────────────────────────────────────────────
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )
    langsmith_api_key: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", "")
    )
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "agentic-rag-langgraph")
    )


config = Config()
