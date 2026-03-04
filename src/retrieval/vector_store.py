"""
ChromaDB-backed dense vector store.
Handles ingestion, similarity search, and full document retrieval for BM25 re-use.
"""

from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from ..config import config


class VectorStore:
    """Persistent ChromaDB store with sentence-transformers embeddings."""

    def __init__(self) -> None:
        Path(config.vector_store_dir).mkdir(parents=True, exist_ok=True)

        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        self._client = chromadb.PersistentClient(path=config.vector_store_dir)
        self._collection = self._client.get_or_create_collection(
            name="documents",
            embedding_function=self._ef,
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def ingest(self, chunks: List[str]) -> None:
        """Add document chunks (strings) to the collection."""
        if not chunks:
            return

        existing = self._collection.count()
        ids = [f"chunk_{existing + i}" for i in range(len(chunks))]

        self._collection.add(documents=chunks, ids=ids)
        print(
            f"  [VectorStore] Ingested {len(chunks)} chunks "
            f"(total: {existing + len(chunks)})"
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 10) -> List[str]:
        """Return the top-k most similar chunks for *query*."""
        n = self._collection.count()
        if n == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, n),
        )
        return results["documents"][0]

    def get_all_documents(self) -> List[str]:
        """Return every stored chunk — used to rebuild the BM25 index."""
        if self._collection.count() == 0:
            return []
        return self._collection.get()["documents"]

    def count(self) -> int:
        return self._collection.count()
