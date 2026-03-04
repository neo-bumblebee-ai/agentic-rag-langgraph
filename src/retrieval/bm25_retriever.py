"""
Sparse retrieval using Okapi BM25.
Complements dense embeddings — especially strong for keyword-heavy queries.
"""

from typing import List

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """In-memory BM25 index over document chunks."""

    def __init__(self) -> None:
        self._documents: List[str] = []
        self._bm25: BM25Okapi | None = None

    # ── Write ──────────────────────────────────────────────────────────────────

    def ingest(self, documents: List[str]) -> None:
        """Build (or rebuild) the BM25 index from *documents*."""
        self._documents = documents
        tokenized = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        print(f"  [BM25] Index built — {len(documents)} documents")

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> List[str]:
        """Return top-k documents ranked by BM25 score."""
        if self._bm25 is None or not self._documents:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]
        return [self._documents[i] for i in top_indices if scores[i] > 0]
