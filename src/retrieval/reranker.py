"""
Cross-encoder reranker.

Unlike bi-encoders (cosine similarity), a cross-encoder processes the
query and document *together* — dramatically improving ranking accuracy
at the cost of latency. Used as a final scoring pass over the fused candidates.
"""

from typing import List

from sentence_transformers import CrossEncoder

from ..config import config


class CrossEncoderReranker:
    """Rerank candidates using a cross-encoder (query, document) pair scorer."""

    def __init__(self) -> None:
        print(f"  [Reranker] Loading: {config.reranker_model}")
        self._model = CrossEncoder(config.reranker_model)

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """
        Score every (query, document) pair and return *top_k* by descending score.

        Args:
            query:     The user question.
            documents: Candidate chunks from hybrid retrieval.
            top_k:     How many to return after reranking.

        Returns:
            List of document strings sorted best-first.
        """
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
