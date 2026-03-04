"""
Hybrid retrieval: Dense + Sparse → Reciprocal Rank Fusion → Cross-Encoder Reranking.

Pipeline:
  1. ChromaDB (dense cosine similarity)  — top-K candidates
  2. BM25 (sparse term matching)         — top-K candidates
  3. Reciprocal Rank Fusion              — merges both ranked lists
  4. CrossEncoder reranker               — final top-K by joint scoring
"""

from typing import List

from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from .reranker import CrossEncoderReranker
from ..config import config


class HybridRetriever:
    """Orchestrates dense + sparse retrieval with cross-encoder reranking."""

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self.reranker = CrossEncoderReranker()

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, chunks: List[str]) -> None:
        """Store chunks in ChromaDB and build the BM25 index."""
        self.vector_store.ingest(chunks)
        self.bm25.ingest(chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[str]:
        """
        Full hybrid retrieval pipeline for *query*.

        Returns top-K chunks after RRF fusion and cross-encoder reranking.
        """
        k = config.top_k_retrieval

        # Step 1 — dense retrieval
        dense_results = self.vector_store.similarity_search(query, k=k)
        print(f"  [Dense]   {len(dense_results)} chunks retrieved")

        # Step 2 — sparse retrieval
        bm25_results = self.bm25.search(query, k=k)
        unique_bm25 = [d for d in bm25_results if d not in dense_results]
        print(f"  [BM25]    {len(bm25_results)} chunks retrieved ({len(unique_bm25)} unique)")

        # Step 3 — Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion([dense_results, bm25_results])
        print(f"  [RRF]     {len(fused)} candidates fused")

        # Step 4 — cross-encoder reranking
        reranked = self.reranker.rerank(query, fused, top_k=config.top_k_final)
        print(f"  [Reranker] Reranked to top {len(reranked)}")

        return reranked

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        result_lists: List[List[str]], k: int = 60
    ) -> List[str]:
        """
        Merge multiple ranked lists via Reciprocal Rank Fusion (RRF).

        Score(doc) = Σ  1 / (k + rank_i)   for each list i that contains doc.
        Higher score → better combined rank.
        """
        scores: dict[str, float] = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)

        return sorted(scores, key=lambda d: scores[d], reverse=True)
