"""Tests for BM25 retriever."""

from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_ingest_and_search():
    retriever = BM25Retriever()
    docs = [
        "Retrieval-Augmented Generation combines retrieval with generation.",
        "BM25 is a keyword-based ranking algorithm.",
        "ChromaDB stores vector embeddings for semantic search.",
    ]
    retriever.ingest(docs)
    results = retriever.search("BM25 keyword ranking", k=2)
    assert len(results) <= 2
    # The BM25 doc should rank highest for this query
    assert any("BM25" in r for r in results)


def test_bm25_search_without_ingest():
    retriever = BM25Retriever()
    results = retriever.search("anything", k=5)
    assert results == []


def test_bm25_k_capped_at_corpus_size():
    retriever = BM25Retriever()
    retriever.ingest(["doc one", "doc two"])
    results = retriever.search("doc", k=10)
    assert len(results) <= 2
