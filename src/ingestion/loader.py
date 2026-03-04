"""
Document loader and chunker.

Supports PDF and plain-text files. On first run, documents are loaded,
split into overlapping chunks, and stored in the hybrid retriever.
Subsequent runs reload from the persisted ChromaDB — no re-embedding needed.
"""

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from ..config import config


def load_documents() -> List[Document]:
    """
    Walk the PDF and text data directories, load every file,
    and split into overlapping chunks.

    Returns:
        List of LangChain Document objects (each is one chunk).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    documents: List[Document] = []

    # ── PDFs ───────────────────────────────────────────────────────────────────
    pdf_dir = Path(config.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            loader = PyPDFLoader(str(pdf_path))
            raw    = loader.load()
            chunks = splitter.split_documents(raw)
            documents.extend(chunks)
            print(f"  [Loader] {pdf_path.name} — {len(raw)} pages → {len(chunks)} chunks")
        except Exception as exc:
            print(f"  [Loader] WARNING: could not load {pdf_path.name}: {exc}")

    # ── Plain text ─────────────────────────────────────────────────────────────
    txt_dir = Path(config.text_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    for txt_path in sorted(txt_dir.glob("*.txt")):
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            raw    = loader.load()
            chunks = splitter.split_documents(raw)
            documents.extend(chunks)
            print(f"  [Loader] {txt_path.name} — {len(chunks)} chunks")
        except Exception as exc:
            print(f"  [Loader] WARNING: could not load {txt_path.name}: {exc}")

    return documents
