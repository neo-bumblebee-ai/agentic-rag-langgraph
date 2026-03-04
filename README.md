# Agentic RAG with LangGraph

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.com)
[![Vector DB](https://img.shields.io/badge/vectordb-ChromaDB-blueviolet.svg)](https://www.trychroma.com)

> **Part 3 of a hands-on AI engineering series** — building in public, month by month.
>
> ← [Part 2 — Traditional RAG Pipeline](https://github.com/neo-bumblebee-ai/traditional-rag-pipeline) &nbsp;|&nbsp; [Part 4 — LLMOps Evaluation Platform →](https://github.com/neo-bumblebee-ai) *(coming April 2026)*

---

Traditional RAG is a straight line: retrieve → generate. Real-world questions are rarely that simple.

This project upgrades the pipeline into a **self-correcting, multi-agent system** built with [LangGraph](https://langchain-ai.github.io/langgraph/) — one that can intelligently route queries, grade its own retrieved context, detect hallucinations, rewrite failing queries, and loop until the answer is trustworthy.

Every component is explicit and swappable. No black-box wrappers.

---

## Why This Matters

| Problem with Traditional RAG | How This Fixes It |
|---|---|
| Retrieves chunks regardless of relevance | **Document Grader Agent** filters irrelevant docs before generation |
| Answers even when context is wrong or empty | **Hallucination Checker** triggers regeneration if answer isn't grounded |
| Same failing query is retried unchanged | **Query Rewriter** improves the question semantically for better recall |
| Only dense (cosine) retrieval | **Hybrid Search**: BM25 sparse + dense vectors fused via Reciprocal Rank Fusion |
| Single-pass retrieval, no second chances | **Cross-Encoder Reranker** rescores candidates by reading query+doc together |
| Stateless — no memory between turns | **Conversation Memory** with automatic LLM-driven summarisation |

---

## Architecture

```
                    ╔══════════════════════════════════════════╗
                    ║     Agentic RAG  —  LangGraph Graph       ║
                    ╚══════════════════════════════════════════╝

                                    START
                                      │
                          ┌───────────▼───────────┐
                          │     Route Question     │  ← LLM decides datasource
                          └───────┬───────────┬───┘
                       vectorstore│           │web_search
                                  │           │
              ┌───────────────────▼──┐   ┌────▼──────────────┐
              │       Retrieve        │   │    Web Search      │
              │  ① Dense (ChromaDB)   │   │  (Tavily API)      │
              │  ② Sparse (BM25)      │   └────────┬──────────┘
              │  ③ RRF Fusion         │            │
              │  ④ Cross-Encoder      │            │
              └───────────┬───────────┘            │
                          │                        │
              ┌───────────▼───────────┐            │
              │    Grade Documents    │            │
              │  (per-doc LLM score)  │            │
              └──────┬────────┬───────┘            │
                     │        │                    │
              relevant?    not enough              │
                     │        │                    │
                     │   ┌────▼──────────────┐     │
                     │   │  Transform Query   │     │
                     │   │  (LLM rewrite)     │     │
                     │   └────────┬──────────┘     │
                     │            │ (loops → Retrieve)
                     │            │
              ┌──────▼────────────┴────────────────┘
              │              Generate               │
              │   (context-grounded answer)         │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │         Grade Generation             │
              │  ① Hallucination check               │
              │  ② Answer quality check              │
              └────┬──────────┬──────────┬───────────┘
                   │          │          │
               useful   not supported  not useful
                   │          │          │
                  END      Generate   Transform Query
                           (retry)    (rewrite + retrieve)
```

---

## Retrieval Deep Dive

The retriever runs three stages before a single token is generated:

```
Query
  │
  ├─① Dense Retrieval ──────────────────────────────────────────────────────┐
  │   ChromaDB + sentence-transformers (all-MiniLM-L6-v2)                   │
  │   Returns top-10 by cosine similarity                                   │
  │                                                                          │
  ├─② Sparse Retrieval (BM25) ──────────────────────────────────────────────┤
  │   Okapi BM25 over all ingested chunks                                   │
  │   Returns top-10 by term frequency score                                │
  │                                                                          ▼
  └─③ Reciprocal Rank Fusion ──────────► ④ Cross-Encoder Reranking ──► Top-5
      Merges both ranked lists               (query, doc) pair scoring
      Score = Σ 1/(k + rank_i)              cross-encoder/ms-marco-MiniLM-L-6-v2
```

**Why hybrid?** Dense retrieval excels at semantic similarity. BM25 excels at exact keyword matches. Neither alone is optimal — RRF combines both without requiring manual weight tuning.

**Why cross-encoder reranking?** Bi-encoders (cosine similarity) encode query and document independently. A cross-encoder reads them *together*, capturing fine-grained relevance signals. Much more accurate, used as a final pass over the fused candidates.

---

## Demo

```
=== Agentic RAG Pipeline — LangGraph Multi-Agent System ===

Vector store ready — 847 chunks loaded.
BM25 index rebuilt from existing documents.

Compiling agent graph …
  Model      : llama3.2
  Embeddings : all-MiniLM-L6-v2
  Reranker   : cross-encoder/ms-marco-MiniLM-L-6-v2
  Max loops  : 3

Question: What are the steps to reset the HVAC unit after a power outage?

--- EDGE: ROUTE QUESTION ---
  → Routing to: vectorstore

--- NODE: RETRIEVE ---
  [Dense]    10 chunks retrieved
  [BM25]      8 chunks retrieved (5 unique)
  [RRF]      14 candidates fused
  [Reranker] Reranked to top 5

--- NODE: GRADE DOCUMENTS ---
  doc_0 → yes ✓
  doc_1 → yes ✓
  doc_2 → no  ✗
  doc_3 → yes ✓
  doc_4 → yes ✓

--- EDGE: DECIDE TO GENERATE ---
  → 4 relevant docs found — generating answer

--- NODE: GENERATE ---

--- EDGE: GRADE GENERATION ---
  → Hallucination check: grounded ✓
  → Answer quality: useful ✓

──────────────────────────────────────────────────────────
Answer:
After a power outage, reset the HVAC unit by: 1) waiting 5 minutes for
pressures to equalise, 2) pressing the "Fault Reset" button on the control
panel, 3) verifying all dampers have returned to their home positions, and
4) monitoring the system for 15 minutes before leaving unattended.
──────────────────────────────────────────────────────────
```

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| Agent orchestration | **LangGraph** | Stateful, cyclical multi-agent graph |
| LLM | **Ollama** (`llama3.2`) | Fully local — no API key required |
| Dense retrieval | **ChromaDB** + sentence-transformers | `all-MiniLM-L6-v2` embeddings |
| Sparse retrieval | **BM25** (`rank-bm25`) | Okapi BM25 over all document chunks |
| Fusion | **Reciprocal Rank Fusion** | Combines dense + sparse rankings |
| Reranking | **CrossEncoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Web search | **Tavily** | Optional — requires `TAVILY_API_KEY` |
| Tracing | **LangSmith** | Optional — requires `LANGCHAIN_API_KEY` |
| Memory | Custom LLM summarisation | Compresses history when it grows too long |
| Package manager | `uv` | Fast, modern Python package manager |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/neo-bumblebee-ai/agentic-rag-langgraph.git
cd agentic-rag-langgraph
```

### 2. Install

```bash
uv sync
# or: pip install -r requirements.txt
```

### 3. Pull an Ollama model

```bash
ollama pull llama3.2       # ~2 GB — recommended
ollama pull llama3.2:1b    # ~800 MB — faster on low-end hardware
ollama pull mistral        # alternative
```

### 4. Add your documents

```
data/
├── pdf_files/      ← drop any PDFs here
└── text_files/     ← or plain .txt files
```

### 5. Configure (optional)

```bash
cp .env.example .env
# Add TAVILY_API_KEY for web search
# Add LANGCHAIN_API_KEY for LangSmith tracing
```

### 6. Run

```bash
uv run python main.py
```

First run ingests and embeds all documents. Subsequent runs load from disk — no re-embedding.

---

## Configuration

All settings live in `src/config.py`:

```python
ollama_model     = "llama3.2"                            # any Ollama model
embedding_model  = "all-MiniLM-L6-v2"                   # sentence-transformers model
reranker_model   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
chunk_size       = 1000
chunk_overlap    = 200
top_k_retrieval  = 10    # candidates fetched by each retriever
top_k_final      = 5     # chunks passed to LLM after reranking
max_iterations   = 3     # self-correction loop guard
```

---

## Project Structure

```
agentic-rag-langgraph/
│
├── main.py                          # Entry point — run this
│
├── src/
│   ├── config.py                    # All configuration in one place
│   │
│   ├── graph/
│   │   ├── state.py                 # LangGraph AgentState TypedDict
│   │   ├── nodes.py                 # All node + edge functions
│   │   └── workflow.py              # Graph construction + compilation
│   │
│   ├── retrieval/
│   │   ├── vector_store.py          # ChromaDB persistent vector store
│   │   ├── bm25_retriever.py        # BM25 sparse retrieval
│   │   ├── hybrid_search.py         # RRF fusion + reranking orchestrator
│   │   └── reranker.py              # Cross-encoder reranking
│   │
│   ├── memory/
│   │   └── conversation.py          # Rolling history + LLM summarisation
│   │
│   └── ingestion/
│       └── loader.py                # PDF + text loading and chunking
│
├── data/
│   ├── pdf_files/                   # Your PDFs (not committed)
│   ├── text_files/                  # Your .txt files (not committed)
│   └── vector_store/                # ChromaDB persisted here (not committed)
│
├── notebooks/
│   ├── 01_graph_walkthrough.ipynb   # Step-by-step graph visualisation
│   └── 02_hybrid_search_demo.ipynb  # BM25 vs dense vs hybrid comparison
│
├── tests/
│   ├── test_retrieval.py
│   ├── test_grader.py
│   └── test_workflow.py
│
├── .env.example
├── pyproject.toml
├── requirements.txt
└── CONTRIBUTING.md
```

---

## Roadmap

- [x] Hybrid search (dense + BM25 + RRF)
- [x] Cross-encoder reranking
- [x] Document relevance grading
- [x] Hallucination detection + self-correction
- [x] Query rewriting
- [x] Conversation memory with summarisation
- [ ] LangSmith tracing dashboard integration
- [ ] Streamlit web UI
- [ ] RAGAS evaluation suite
- [ ] Multi-document source attribution

---

## Series

This is **Part 3** of a 6-month build-in-public AI engineering series:

| Month | Project | Status |
|---|---|---|
| January | [Databricks AI Engineering Challenge](https://github.com/neo-bumblebee-ai/databricks-ai-engineering-challenge) | ✅ Complete |
| February | [Traditional RAG Pipeline](https://github.com/neo-bumblebee-ai/traditional-rag-pipeline) | ✅ Complete |
| **March** | **Agentic RAG with LangGraph (this repo)** | 🔨 In Progress |
| April | LLMOps Evaluation Platform | 🔜 Coming |
| May | LLM Fine-Tuning + NVIDIA NIM | 🔜 Coming |
| June | Enterprise AI Platform (Capstone) | 🔜 Coming |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

[MIT License](LICENSE)
