# Agentic RAG with LangGraph

[![CI](https://github.com/neo-bumblebee-ai/agentic-rag-langgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/neo-bumblebee-ai/agentic-rag-langgraph/actions/workflows/ci.yml)
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

## What Is This?

Traditional RAG (Retrieval-Augmented Generation) is a straight line: **retrieve → generate**. You ask a question, the system grabs some documents, and the AI answers. Simple — but it breaks down when:

- The retrieved documents aren't actually relevant
- The AI makes up an answer not supported by the documents
- The search query itself is poorly worded

This project fixes all three by turning the pipeline into a **self-correcting, multi-agent loop** using [LangGraph](https://langchain-ai.github.io/langgraph/). Instead of answering once and hoping for the best, the system:

1. **Checks** if retrieved documents are actually relevant (and discards ones that aren't)
2. **Detects** when the AI's answer isn't grounded in facts
3. **Rewrites** the question and tries again if retrieval fails
4. **Remembers** your previous questions in the same session

Everything runs **100% on your local machine**. No cloud services, no API keys required to get started.

---

## How It Works — Plain English

```
You ask a question
       ↓
Router decides: search my documents OR search the web?
       ↓
Retrieval runs 3 search methods simultaneously and merges results
       ↓
Document Grader checks each result — throws out irrelevant ones
       ↓
AI writes an answer using only the vetted documents
       ↓
Fact Checker: Is the answer actually supported by the documents?
Quality Check: Does it actually answer the question?
       ↓ both pass → you get the answer
       ↓ either fails → query is rewritten → loop back
```

---

## Architecture

```
                    ╔══════════════════════════════════════════╗
                    ║     Agentic RAG  —  LangGraph Graph      ║
                    ╚══════════════════════════════════════════╝

                                    START
                                      │
                          ┌───────────▼───────────┐
                          │     Route Question    │  ← LLM decides datasource
                          └───────┬───────────┬───┘
                       vectorstore│           │web_search
                                  │           │
              ┌───────────────────▼── ┐   ┌────▼──────────────┐
              │       Retrieve        │   │    Web Search     │
              │  ① Dense (ChromaDB)   │   │  (Tavily API)     │
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
                     │   │  Transform Query  │     │
                     │   │  (LLM rewrite)    │     │
                     │   └────────┬──────────┘     │
                     │            │ (loops → Retrieve)
                     │            │                |
              ┌──────▼────────────┴────────────────┘
              │              Generate              │
              │   (context-grounded answer)        │
              └──────────────┬─────────────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │         Grade Generation            │
              │  ① Hallucination check              │
              │  ② Answer quality check             │
              └────┬──────────┬──────────┬──────────┘
                   │          │          │
               useful   not supported  not useful
                   │          │          │
                  END      Generate   Transform Query
                           (retry)    (rewrite + retrieve)
```

---

## Why This Beats Traditional RAG

| Problem | Traditional RAG | This System |
|---|---|---|
| Irrelevant documents | ❌ All top-K passed to LLM | ✅ Document Grader filters each one |
| Hallucinations | ❌ No protection | ✅ Fact-checked before delivery |
| Bad query → bad results | ❌ Returns empty / wrong answer | ✅ Query Rewriter retries automatically |
| Keyword vs semantic mismatch | ❌ Dense cosine only | ✅ BM25 + Dense + RRF + Reranking |
| No memory between questions | ❌ Stateless | ✅ Rolling conversation memory |

---

## Prerequisites

Before you start, make sure you have these installed:

| Tool | Purpose | Install |
|---|---|---|
| **Python 3.11+** | Run the code | [python.org](https://www.python.org/downloads/) |
| **uv** | Package manager (fast) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Ollama** | Run AI models locally | [ollama.com/download](https://ollama.com/download) |
| **Git** | Clone the repo | [git-scm.com](https://git-scm.com) |

> **Don't have uv?** You can use `pip` instead — every step has a `pip` alternative.

---

## Quick Start

### Step 1 — Clone the repository

```bash
git clone https://github.com/neo-bumblebee-ai/agentic-rag-langgraph.git
cd agentic-rag-langgraph
```

### Step 2 — Install dependencies

```bash
# With make (recommended)
make install

# With uv directly
uv sync

# With pip (alternative)
pip install -r requirements.txt
```

### Step 3 — Download an AI model

```bash
# Start Ollama first (if not already running)
ollama serve

# Then pull a model (in a new terminal)
ollama pull llama3.2        # ~2 GB  — best quality, recommended
ollama pull llama3.2:1b     # ~800 MB — faster, good for low-end machines
ollama pull mistral         # ~4 GB  — alternative option
```

> **Which model should I pick?** Start with `llama3.2`. If your machine is slow, use `llama3.2:1b`.

### Step 4 — Add your documents

Drop any PDF or text files into the `data/` folder:

```
data/
├── pdf_files/      ← put your PDFs here (user manuals, reports, articles, etc.)
└── text_files/     ← put your .txt files here
```

> **No documents yet?** That's fine — you can add them later. The system will just tell you the folder is empty.

### Step 5 — Run

```bash
# With make (recommended)
make run

# With uv directly
uv run python main.py

# With pip / regular Python
python main.py
```

**What happens on first run:**
1. All documents in `data/` are loaded and split into chunks
2. Each chunk is embedded and stored in ChromaDB (takes a few minutes)
3. BM25 index is built in memory
4. The agent graph is compiled
5. You get a question prompt

**What happens on later runs:**
- Steps 1–3 are skipped — everything is loaded from disk instantly

---

## Example Session

```
=== Agentic RAG Pipeline — LangGraph Multi-Agent System ===

Vector store ready — 847 chunks loaded.
BM25 index rebuilt from existing documents.

Model      : llama3.2
Embeddings : all-MiniLM-L6-v2
Reranker   : cross-encoder/ms-marco-MiniLM-L-6-v2

Type a question, 'clear' to reset memory, or 'quit' to exit.

Question: What are the maintenance steps after a power outage?

  → Routing to: vectorstore
  [Dense]    10 chunks retrieved
  [BM25]      8 chunks retrieved
  [Reranker] Reranked to top 5
  doc_0 → relevant ✓
  doc_1 → relevant ✓
  doc_2 → not relevant ✗
  doc_3 → relevant ✓
  → Hallucination check: grounded ✓
  → Answer quality: useful ✓

Answer:
After a power outage: 1) Wait 5 minutes for pressures to equalise,
2) Press the Fault Reset button on the control panel, 3) Verify all
dampers returned to home positions, 4) Monitor for 15 minutes.
```

---

## Configuration

All settings are in one file — `src/config.py`. You don't need to touch this to get started, but here's what you can change:

| Setting | Default | What It Does |
|---|---|---|
| `ollama_model` | `llama3.2` | Which AI model to use — any model you've pulled with Ollama |
| `embedding_model` | `all-MiniLM-L6-v2` | How documents are converted to searchable vectors |
| `chunk_size` | `1000` | How many characters per document chunk |
| `chunk_overlap` | `200` | Overlap between chunks so context isn't lost at boundaries |
| `top_k_retrieval` | `10` | How many candidates each retriever fetches |
| `top_k_final` | `5` | How many chunks the AI actually sees after reranking |
| `max_iterations` | `3` | How many self-correction attempts before giving up |

---

## Optional Features

### Web Search (Tavily)

Web search runs in **placeholder mode** when no API key is configured. The full agent graph still executes end-to-end — instead of live results, a placeholder document is injected so you can see the complete web search flow working locally without signing up for anything.

| Mode | Behaviour |
|---|---|
| **No API key** (default) | Placeholder mode — full graph runs, local docs only, no crash |
| **With API key** | Live web results fetched and appended alongside local docs |

To enable live web results:

1. Get a free API key at [tavily.com](https://tavily.com)
2. Copy the example env file: `cp .env.example .env`
3. Add your key: `TAVILY_API_KEY=tvly-your-key-here`

> **Important:** The AI model (Ollama) always runs locally — even with Tavily enabled. Tavily only fetches raw web text. Your prompts and answers never leave your machine.

### LangSmith Tracing

Visualise the full agent graph execution step-by-step:

1. Create a free account at [smith.langchain.com](https://smith.langchain.com)
2. Add to `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-key-here
LANGCHAIN_PROJECT=agentic-rag-langgraph
```

---

## Troubleshooting

**`Failed to connect to Ollama`**
Ollama isn't running. Start it:
```bash
ollama serve
```

**`No documents found`**
The `data/pdf_files/` and `data/text_files/` folders are empty. Add at least one `.pdf` or `.txt` file.

**`Collection has 0 docs` on restart**
The vector store was deleted or path changed. Delete `data/vector_store/` and re-run to rebuild.

**Slow responses**
- Switch to a smaller model: `ollama pull llama3.2:1b` then set `ollama_model = "llama3.2:1b"` in `src/config.py`
- Reduce `top_k_retrieval` from 10 to 5

**Poor answer quality**
- Reduce `chunk_size` to `500` for more granular retrieval
- Make sure your documents are actually relevant to what you're asking

**`ModuleNotFoundError`**
Dependencies aren't installed. Run `uv sync` or `pip install -r requirements.txt`.

---

## Project Structure

```
agentic-rag-langgraph/
│
├── main.py                      # ← START HERE — run this file
├── Makefile                     # make install / run / test / clean
│
├── .github/
│   └── workflows/
│       └── ci.yml               # Runs lint + tests on every push
│
├── src/
│   ├── config.py                # All settings in one place
│   ├── graph/
│   │   ├── state.py             # Shared data structure between agents
│   │   ├── nodes.py             # What each agent does
│   │   └── workflow.py          # How agents connect to each other
│   ├── retrieval/
│   │   ├── vector_store.py      # Semantic (meaning-based) search
│   │   ├── bm25_retriever.py    # Keyword-based search
│   │   ├── hybrid_search.py     # Combines both + reranking
│   │   └── reranker.py          # Final quality scoring
│   ├── memory/
│   │   └── conversation.py      # Remembers your previous questions
│   └── ingestion/
│       └── loader.py            # Reads your PDFs and text files
│
├── tests/                       # Test suite (pytest)
│
├── data/
│   ├── pdf_files/               # ← put your PDFs here
│   ├── text_files/
│   │   └── sample_rag_overview.txt  # ← sample document to query immediately
│   └── vector_store/            # Auto-generated — do not edit
│
├── .env.example                 # Template for optional API keys
├── pyproject.toml               # Project dependencies
└── requirements.txt             # pip-compatible dependency list
```

---

## Tech Stack

| Component | Tool | Notes |
|---|---|---|
| Agent orchestration | **LangGraph** | Stateful, cyclical multi-agent graph |
| LLM | **Ollama** (`llama3.2`) | Fully local — no API key required |
| Dense retrieval | **ChromaDB** + sentence-transformers | `all-MiniLM-L6-v2` embeddings |
| Sparse retrieval | **BM25** (`rank-bm25`) | Okapi BM25 over all document chunks |
| Fusion | **Reciprocal Rank Fusion** | Combines dense + sparse rankings |
| Reranking | **CrossEncoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Web search | **Tavily** | Optional — needs `TAVILY_API_KEY`. LLM always stays local. |
| Tracing | **LangSmith** | Optional — needs `LANGCHAIN_API_KEY` |
| Memory | Custom LLM summarisation | Compresses history when it grows too long |
| Package manager | `uv` | Fast, modern Python package manager |

---

## Roadmap

- [x] Hybrid search (dense + BM25 + RRF)
- [x] Cross-encoder reranking
- [x] Document relevance grading
- [x] Hallucination detection + self-correction
- [x] Query rewriting
- [x] Conversation memory with summarisation
- [ ] Streamlit web UI
- [ ] RAGAS evaluation suite
- [ ] LangSmith tracing dashboard integration
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

## Author

**Jignesh Patel** — [@neo-bumblebee-ai](https://github.com/neo-bumblebee-ai)
Senior Data Architect & Engineering Lead · 17+ years building large-scale data platforms.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the full contributor list.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

[MIT License](LICENSE)
