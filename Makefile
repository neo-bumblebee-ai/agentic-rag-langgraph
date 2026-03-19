.PHONY: install run test lint clean help

# ─── Default ─────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  Agentic RAG with LangGraph — available commands"
	@echo ""
	@echo "  make install   Install all dependencies (requires uv)"
	@echo "  make run       Start the interactive RAG agent"
	@echo "  make test      Run the test suite"
	@echo "  make lint      Run ruff linter"
	@echo "  make clean     Remove generated files (vector store, __pycache__)"
	@echo ""

# ─── Setup ───────────────────────────────────────────────────────────────────

install:
	@echo "→ Installing dependencies with uv..."
	uv sync --extra dev
	@echo "✓ Done. Run 'make run' to start."

# ─── Run ─────────────────────────────────────────────────────────────────────

run:
	@echo "→ Starting Agentic RAG agent..."
	uv run python main.py

# ─── Quality ─────────────────────────────────────────────────────────────────

lint:
	@echo "→ Running ruff linter..."
	uv run ruff check src/ tests/ main.py

test:
	@echo "→ Running tests..."
	uv run pytest tests/ -v --tb=short

# ─── Clean ───────────────────────────────────────────────────────────────────

clean:
	@echo "→ Removing generated files..."
	rm -rf data/vector_store/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean. Run 'make run' to rebuild from scratch."
