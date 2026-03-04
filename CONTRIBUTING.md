# Contributing

Thank you for your interest in contributing to this project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/agentic-rag-langgraph.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `uv sync`

## Making Changes

- Keep changes focused and atomic
- Add tests for new functionality in `tests/`
- Follow the existing code style (ruff enforced)
- Run `ruff check .` before committing

## Submitting a Pull Request

1. Push your branch: `git push origin feature/your-feature-name`
2. Open a PR against `main`
3. Fill in the pull request template
4. Reference any related issues

## Reporting Issues

Use the GitHub issue templates for:
- **Bug reports** — include steps to reproduce, expected vs. actual behaviour, and your environment
- **Feature requests** — describe the use case and proposed solution

## Code Style

- Python 3.11+
- Type hints on all public functions
- Docstrings for all modules, classes, and public methods
- Line length: 100 characters (ruff enforced)
