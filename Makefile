.PHONY: help install install-dev build clean test test-c coverage docs lint format format-python format-c check all

# Default target
help:
	@echo "Available targets:"
	@echo "  install        - Install the package using UV"
	@echo "  build          - Build the package wheel"
	@echo "  clean          - Remove build artifacts and caches"
	@echo "  test           - Run tests with pytest"
	@echo "  test-c         - Build and run the standalone C library tests (CTest)"
	@echo "  coverage       - Run tests with coverage report"
	@echo "  docs           - Build Sphinx documentation"
	@echo "  lint           - Run linting (ruff for Python, clang-tidy for C)"
	@echo "  format         - Format all code (Python and C)"
	@echo "  format-python  - Format Python code with ruff"
	@echo "  format-c       - Format C code with clang-format"
	@echo "  check          - Run linting and tests"
	@echo "  all            - Clean, format, lint, test, and build docs"

# Installation targets
install:
	uv sync

build:
	uv pip install build
	uv run python -m build

# Cleaning targets
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.pyd" -delete
	@echo "Clean complete!"

# Testing targets
test:
	uv run pytest tests/ -v

test-c:
	cmake -S . -B build/c-tests -DKMEANS_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
	cmake --build build/c-tests --target test_kmeans_lib
	ctest --test-dir build/c-tests --output-on-failure

coverage:
	@echo "Running tests with coverage (Python code only)..."
	uv run pytest tests/ --cov=src/kmeans --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo ""
	@echo "Coverage report generated!"
	@echo "HTML report: htmlcov/index.html"
	@echo "Note: C extension code is not included in coverage"

# Documentation targets
docs:
	@echo "Building documentation..."
	@echo "Syncing dependencies..."
	uv sync --group dev
	@echo "Building HTML documentation..."
	cd docs && uv run sphinx-build -b html . _build/html
	@echo ""
	@echo "Documentation built! Open docs/_build/html/index.html"
	@echo "Note: If autodoc fails, run 'uv pip install -e .' first"

docs-clean:
	@echo "Cleaning documentation build..."
	rm -rf docs/_build/

# Formatting targets
format: format-python format-c

format-python:
	@echo "Formatting Python code with ruff..."
	uv run ruff format src/ tests/ || pip install ruff && ruff format src/ tests/

format-c:
	@echo "Formatting C code with clang-format..."
	@if command -v clang-format >/dev/null 2>&1; then \
		find src/ include/ tests/c -name "*.c" -o -name "*.h" | xargs clang-format -i; \
		echo "C code formatted!"; \
	else \
		echo "Warning: clang-format not found, skipping C formatting"; \
	fi

# Linting targets
lint: lint-python lint-c

lint-python:
	@echo "Linting Python code with ruff..."
	uv run ruff check src/ tests/ || pip install ruff && ruff check src/ tests/

lint-c:
	@echo "Checking C code..."
	@if command -v clang-tidy >/dev/null 2>&1; then \
		find src/ -name "*.c" | xargs clang-tidy --quiet; \
	else \
		echo "Warning: clang-tidy not found, skipping C linting"; \
	fi

# Combined targets
check: lint test test-c

all: clean format lint test docs
	@echo "All tasks completed successfully!"