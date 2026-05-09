VENV ?= /Users/vivekanandakota/Agentic-AI/AI-Finance-Assistant/.venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
STREAMLIT := $(VENV)/bin/streamlit

.PHONY: install test test-unit test-security test-integration test-all lint format run clean help

help:
	@echo "Targets:"
	@echo "  install          Install dependencies"
	@echo "  run              Start the Streamlit app"
	@echo "  test             Run unit + security tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-all         Run all tests with coverage"
	@echo "  lint             Run ruff linter"
	@echo "  format           Auto-format with ruff"
	@echo "  clean            Remove generated files"

install:
	$(PIP) install -r requirements.txt

run:
	PYTHONPATH=. $(STREAMLIT) run src/ui/streamlit_app.py --server.port 8501

test:
	PYTHONPATH=. $(PYTEST) tests/unit tests/security -v

test-unit:
	PYTHONPATH=. $(PYTEST) tests/unit -v

test-security:
	PYTHONPATH=. $(PYTEST) tests/security -v

test-integration:
	PYTHONPATH=. $(PYTEST) tests/integration -v

test-all:
	PYTHONPATH=. $(PYTEST) tests/ -v --cov=src --cov-report=term-missing

lint:
	$(VENV)/bin/ruff check src/ tests/ || true

format:
	$(VENV)/bin/ruff check --fix src/ tests/ && $(VENV)/bin/ruff format src/ tests/ || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	rm -f data/calls.db
