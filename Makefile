# =============================================================================
# Engram Makefile
# Convenient shortcuts for common development tasks
# =============================================================================

.PHONY: help docker-up docker-down docker-reset docker-status docker-logs \
        install dev test lint format type-check clean build docs

# Default target
help:
	@echo ""
	@echo "Engram Development Commands"
	@echo "==========================="
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up      Start PostgreSQL (auto-detect port)"
	@echo "  make docker-down    Stop PostgreSQL"
	@echo "  make docker-reset   Reset database (delete all data)"
	@echo "  make docker-status  Show container status"
	@echo "  make docker-logs    Follow container logs"
	@echo "  make docker-shell   Open psql shell"
	@echo ""
	@echo "Development:"
	@echo "  make install        Install package in editable mode"
	@echo "  make dev            Install with dev dependencies"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make lint           Run linter (ruff)"
	@echo "  make format         Format code (ruff)"
	@echo "  make type-check     Run type checker (mypy)"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Build & Release:"
	@echo "  make build          Build package"
	@echo "  make docs           Build documentation"
	@echo ""

# =============================================================================
# Docker Commands
# =============================================================================

docker-up:
	@./scripts/docker-setup.sh

docker-down:
	@./scripts/docker-setup.sh --down

docker-reset:
	@./scripts/docker-setup.sh --reset

docker-status:
	@./scripts/docker-setup.sh --status

docker-logs:
	@./scripts/docker-setup.sh --logs

docker-shell:
	@./scripts/docker-setup.sh --shell

# =============================================================================
# Development Commands
# =============================================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev,all]"
	pre-commit install

test:
	pytest tests/unit -v

test-cov:
	pytest tests/unit -v --cov=src/engram --cov-report=term-missing --cov-report=html

test-integration:
	pytest tests/integration -v --run-integration

lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

type-check:
	mypy src

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Build Commands
# =============================================================================

build: clean
	python -m build

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# =============================================================================
# Quick Start
# =============================================================================

quickstart: docker-up dev
	@echo ""
	@echo "✅ Engram is ready for development!"
	@echo "   Run 'python examples/basic_usage.py' to test"
