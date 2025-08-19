# Makefile for LangExtract Anthropic Provider

.PHONY: help install install-dev clean lint format test test-unit test-integration test-coverage build check-build release

# Default target
help:
	@echo "LangExtract Anthropic Provider - Development Commands"
	@echo "===================================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install package using UV"
	@echo "  install-dev  Install with development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run code linting with ruff"
	@echo "  typecheck    Run type checking with mypy"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test         Run comprehensive test suite"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration  Run Anthropic API integration tests"
	@echo "  test-coverage     Run tests with coverage report"
	@echo ""
	@echo "Build Commands:"
	@echo "  build        Build distribution packages"
	@echo "  check-build  Validate package build"
	@echo ""
	@echo "Release Commands:"
	@echo "  release      Run full release process"
	@echo "  bump-patch   Bump patch version (1.0.0 -> 1.0.1)"
	@echo "  bump-minor   Bump minor version (1.0.0 -> 1.1.0)"
	@echo "  bump-major   Bump major version (1.0.0 -> 2.0.0)"

# Setup commands
install:
	uv sync

install-dev:
	uv sync --all-extras --dev

# Development commands
clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

format:
	uv run black .
	uv run isort .

lint:
	uv run ruff check .

typecheck:
	uv run mypy langextract_anthropic

# Testing commands
test:
	python scripts/run_tests.py

test-unit:
	uv run pytest tests/test_provider_unit.py -v

test-integration:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "⚠️  Set ANTHROPIC_API_KEY to run integration tests"; \
		exit 1; \
	fi
	uv run python tests/test_anthropic_parameters.py

test-coverage:
	uv run pytest tests/ --cov=langextract_anthropic --cov-report=html --cov-report=term-missing

# Build commands
build: clean
	uv build

check-build:
	python scripts/check_build.py

# Release commands
release:
	python scripts/release.py

bump-patch:
	python scripts/bump_version.py patch

bump-minor:
	python scripts/bump_version.py minor

bump-major:
	python scripts/bump_version.py major

# Quality gates (for CI)
ci-check: lint typecheck test-unit build check-build
	@echo "✅ All CI checks passed!"

# Development workflow
dev-check: clean format lint typecheck test
	@echo "✅ Development checks completed!"