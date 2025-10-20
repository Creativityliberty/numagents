.PHONY: help install install-dev test test-cov lint format type-check clean build docs pre-commit

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Nüm Agents SDK - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in production mode
	pip install -e .

install-dev: ## Install the package in development mode with all dev dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests without coverage
	pytest -v

test-cov: ## Run tests with coverage report
	pytest --cov=num_agents --cov-report=term-missing --cov-report=html --cov-report=xml

test-watch: ## Run tests in watch mode
	pytest-watch

test-parallel: ## Run tests in parallel
	pytest -n auto

lint: ## Run all linters (black, isort, ruff, bandit)
	@echo "Running Black..."
	black --check .
	@echo "\nRunning isort..."
	isort --check-only .
	@echo "\nRunning Ruff..."
	ruff check .
	@echo "\nRunning Bandit (security)..."
	bandit -r num_agents -c pyproject.toml

format: ## Auto-format code with black and isort
	black .
	isort .
	ruff check --fix .

type-check: ## Run mypy type checking
	mypy num_agents

check: lint type-check test ## Run all checks (lint, type-check, test)

clean: ## Clean up build artifacts, cache, and coverage files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build distribution packages
	python -m build

build-check: build ## Build and check package with twine
	twine check dist/*

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

coverage-report: ## Open HTML coverage report in browser
	@python -m webbrowser htmlcov/index.html

cli-help: ## Show CLI help
	num-agents --help

cli-generate: ## Generate example agent
	num-agents generate examples/agent.yaml

cli-audit: ## Run audit on generated agent
	@if [ -d "ExampleAgent" ]; then \
		num-agents audit ExampleAgent --agent-spec examples/agent.yaml; \
	else \
		echo "Error: ExampleAgent directory not found. Run 'make cli-generate' first."; \
	fi

cli-graph: ## Generate logical graph for example agent
	@if [ -d "ExampleAgent" ]; then \
		num-agents graph ExampleAgent; \
	else \
		echo "Error: ExampleAgent directory not found. Run 'make cli-generate' first."; \
	fi

dev-setup: install-dev ## Complete development environment setup
	@echo "Development environment setup complete!"
	@echo "Run 'make help' to see available commands."

version: ## Show current version
	@python -c "from num_agents import __version__; print(f'Nüm Agents SDK v{__version__}')"

info: ## Show development environment info
	@echo "Python version:"
	@python --version
	@echo "\nInstalled packages:"
	@pip list | grep -E "(num-agents|pytest|black|ruff|mypy)"

.PHONY: all
all: format lint type-check test ## Run format, lint, type-check, and test
