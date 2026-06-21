.PHONY: install dev test test-fast lint typecheck format check clean run info

# Default target
all: check

# Install package in development mode
install:
	pip install -e .

# Install with development dependencies
dev:
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v --tb=short

# Run tests (fast, no verbose)
test-fast:
	python -m pytest tests/ --tb=no -q

# Run linter
lint:
	python -m flake8 src/brain_agent/ --max-line-length=120 --exclude=src/brain_agent/generated/

# Run type checker
typecheck:
	python -m mypy src/brain_agent/ --ignore-missing-imports

# Format code
format:
	python -m black src/brain_agent/ --line-length=120 --exclude='generated/'
	python -m isort src/brain_agent/ --skip-glob='*/generated/*'

# Run all checks (lint + typecheck + test)
check: lint typecheck test-fast

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete

# Run the simulation
run:
	brain-agent run --steps 100

# Show system info
info:
	brain-agent info
