.DEFAULT_GOAL := help

.PHONY: help install lint format test docs docs-serve check-books check-toop \
        abbreviations precommit clean

help: ## Show this list of available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the environment (from the committed lockfile) and install the git hooks
	uv sync --all-groups --frozen
	uv run --frozen pre-commit install

lint: ## Check formatting and lint rules without modifying files
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format: ## Auto-fix formatting and lint issues
	uv run --frozen ruff format .
	uv run --frozen ruff check --fix .

test: ## Run the test suite (live archive.org tests excluded; `pytest -m network` runs them)
	uv run --frozen pytest

docs: ## Build the course book (--strict: warnings are failures)
	uv run --frozen mkdocs build --strict
	uv run --frozen python tools/gen_abbreviations.py --check

docs-serve: ## Serve the course book locally with live reload
	uv run --frozen mkdocs serve

check-books: ## Check every archive.org access label against archive.org (network)
	uv run --frozen python tools/check_archive_labels.py docs

check-toop: ## Check that every cited ToOp path exists (TOOP_REPO, default ../ToOp)
	uv run --frozen python tools/check_toop_paths.py docs

abbreviations: ## Regenerate the acronym tooltips (includes/abbreviations.md) from the glossary
	uv run --frozen python tools/gen_abbreviations.py

precommit: ## Run all pre-commit hooks against every file
	uv run --frozen pre-commit run --all-files

clean: ## Remove the built site and tool caches
	rm -rf site/ .pytest_cache/ .ruff_cache/ .cache/
