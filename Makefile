.DEFAULT_GOAL := help

.PHONY: help install lint format test docs docs-serve check-books check-toop test-toop test-labs \
        abbreviations precommit links clean

help: ## Show this list of available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the environment (from the committed lockfile) and install the git hooks
	uv sync --all-groups --frozen
	uv run --frozen python -m pre_commit install

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

test-toop: ## Run the chapter tests marked `toop` with ToOp's own interpreter (TOOP_PYTHON, default: the venv in ../ToOp)
	@py="$(TOOP_PYTHON)"; \
	if [ -z "$$py" ]; then py=$$(ls ../ToOp/.venv/bin/python ../ToOp/.venv/Scripts/python.exe 2>/dev/null | head -1); fi; \
	if [ -z "$$py" ]; then echo "ToOp interpreter not found: set TOOP_PYTHON or create ../ToOp/.venv"; exit 1; fi; \
	"$$py" -m pytest -m toop -o addopts="" -p no:cacheprovider -W ignore $$(find docs -name 'test_*_toop.py') tests/test_notebooks.py

test-labs: ## Run the chapter tests and notebooks marked `labs` inside the labs container (docker/labs; build it first)
	@repo=$$(pwd); case "$$(uname -s)" in MINGW*|MSYS*) repo=$$(cygpath -m "$$repo");; esac; \
	MSYS_NO_PATHCONV=1 docker run --rm -v "$$repo:/course" -v course-labs-data:/root/data_grid2op -w /course \
	  course-labs:latest python -m pytest -m labs -o addopts="" -p no:cacheprovider -W ignore \
	  $$(find docs -name 'test_*_labs.py') tests/test_notebooks.py

check-toop: ## Check that every cited ToOp path exists (TOOP_REPO, default ../ToOp)
	uv run --frozen python tools/check_toop_paths.py docs

abbreviations: ## Regenerate the acronym tooltips (includes/abbreviations.md) from the glossary
	uv run --frozen python tools/gen_abbreviations.py

precommit: ## Run all pre-commit hooks against every file
	uv run --frozen python -m pre_commit run --all-files

links: ## Check every external link in the book (network; some hosts fail from anywhere, judge by opening the link)
	uv run --frozen python -m pre_commit run markdown-link-check --all-files

clean: ## Remove the built site and tool caches
	rm -rf site/ .pytest_cache/ .ruff_cache/ .cache/
