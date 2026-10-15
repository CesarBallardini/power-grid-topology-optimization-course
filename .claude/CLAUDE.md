# Project: Power Grid Topology Optimization Course

## Project Goals

Maintain a five-semester university course (36 chapters, 8000+ lines) that takes students from college algebra through GPU-accelerated topology optimization (ToOp). The course is published as an MkDocs book with versioned releases via GitHub Pages.

## Course Structure

**Six parts, 36 chapters, five semesters:**
- Part 0: Orientation and setup (1 chapter)
- Part I: Mathematical foundations (6 chapters)
- Part II: Electricity and circuits (5 chapters)
- Part III: Power systems and grid computations (11 chapters)
- Part IV: Optimization (7 chapters)
- Part V: Computing (5 chapters)
- Part VI: Capstone (1 chapter)

Chapter dependencies are defined in `docs/index.md` (Mermaid flowchart) and enforce a teaching order across semesters.

## Development Workflow

### Adding or Updating Content

1. **Edit chapter files** in `docs/part-*/ch*.md` — they auto-build with live reload (`make docs-serve`)
2. **Check formatting** — run `make lint` before committing (ruff enforces style)
3. **Verify links** — `make docs` (strict mode catches broken internal links; external links are checked locally only, with `make links`, never in CI)
4. **Update glossary** — if you add new acronyms, run `make abbreviations` to regenerate tooltips
5. **Commit with Conventional Commits** — commitizen (`cz_conventional_commits`) enforces `<type>: <message>`; the book uses `docs:` and `fix:`, CI changes `ci:`

On commit, `markdown-link-check` checks every link in each staged file, not only the lines changed, so an old link that fails can block an unrelated edit. pre-commit runs with the **staged** `.pre-commit-config.yaml` and `.markdown-link-check.json` (unstaged changes are stashed). After opening the link to confirm it works, skip only that hook: `SKIP=markdown-link-check git commit ...`.

### Maintenance Tasks

- **`make check-books`** — Verify every archive.org link still works (network test, run weekly)
- **`make check-toop`** — Validate ToOp source paths cited in the book (requires `../ToOp` sibling repo)
- **`make links`** — Check every external link (network, ~2.5 min). Local only, never in CI: GitHub runner IPs are blocked by many hosts. `202` counts as alive, since eur-lex answers every bot with a 202 firewall challenge. A few links fail from anywhere (techrxiv 403, tudelft/gmu/ou time out in the checker); confirm by opening the link, not by trusting the exit code
- **`make test`** — Run unit tests on maintenance tools (72 tests)
- **`make docs`** — Full strict build: any warning fails the build

### Release Process

1. Create a git tag: `git tag v1.0.0`
2. Push the tag: `git push origin v1.0.0`
3. GitHub Actions publishes to https://katra.ballardini.com.ar/power-grid-topology-optimization-course/
4. `mike` automatically updates `/latest/` and maintains version menu
5. A push to `main` that changes the book (or `gh workflow run docs`) refreshes `/dev/` without a tag

Tag format: `vX.Y.Z` (PEP 440, enforced by commitizen). Squash merges take PR title as commit subject.

`gh-pages` is the published site, written only by `mike` (one folder per version, plus `latest` and `dev`). It shares no history with `main`, so GitHub's "behind/ahead" for it is meaningless: never merge into it or edit it by hand. Pages is configured as "Deploy from a branch" → `gh-pages` `/` (not "GitHub Actions").

## CI

One workflow file per independent job, each with a README badge:

- `check.yml` — pre-commit hooks (without `markdown-link-check`) and pytest
- `build.yml` — `mkdocs build --strict`
- `docs.yaml` — publishes with `mike`: `v*` tag → that version + `latest`; push to `main` touching the book, or manual run → `dev`
- `validate_pr_title.yaml` — Conventional Commits check on the PR title

There is no link-check workflow on purpose. The `main` ruleset has no required status checks; if you add some, use the job names `check`, `build` and `validate-pr-title`.

## Windows notes

- `uv` script launchers in `.venv/Scripts` (`pre-commit.exe`, `cz.exe`, ...) can break with "uv trampoline failed to canonicalize script path". Local hooks and Make targets therefore call tools as modules (`python -m pre_commit`, `python -m commitizen`). Repair a broken venv with `uv sync --all-groups --frozen --reinstall`.
- `markdown-link-check` sometimes reports "files were modified by this hook" with no dead links listed; rerun.
- Commands typed with the `!` prefix in Claude Code run in Git Bash, not PowerShell: set a variable for one command as `SKIP=hook git commit ...`. In Git Bash, `$env:SKIP=...` silently sets nothing.
- `uv run` can recreate `.venv` (for example after a Python upgrade) with only the default groups, so `make docs` fails with "Failed to spawn: mkdocs". Restore every group with `uv sync --all-groups --frozen`.

## Chapter Conventions

- **Prerequisites line** at the top of each chapter lists all dependencies (not just parents)
- **"Why it matters for ToOp"** section explains the relevance to the optimization engine
- **Lab** section describes hands-on exercises with required tools (Jupyter, NumPy, JAX, etc.)
- **Exercises** are graded tasks; labs are lab-based checkpoints per semester
- **Resources** cite freely available books (archive.org labels: FREE, BORROW, PRINT-DISABLED)
- **Tools in labs** are free and run on Windows, macOS and Linux. Commercial or Windows-only tools appear only as optional steps or resources. Example: PowerWorld Simulator (free demo limited to 13 buses, so IEEE 14 does not fit) is in the course guide's tool table, the Chapter 15 lab (step 6), and its free training modules are cited in chapters 13, 15, 17 and 19

## Bibliography & Archive.org

Every book has an access label verified against archive.org in September 2026. Update these only if:
- A newer edition becomes available
- Archive.org status changes (use `check_archive_labels.py` to verify)
- A Spanish translation is found (list it alongside English)

Labels:
- **FREE**: openly licensed or author-hosted
- **BORROW**: archive.org lending library (free account can borrow)
- **PRINT-DISABLED**: restricted to certified users
- **NOT ON IA**: use university library
- **USER-UPLOAD (ES)**: last resort for Spanish scans (not stable, check university first)

## Tools

All tools are plain Python scripts in `tools/`:

- **`gen_abbreviations.py`** — Auto-generate acronym tooltips from `appendices/d-glossary.md` to `includes/abbreviations.md`. Called by build and `make abbreviations`.
- **`check_archive_labels.py`** — Validate every archive.org link label against live API (slow, network test).
- **`check_toop_paths.py`** — Verify every ToOp source path cited in the book exists. Requires `TOOP_REPO=../ToOp` env var.

## Key Files

- `mkdocs.yml` — site config, navigation structure, Material theme settings
- `pyproject.toml` — uv environment (dev/doc groups), commitizen config, pytest settings
- `.pre-commit-config.yaml` — git hooks: ruff (lint/format), commitizen (message validation), markdown-link-check
- `.markdown-link-check.json` — link checker settings; the only ignore rule is the plugin-injected `../print_page/` link, with no per-host exceptions; alive status codes are 200, 202 (bot challenges such as eur-lex's) and 206
- `Makefile` — convenience targets for all common tasks

## Style & Tone

- Write in **second person** ("you") for exercises; **neutral** for exposition
- Use **en-dashes** (–) for ranges and emphasis, not hyphens
- Math in `$...$` (inline) or `$$...$$` (display) for MathJax rendering
- Code blocks use syntax highlighting; cite ToOp source as `path/file.py:function_name`
- Acronyms are auto-tooltipped from the glossary; don't define them inline

## Remember

- The course is not a reference manual; it teaches *why* (intuition first, then math)
- Students start at sophomore level; assume basic calculus/linear algebra/physics only
- Every chapter should answer a question a student has already asked (set by Chapter 0)
- The capstone is the payoff: reading and debugging the real ToOp codebase
