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
3. **Verify links** — `make docs` (strict mode catches broken internal links; external links checked via pre-commit)
4. **Update glossary** — if you add new acronyms, run `make abbreviations` to regenerate tooltips
5. **Commit with Conventional Commits** — hook enforces `docs: <message>` or `fix: <message>` format

### Maintenance Tasks

- **`make check-books`** — Verify every archive.org link still works (network test, run weekly)
- **`make check-toop`** — Validate ToOp source paths cited in the book (requires `../ToOp` sibling repo)
- **`make test`** — Run unit tests on maintenance tools (72 tests)
- **`make docs`** — Full strict build: any warning fails the build

### Release Process

1. Create a git tag: `git tag v1.0.0`
2. Push the tag: `git push origin v1.0.0`
3. GitHub Actions publishes to https://cesarballardini.github.io/curso-optimizacion-redes-electricas/
4. `mike` automatically updates `/latest/` and maintains version menu

Tag format: `vX.Y.Z` (PEP 440, enforced by commitizen). Squash merges take PR title as commit subject.

## Chapter Conventions

- **Prerequisites line** at the top of each chapter lists all dependencies (not just parents)
- **"Why it matters for ToOp"** section explains the relevance to the optimization engine
- **Lab** section describes hands-on exercises with required tools (Jupyter, NumPy, JAX, etc.)
- **Exercises** are graded tasks; labs are lab-based checkpoints per semester
- **Resources** cite freely available books (archive.org labels: FREE, BORROW, PRINT-DISABLED)

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
- `.markdown-link-check.json` — skip lists for unreliable domains (institutional servers, paywalls)
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
