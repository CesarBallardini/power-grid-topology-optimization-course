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

## No AI attribution

Nothing in this repository names an AI assistant, model or vendor as an author or tool: no `Co-Authored-By`
lines for AI models in commit messages, no "Generated with ..." footers in pull request descriptions, no mentions
of Claude, Anthropic, ChatGPT, LLMs and the like in code comments, docstrings, book pages or solutions. This
rule overrides the assistant's own default attribution reminders. Commit messages end after the body.

## Development Workflow

### Adding or Updating Content

1. **Edit chapter files** in `docs/part-*/chNN-slug/index.md` — they auto-build with live reload (`make docs-serve`)
2. **Check formatting** — run `make lint` before committing (ruff enforces style)
3. **Verify links** — `make docs` (strict mode catches broken internal links; external links are checked locally only, with `make links`, never in CI)
4. **Update glossary** — if you add new acronyms, run `make abbreviations` to regenerate tooltips
5. **Commit with Conventional Commits** — commitizen (`cz_conventional_commits`) enforces `<type>: <message>`; the book uses `docs:` and `fix:`, CI changes `ci:`

On commit, `markdown-link-check` checks every link in each staged file, not only the lines changed, so an old link that fails can block an unrelated edit. pre-commit runs with the **staged** `.pre-commit-config.yaml` and `.markdown-link-check.json` (unstaged changes are stashed). After opening the link to confirm it works, skip only that hook: `SKIP=markdown-link-check git commit ...`.

### Python with uv

Use `uv` to install Python and to manage dependencies; do not use `pip`, `python -m venv` or OS installers. Install
interpreters with `uv python install`, add libraries with `uv add --group <group>` (never by hand-editing
`pyproject.toml` without `uv lock`), restore the course venv with `uv sync --all-groups --frozen`, and run tools with
`uv run`. Requirements always live in a `pyproject.toml` with its `uv.lock`, never in a `requirements.txt`. Scratch
environments are `uv venv` + `uv pip install`. Dockerfiles copy a pinned `uv` binary
(`COPY --from=ghcr.io/astral-sh/uv:<version> /uv /uvx /bin/`), copy the image's `pyproject.toml` and `uv.lock`, and
install with `uv sync --frozen --no-install-project`. ToOp's
own environment (`../ToOp/.venv`, the `toop-course` image) is managed by ToOp's uv setup: never install into it.

### Maintenance Tasks

- **`make check-books`** — Verify every archive.org link still works (network test, run weekly)
- **`make check-toop`** — Validate ToOp source paths cited in the book (requires `../ToOp` sibling repo)
- **`make links`** — Check every external link (network, ~2.5 min). Local only, never in CI: GitHub runner IPs are blocked by many hosts. `202` counts as alive, since eur-lex answers every bot with a 202 firewall challenge. A few links fail from anywhere (techrxiv 403, udemy 403 and realpython.com 403 for scripted requests, tudelft/gmu/ou time out in the checker); confirm by opening the link, not by trusting the exit code
- **`make test`** — Run the unit tests of the maintenance tools and of the chapters' `src/` code (ToOp-dependent tests skip)
- **`make test-toop`** — Run the chapter tests marked `toop` with ToOp's own interpreter (`TOOP_PYTHON`, default `../ToOp/.venv`); the course venv never gets ToOp installed, because ToOp pins its own JAX and Python
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
- `test-labs.yml` — the `labs` tests and notebooks: syncs `docker/labs` (`uv sync --project docker/labs --frozen`) straight onto the runner instead of building the image, with the Grid2Op dataset cached in `~/data_grid2op`; the pytest command is the one `make test-labs` runs in the container, so keep the two in step
- `test-octave.yml` — the `%!test` blocks of the chapters' `.m` files (`tests/test_octave_sources.py`): Octave from Ubuntu's packages, MATPOWER (`MATPOWER_VERSION`) from its GitHub release, cached in `~/.local/share/matpower` and passed as `$MATPOWER`, so a missing MATPOWER fails the run instead of letting its blocks skip
- `docs.yaml` — publishes with `mike`: `v*` tag → that version + `latest`; push to `main` touching the book, or manual run → `dev`
- `validate_pr_title.yaml` — Conventional Commits check on the PR title

There is no link-check workflow on purpose, and no ToOp workflow: `toop` tests run locally with `make test-toop`. The `main` ruleset has no required status checks; if you add some, use the job names `check`, `build`, `test-labs`, `test-octave` and `validate-pr-title`.

## Windows notes

- `uv` script launchers in `.venv/Scripts` (`pre-commit.exe`, `cz.exe`, ...) can break with "uv trampoline failed to canonicalize script path". Local hooks and Make targets therefore call tools as modules (`python -m pre_commit`, `python -m commitizen`). Repair a broken venv with `uv sync --all-groups --frozen --reinstall`.
- `markdown-link-check` sometimes reports "files were modified by this hook" with no dead links listed; rerun.
- Commands typed with the `!` prefix in Claude Code run in Git Bash, not PowerShell: set a variable for one command as `SKIP=hook git commit ...`. In Git Bash, `$env:SKIP=...` silently sets nothing.
- ToOp can be run without the container: `../ToOp/.venv/Scripts/python.exe` (JAX on CPU, pandapower, pypowsybl) imports the solver, which is enough to verify lab code when Docker Desktop is not running. Create scratch virtual environments under a short path such as `%LOCALAPPDATA%\Temp\v7`: under the long scratchpad path, networkx fails to import (Windows path length).
- Octave (Chocolatey) installs to `C:\Program Files\GNU Octave\Octave-<version>\mingw64\bin\octave-cli.exe` without adding it to the PATH; `tests/test_octave_sources.py` looks there. MATPOWER 8.1 is unzipped under `%LOCALAPPDATA%\matpower\matpower8.1` and loaded per run with `install_matpower(1, 0, 0)`; set `MATPOWER=<folder>` to use another copy
- `uv run` can recreate `.venv` (for example after a Python upgrade) with only the default groups, so `make docs` fails with "Failed to spawn: mkdocs". Restore every group with `uv sync --all-groups --frozen`.

## Chapter Conventions

- **Prerequisites line** at the top of each chapter lists all dependencies (not just parents)
- **"Why it matters for ToOp"** section explains the relevance to the optimization engine
- **Lab** section describes hands-on exercises with required tools (Jupyter, NumPy, JAX, etc.)
- **Exercises** are graded tasks; labs are lab-based checkpoints per semester
- **Resources** cite freely available books (archive.org labels: FREE, BORROW, PRINT-DISABLED)
- **Video courses** in Resources are online university or college courses that are well known and recognized (MIT OCW, NPTEL, Coursera/edX from named universities, faculty lecture series such as Brunton's), or courses built on a popular textbook of the subject. Prefer college level; precalculus material stays only as a labelled "Refresher". Udemy courses are allowed where no recognized course fits the topic, marked as paid and not university. Mark paid or subscriber-only courses as such (for example **Real Python subscribers only**; the maintainer is a subscriber). **Every online course states whether it can be audited for free**, with one of: `free` (NPTEL, OCW, YouTube, Hugging Face), `free to audit` (edX), `paid, with a free trial; no free audit shown` (Coursera), `paid` (Udemy), `Real Python subscribers only`. Check the course page (see "Checking course pages") and never assume
- **Real Python free tutorials** are valid resources on their own: cite them without looking for a companion video
- **Tools in labs** are free and run on Windows, macOS and Linux. Commercial or Windows-only tools appear only as optional steps or resources. Example: PowerWorld Simulator (free demo limited to 13 buses, so IEEE 14 does not fit) is in the course guide's tool table, the Chapter 15 lab (step 6), and its free training modules are cited in chapters 13, 15, 17 and 19
- **Free software first**: whenever a MATLAB solution is used, write the equivalent Octave solution (MATPOWER runs on both); proprietary software is not forbidden, but every exercise must be solvable with free tools
- **Chapter layout**: each chapter is a folder, `docs/part-*/chNN-slug/index.md`, so its URL is `.../chNN-slug/`; the `navigation.indexes` feature is off on purpose, since it would turn the first `index.md` of each Part into that Part's landing page and hide the chapter
- **Chapter code lives in `src/`, is included, never copied.** Every program a chapter shows (Python, Octave, data files) is a file in `docs/part-*/chNN-slug/src/`, pulled into the page with a snippet inside the fence: ```` ```python title="file.py" ```` then `--8<-- "docs/part-X/chNN-slug/src/file.py"` (optionally `:section`, delimited in the file by `# --8<-- [start:section]` / `# --8<-- [end:section]`). Paths are relative to the repository root and a missing file fails the build (`check_paths`). Shell commands to type (`docker compose ...`) stay inline. Directly below every included snippet goes a link to that chapter's `src/` folder on GitHub (`https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-X/chNN-slug/src`, with `{ target="_blank" }` so it opens in a new tab), so students can browse and download the files. The link 404s until the branch is merged, so the link check in the pre-commit hook may flag it: skip that hook as usual MkDocs serves the files with the book, so pages may also link to them (`src/file.py`), and the strict build checks those links
- **Every `src/` file has unit tests next to it**: `test_<name>.py` (pytest collects `docs/` too, see `testpaths`; the test's folder is on `sys.path`, so `from file import ...` works; keep test basenames unique across chapters). Octave files end with `%!test` blocks; `tests/test_octave_sources.py` runs them when Octave is on the PATH and skips otherwise; `test-octave.yml` runs them in CI with MATPOWER. A block that needs MATPOWER skips itself with `MATPOWER not on the path, test skipped`, and the pytest wrapper fails on that message whenever MATPOWER was found. Tests that need ToOp are named `test_<name>_toop.py` and carry `pytestmark = pytest.mark.toop` plus `pytest.importorskip('toop_engine_dc_solver')`; `make test` skips them and `make test-toop` collects only those files with ToOp's interpreter (`-o addopts="" -W ignore`: ToOp's environment lacks the `labs` libraries and emits its own warnings). Libraries the chapter code needs are in the `labs` dependency group (`uv add --group labs ...`); pulp is pinned `>=3,<4` because 4.0 replaced the `LpVariable.dicts` API. pytest treats warnings as errors: a library warning needs a specific `filterwarnings` entry in `pyproject.toml` with its reason
- **Labs container**: libraries that are not in the course `.venv` (Grid2Op, LightSim2Grid, Gymnasium, Stable-Baselines3 with CPU torch, QDax, pymoo, moocore) live in the Docker image `course-labs:latest` built from `docker/labs` (dependencies in its own `pyproject.toml` + `uv.lock`, installed with `uv sync --frozen`; README with sizes and the one-time download of `l2rpn_case14_sandbox` into the named volume `course-labs-data`). Chapter code that needs them is named `*_labs.py`, its tests `test_<name>_labs.py` with `pytestmark = pytest.mark.labs` plus `pytest.importorskip('grid2op')` (or `qdax`, `pymoo`, `moocore`), and notebooks `*_labs.ipynb`; `make test` skips them and `make test-labs` runs them in the container (`docker run` with the repository bind-mounted at `/course`; in Git Bash with `MSYS_NO_PATHCONV=1` and a `C:/...` mount path). Solutions quote only numbers produced in the container, with the CPU budget stated
- **Solutions** of a chapter's labs and exercises are all in `chNN-slug/solutions.md` (not in `mkdocs.yml`'s nav; the chapter links to it from each task and from the Exercises intro), with the numbers and the ToOp commit that produced them. Any program written to find a solution lives in that chapter's `src/` and prints every number the solution quotes; never quote results from scratch scripts
- Running a script inside `docs/` leaves a `__pycache__` that MkDocs would copy into the site: delete it before building (`find docs -name __pycache__ -prune -exec rm -rf {} +`)
- Math belongs in the page, not in docstrings: MathJax does not render inside code blocks, so a docstring states a formula in plain or Unicode text (`x² + y² − 4`) and the chapter prose next to the include carries the typeset version
- **Notebooks** are thin drivers, three per chapter in `src/`: `examples.ipynb` (the worked examples run live), `lab.ipynb` (the lab steps with guidance and empty answer cells) and `exercises.ipynb` (the exercise statements with empty answer cells). They import the chapter's `src` modules (`from case7 import dispatch`; the notebook's folder is the working directory) and never copy their code. Write them from a percent-format script with `tools/make_notebook.py` (`# %% [markdown]` / `# %%` cells), commit them without outputs (the `nbstripout` hook strips them), and `tests/test_notebooks.py` executes every notebook headlessly in `make test`; a notebook that needs ToOp is named `*_toop.ipynb` and runs under `make test-toop`. Written solutions stay in `solutions.md`, not in notebooks. The chapter links its notebooks from the Lab and Exercises sections (`[lab.ipynb](src/lab.ipynb)`) and pages served by MkDocs can be downloaded from there
- **Background agents log their progress** in `.progress/<chNN-slug>.md`, as described in "Background agents and progress logs" below

## Background agents and progress logs

Every background agent keeps a log in `.progress/<slug>.md` (gitignored), whatever its task: a chapter, a container, CI, a benchmark. The log must let anyone check or restart the work after a network outage, a usage limit, or a crash of the agent or of the session that launched it, so that only the item in flight is lost. Its parts, in this order:

1. **Task**, written before any work: the task statement as given, or a self-contained restatement of it. It covers the goal, the scope, which files the agent may and may not touch, the constraints, and where the results go. A new agent given only the log must be able to continue.
2. **Status table**: one row per item (lab step, exercise, notebook, workflow, run), each `todo`, `started`, `done` or `blocked: reason`.
3. **Resources**: everything the agent created outside the repository, by name: scratch folders, Docker containers, images and volumes, and background processes. Mark each one when it is removed.
4. **Log**: timestamped entries (date, time and time zone), appended before starting and after finishing each item, with the outcome: numbers, file paths, and decisions with their reasons. Write the entry before a long-running step, not only after it.
5. **`COMPLETE`**, a final line written once verification has passed and the resources are cleaned up.

When an agent starts, it reads its log first if one exists, and resumes from the first item that is not `done`. A `started` item may be half done, so the agent first checks that the files and resources the log names are in the state it describes. The session that launches an agent puts the log path in the prompt. After a crash it restarts every log that has no `COMPLETE` line.

## Checking course pages

Course sites (Udemy, Coursera, Real Python, VideoLectures) often answer scripted requests with 403 or 404, and WebFetch is refused by some. Before judging a link dead or a course unverifiable, fetch the public page with `curl` and a full set of Chrome headers; a user agent alone is not enough (Udemy gave 403 with only the user agent and 200 with all of these). Run it in Git Bash:

```bash
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
curl -s -L -A "$UA" \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' \
  -H 'Accept-Language: en-US,en;q=0.9' \
  -H 'sec-ch-ua: "Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"' \
  -H 'sec-ch-ua-mobile: ?0' -H 'sec-ch-ua-platform: "Windows"' \
  -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: none' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -o page.html -w '%{http_code}\n' "$URL"
```

- Free-audit status, from the saved HTML: edX pages have `"isAccessibleForFree":true`, an `auditTrack` string and a "How to Audit a Course" link, with the certificate price in the JSON-LD `offers` ("Partially Free"); Coursera pages show "Free trial", "Financial Aid available" and no audit option (the "Full Course, No Certificate" option is only visible after signing in, so check it logged in before claiming a free audit); Udemy is paid.
- Read the title, instructor, lecture count and topics from the saved HTML (`og:title`, `og:description`, the JSON-LD `instructor` and `ratingValue`); leave ratings and student counts out of the book, they change.
- YouTube videos and playlists: check `https://www.youtube.com/oembed?url=<video url>&format=json` for the title and channel, and the playlist page for `numVideosText`.
- Members-only Real Python lessons show "Unlock All Content"; a real 404 stays 404 with these headers (Coursera `discrete-optimization`), so then say so in the entry.
- This is for reading public pages only. Do not log in, and do not work around a page that asks for sign-in.

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
- `.markdown-link-check.json` — link checker settings; the only ignore rule is the plugin-injected `../print_page/` link, with no per-host exceptions; every request carries the full Chrome header set (global `httpHeaders` entry for `http://` and `https://`, the same headers as in "Checking course pages"); alive status codes are 200, 202 (bot challenges such as eur-lex's) and 206. Udemy, techrxiv and realpython.com still answer 403 to the checker even with these headers (their bot protection appears to look at the TLS fingerprint of Node, which a config cannot change, while `curl` with the same headers gets 200), so judge those three hosts by opening them
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
