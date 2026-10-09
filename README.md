# Power Grid Topology Optimization — course book

[![check](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/check.yml/badge.svg)](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/check.yml)
[![links](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/links.yml/badge.svg)](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/links.yml)
[![build](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/build.yml/badge.svg)](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/build.yml)
[![docs](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/docs.yaml/badge.svg)](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/docs.yaml)
[![validate-pr-title](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/validate_pr_title.yaml/badge.svg)](https://github.com/CesarBallardini/power-grid-topology-optimization-course/actions/workflows/validate_pr_title.yaml)
[![site](https://img.shields.io/website?url=https://katra.ballardini.com.ar/power-grid-topology-optimization-course/&label=site)](https://katra.ballardini.com.ar/power-grid-topology-optimization-course/)

A five-semester course that takes students from sophomore-level college algebra, calculus and physics to
comprehending, using, debugging and improving [ToOp](https://github.com/eliagroup/ToOp), the open-source GPU
topology optimization engine by Elia Group and 50Hertz, and topology optimization software in general.

The course is written as a book built with [MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/): 36 chapters in six parts (mathematics,
electricity and circuits, power systems, optimization, computing, capstone), plus appendices with the
university syllabi consulted, the bibliography (archive.org access status and Spanish editions) and a concept
map of the ToOp code. All material is in English.

**Read it online:** <https://katra.ballardini.com.ar/power-grid-topology-optimization-course/>
(published from release tags; a printable single-page version is included).

## Working on the book

Prerequisites: [uv](https://docs.astral.sh/uv/) (Python 3.14 is installed by uv from `.python-version`), Git,
and `make`. Optional: [gitleaks](https://github.com/gitleaks/gitleaks) on `PATH` for a full secret scan.

```bash
make install      # sync the environment from uv.lock and install the git hooks
make docs-serve   # live preview at http://127.0.0.1:8000
make docs         # strict build: any warning (broken link, missing page) fails
make test         # tests of the maintenance tools
make check-toop   # every ToOp source path cited in the book exists (needs ../ToOp)
make check-books  # every archive.org access label still matches archive.org (network)
make              # list all targets
```

## Layout

```
docs/
  index.md                 introduction and chapter dependency flowchart
  course-guide.md          audience, schedule, lab tools, conventions
  part-0-orientation/ … part-6-capstone/   one page per chapter
  appendices/              syllabi, bibliography, ToOp concept map
tools/                     maintenance scripts (archive.org labels, ToOp paths)
tests/                     tests for the tools
mkdocs.yml                 site configuration and navigation
```

## Releasing an edition

The site is versioned with [mike](https://github.com/jimporter/mike). Pushing a tag `vX.Y.Z` runs the `docs`
workflow, which publishes that version to GitHub Pages and points `latest` at it. Commit messages and pull
request titles follow [Conventional Commits](https://www.conventionalcommits.org/).
