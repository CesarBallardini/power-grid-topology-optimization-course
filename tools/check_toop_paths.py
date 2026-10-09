"""Check that every ToOp source path cited in the course book exists in a ToOp checkout.

Chapters cite ToOp code as backticked paths. Package paths use a short course prefix that stands for the package
source directory, and repository paths are relative to the ToOp root::

    `dc_solver/jax/compute_batch.py`      -> packages/dc_solver_pkg/src/toop_engine_dc_solver/jax/compute_batch.py
    `optimizer/ac/select_strategy.py:42`  -> packages/topology_optimizer_pkg/src/.../ac/select_strategy.py
    `notebooks/tests/test_notebooks.py`   -> notebooks/tests/test_notebooks.py

Line numbers and symbol suffixes (``:123``, ``:123-456``, ``:func_name``) are dropped before resolving. Tokens
containing spaces or glob/template characters (``*``, ``{``, ``<``) are not paths and are skipped.

Usage::

    python tools/check_toop_paths.py docs [--toop ../ToOp]

The ToOp checkout is ``--toop``, else the ``TOOP_REPO`` environment variable, else ``../ToOp`` relative to this
repository. Exits 1 when any cited path is missing, 2 when the checkout itself is not found.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Course prefix -> path relative to the ToOp repository root. Order does not matter: prefixes are disjoint.
PREFIXES: dict[str, str] = {
    'dc_solver/': 'packages/dc_solver_pkg/src/toop_engine_dc_solver/',
    'optimizer/': 'packages/topology_optimizer_pkg/src/toop_engine_topology_optimizer/',
    'interfaces/': 'packages/interfaces_pkg/src/toop_engine_interfaces/',
    'grid_helpers/': 'packages/grid_helpers_pkg/src/toop_engine_grid_helpers/',
    'contingency/': 'packages/contingency_analysis_pkg/src/toop_engine_contingency_analysis/',
    'importer/': 'packages/importer_pkg/src/toop_engine_importer/',
    'docs/': 'docs/',
    'notebooks/': 'notebooks/',
    'packages/': 'packages/',
}

CODE_SPAN_RE = re.compile(r'`([^`\n]+)`')
# Characters that mark a token as a pattern, placeholder or prose rather than a concrete path.
NOT_A_PATH_RE = re.compile(r'[\s*{<]')
TRAILING_PUNCTUATION = '.,;:!?)]\'"'


@dataclass(frozen=True)
class Reference:
    """One cited ToOp path.

    Attributes:
        path: Markdown file the citation is in.
        line: 1-based line number of the citation.
        token: Path as written in the book, after normalization.
        target: Path relative to the ToOp root that the token resolves to.
    """

    path: Path
    line: int
    token: str
    target: str


def normalize(token: str) -> str | None:
    """Turn a backticked token into a bare course path, or None when it is not a ToOp path.

    Args:
        token: Content of an inline code span.

    Returns:
        The token without line/symbol suffixes and trailing punctuation, or None when it has no known prefix or
        looks like a pattern rather than a path.
    """
    if NOT_A_PATH_RE.search(token):
        return None
    token = token.split(':', 1)[0].split('#', 1)[0].rstrip(TRAILING_PUNCTUATION)
    if not token.startswith(tuple(PREFIXES)):
        return None
    return token


def expand(token: str) -> str:
    """Replace the course prefix of a normalized token with its path in the ToOp repository.

    Args:
        token: Normalized token, starting with one of :data:`PREFIXES`.

    Returns:
        Path relative to the ToOp root.
    """
    prefix = next(p for p in PREFIXES if token.startswith(p))
    return PREFIXES[prefix] + token[len(prefix) :]


def extract_references(path: Path, text: str) -> list[Reference]:
    """Find every cited ToOp path in a Markdown document.

    Args:
        path: File the text came from, recorded in each reference.
        text: Markdown content.

    Returns:
        References in document order.
    """
    references = []
    for match in CODE_SPAN_RE.finditer(text):
        token = normalize(match[1])
        if token is None:
            continue
        line = text.count('\n', 0, match.start()) + 1
        references.append(Reference(path=path, line=line, token=token, target=expand(token)))
    return references


def markdown_files(roots: list[Path]) -> list[Path]:
    """Expand directories into their Markdown files, recursively.

    Args:
        roots: Files and directories given on the command line.

    Returns:
        Sorted Markdown file paths.
    """
    files: set[Path] = set()
    for root in roots:
        files.update(root.rglob('*.md') if root.is_dir() else [root])
    return sorted(files)


def default_toop_root() -> Path:
    """Return the ToOp checkout from ``TOOP_REPO``, falling back to ``../ToOp`` next to this repository.

    Returns:
        Path to the ToOp repository root (not checked for existence).
    """
    return Path(os.environ.get('TOOP_REPO') or REPO_ROOT.parent / 'ToOp')


def main(argv: list[str] | None = None) -> int:
    """Scan the book and report cited ToOp paths that do not exist.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 2 when the ToOp checkout is missing, 1 when any cited path is missing, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('roots', nargs='+', type=Path, help='Markdown files or directories to scan')
    parser.add_argument('--toop', type=Path, default=None, help='ToOp checkout (default: $TOOP_REPO or ../ToOp)')
    args = parser.parse_args(argv)

    toop = args.toop if args.toop is not None else default_toop_root()
    if not toop.is_dir():
        print(f'ToOp checkout not found at {toop}; pass --toop or set TOOP_REPO', file=sys.stderr)
        return 2

    files = markdown_files(args.roots)
    references = [r for path in files for r in extract_references(path, path.read_text(encoding='utf-8'))]
    missing = [r for r in references if not (toop / r.target).exists()]
    for r in missing:
        print(f'{r.path.as_posix()}:{r.line}: missing `{r.token}` -> {r.target}')

    unique = len({r.target for r in references})
    print(
        f'checked {len(references)} references ({unique} unique paths) in {len(files)} files '
        f'against {toop}: {len(missing)} missing'
    )
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
