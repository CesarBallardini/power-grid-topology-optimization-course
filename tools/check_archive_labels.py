"""Check every archive.org access label in the course book against archive.org itself.

The book marks each archive.org copy of a textbook with the access a reader actually gets, written as a bold
label right before the link::

    3rd ed. 1996 **BORROW** [`bwb_Y0-AAK-944`](https://archive.org/details/bwb_Y0-AAK-944)
    **USER-UPLOAD (ES)** [`libroboylestad12ed`](https://archive.org/details/libroboylestad12ed)

Access changes over time (items go dark, move into or out of the lending library), so this script re-reads the
item metadata from ``https://archive.org/metadata/<identifier>`` and reports every label that no longer matches.
The label nearest before a link is the one that describes it; an earlier label in the same sentence belongs to
another edition. Links without any label are reported too, as warnings.

Usage::

    python tools/check_archive_labels.py docs

Exits 1 when a label does not match (or an item could not be checked), 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

METADATA_URL = 'https://archive.org/metadata/'
USER_AGENT = 'curso-optimizacion-redes-electricas/check_archive_labels (+https://archive.org/developers/)'
ATTEMPTS = 3

# Actual access statuses derived from the item metadata.
MISSING = 'MISSING'
DARK = 'DARK'
BORROW = 'BORROW'
PRINT_DISABLED = 'PRINT-DISABLED'
RESTRICTED = 'RESTRICTED'
FREE = 'FREE'
ERROR = 'ERROR'

# Label written in the book -> actual statuses it correctly describes. A Spanish user upload is an ordinary,
# unrestricted item as far as archive.org is concerned, so it shares FREE with the FREE label.
ACCEPTED: dict[str, frozenset[str]] = {
    'BORROW': frozenset({BORROW}),
    'PRINT-DISABLED': frozenset({PRINT_DISABLED}),
    'FREE': frozenset({FREE}),
    'USER-UPLOAD (ES)': frozenset({FREE}),
}

LABEL_RE = re.compile(r'\*\*(' + '|'.join(re.escape(label) for label in ACCEPTED) + r')\*\*')
LINK_RE = re.compile(
    r'\[(?P<text>[^\]\n]*)\]\(https?://(?:www\.)?archive\.org/details/(?P<id>[^)\s/?#]+)[^)\s]*\)'
    r'|<https?://(?:www\.)?archive\.org/details/(?P<bare>[^>\s/?#]+)[^>\s]*>'
)
# Where the text describing a link can start: after a previous link, a table cell border, a list item marker or
# a blank line.
BOUNDARY_RE = re.compile(r'\[|<https?://|\||\n[ \t]*(?:[-*+]|\d+[.)])[ \t]|\n[ \t]*\n')
# How far before a link to look for its label.
WINDOW = 300


@dataclass(frozen=True)
class Citation:
    """One archive.org link found in a Markdown file.

    Attributes:
        path: File the link is in.
        line: 1-based line number of the link.
        identifier: archive.org identifier taken from the URL.
        label: Access label that applies to the link, or None when it has none.
        text_mismatch: Backticked identifier shown as link text when it differs from the URL, else None.
    """

    path: Path
    line: int
    identifier: str
    label: str | None
    text_mismatch: str | None = None


def label_before(text: str, pos: int) -> str | None:
    """Return the access label nearest before ``pos`` within the same sentence-level scope.

    Args:
        text: Whole Markdown document.
        pos: Offset where the link starts.

    Returns:
        The label text (for example ``'BORROW'``), or None when no label precedes the link in its scope.
    """
    window = text[max(0, pos - WINDOW) : pos]
    boundaries = list(BOUNDARY_RE.finditer(window))
    if boundaries:
        window = window[boundaries[-1].end() :]
    labels = LABEL_RE.findall(window)
    return labels[-1] if labels else None


def extract_citations(path: Path, text: str) -> list[Citation]:
    """Find every archive.org details link in a Markdown document and the label that applies to it.

    Args:
        path: File the text came from, recorded in each citation.
        text: Markdown content.

    Returns:
        Citations in document order.
    """
    citations = []
    for match in LINK_RE.finditer(text):
        identifier = urllib.parse.unquote(match['id'] or match['bare'])
        shown = match['text']
        text_mismatch = None
        if shown and len(shown) > 2 and shown[0] == shown[-1] == '`' and shown[1:-1] != identifier:
            text_mismatch = shown[1:-1]
        citations.append(
            Citation(
                path=path,
                line=text.count('\n', 0, match.start()) + 1,
                identifier=identifier,
                label=label_before(text, match.start()),
                text_mismatch=text_mismatch,
            )
        )
    return citations


def classify(meta: dict) -> str:
    """Map an archive.org metadata response to an access status.

    Args:
        meta: Parsed JSON from ``https://archive.org/metadata/<identifier>``.

    Returns:
        One of MISSING, DARK, BORROW, PRINT-DISABLED, RESTRICTED or FREE.
    """
    if not meta:
        return MISSING
    if meta.get('is_dark'):
        return DARK
    metadata = meta.get('metadata', {})
    collections = metadata.get('collection', [])
    collections = [collections] if isinstance(collections, str) else collections
    if 'inlibrary' in collections:
        return BORROW
    if 'printdisabled' in collections:
        return PRINT_DISABLED
    if str(metadata.get('access-restricted-item', '')).lower() == 'true':
        return RESTRICTED
    return FREE


def fetch_status(identifier: str, timeout: float = 30.0, attempts: int = ATTEMPTS) -> str:
    """Query archive.org for one item and classify its access.

    Transient failures (timeouts, dropped connections, HTTP 429 and 5xx) are retried with exponential backoff:
    under a burst of parallel requests archive.org occasionally drops one that succeeds a second later.

    Args:
        identifier: archive.org identifier.
        timeout: Socket timeout in seconds.
        attempts: Total tries before giving up.

    Returns:
        The status from :func:`classify`, MISSING on HTTP 404, or ERROR when the item could not be checked.
    """
    url = METADATA_URL + urllib.parse.quote(identifier, safe='')
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})  # noqa: S310 -- fixed https URL
    for attempt in range(attempts):
        if attempt:
            time.sleep(2**attempt)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 -- fixed https URL
                return classify(json.load(response))
        except urllib.error.HTTPError as exc:
            # The error carries the open response body; close it rather than leave it to the garbage collector.
            exc.close()
            if exc.code == 404:
                return MISSING
            if exc.code != 429 and exc.code < 500:
                return ERROR
        except OSError, ValueError:
            # OSError covers URLError, timeouts and resets mid-read; ValueError a truncated JSON body.
            pass
    return ERROR


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


def main(argv: list[str] | None = None) -> int:
    """Scan the book, check every cited item in parallel, print a report.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 1 when any label mismatches or any item could not be checked, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('roots', nargs='+', type=Path, help='Markdown files or directories to scan')
    parser.add_argument('--workers', type=int, default=8, help='parallel archive.org requests (default: 8)')
    parser.add_argument('--timeout', type=float, default=30.0, help='per-request timeout in seconds (default: 30)')
    args = parser.parse_args(argv)

    files = markdown_files(args.roots)
    citations = [c for path in files for c in extract_citations(path, path.read_text(encoding='utf-8'))]
    unique = sorted({c.identifier for c in citations})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        actual = dict(zip(unique, pool.map(lambda i: fetch_status(i, args.timeout), unique), strict=True))

    failures = 0
    unlabeled = 0
    for c in citations:
        where = f'{c.path.as_posix()}:{c.line}:'
        status = actual[c.identifier]
        if c.text_mismatch:
            failures += 1
            print(f'{where} LINK-MISMATCH text `{c.text_mismatch}` points to {c.identifier}')
        if status == ERROR:
            failures += 1
            print(f'{where} ERROR could not check {c.identifier}')
        elif c.label is None:
            unlabeled += 1
            print(f'{where} UNLABELED {c.identifier} (actual {status})')
        elif status not in ACCEPTED[c.label]:
            failures += 1
            print(f'{where} MISMATCH {c.identifier} stated {c.label}, actual {status}')

    print(
        f'checked {len(citations)} citations ({len(unique)} unique identifiers) in {len(files)} files: '
        f'{failures} problems, {unlabeled} unlabeled'
    )
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
