"""Generate the acronym tooltips file from the glossary (Appendix D).

The glossary's acronym table is the single source of truth. Every row becomes a Markdown abbreviation line
(``*[PTDF]: Power transfer distribution factor``) in ``includes/abbreviations.md``, which ``pymdownx.snippets``
appends to every page so that acronyms show their expansion on hover.

Rows with several acronyms (``KCL / KVL``) are split when the expansion splits into the same number of parts;
otherwise, or when the split parts read badly, an explicit entry in ``OVERRIDES`` wins. Acronyms that are too
common or too ambiguous for tooltips on every occurrence are listed in ``SKIP``.

Usage::

    python tools/gen_abbreviations.py            # rewrite includes/abbreviations.md
    python tools/gen_abbreviations.py --check    # exit 1 if the file is out of date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = REPO_ROOT / 'docs' / 'appendices' / 'd-glossary.md'
OUTPUT = REPO_ROOT / 'includes' / 'abbreviations.md'

# Too frequent (AC, DC), punctuation that abbr matching handles poorly (N-1, p.u.), or proper names that read
# better without a tooltip.
SKIP = {'AC', 'DC', 'N-1', 'p.u.', 'SE', 'MAP-Elites', 'NP-hard', 'NSGA-II', 'UCTE-DEF', 'SO GL', 'ENTSO-E'}

# Rows whose shared expansion does not split cleanly on "/".
OVERRIDES = {
    'KCL': "Kirchhoff's current law",
    'KVL': "Kirchhoff's voltage law",
    'PATL': 'Permanent admissible transmission loading',
    'TATL': 'Temporary admissible transmission loading',
    'PQ': 'Load bus (active and reactive power specified)',
    'PV': 'Generator bus (active power and voltage specified)',
    'SDP': 'Semidefinite programming',
    'SOCP': 'Second-order cone programming',
    'LCC': 'Line-commutated converter',
    'VSC': 'Voltage-source converter',
    'IIDM': 'iTesla Internal Data Model',
    'XIIDM': 'Extended iTesla Internal Data Model (XML)',
}

_ACRONYM = re.compile(r'[A-Za-z]+')


def parse_acronym_rows(markdown: str) -> list[tuple[str, str]]:
    """Return (acronym cell, expansion cell) pairs from the glossary's acronym table.

    Args:
        markdown: Full text of the glossary page.

    Returns:
        One pair per table row of the "Acronyms" section, header and separator rows excluded.
    """
    if '## Acronyms' not in markdown:
        return []
    section = markdown.split('## Acronyms', 1)[1].split('\n## ', 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 2 or cells[0] in ('Acronym', '') or set(cells[0]) <= {'-'}:
            continue
        rows.append((cells[0], cells[1]))
    return rows


def build_entries(rows: list[tuple[str, str]]) -> dict[str, str]:
    """Map each acronym to its tooltip text.

    Args:
        rows: (acronym cell, expansion cell) pairs as returned by `parse_acronym_rows`.

    Returns:
        Acronym to expansion, with `SKIP` removed and `OVERRIDES` applied.
    """
    entries: dict[str, str] = {}
    for acronym_cell, expansion_cell in rows:
        acronyms = [a.strip() for a in acronym_cell.split('/')]
        expansions = [e.strip() for e in expansion_cell.split('/')]
        if len(expansions) != len(acronyms):
            expansions = [expansion_cell] * len(acronyms)
        for acronym, expansion in zip(acronyms, expansions, strict=True):
            if acronym in SKIP or not _ACRONYM.fullmatch(acronym):
                continue
            entries[acronym] = expansion
    for acronym, expansion in OVERRIDES.items():
        if acronym not in SKIP:
            entries[acronym] = expansion
    return entries


def render(entries: dict[str, str]) -> str:
    """Render entries as Markdown abbreviation definitions, sorted by acronym.

    Args:
        entries: Acronym to expansion.

    Returns:
        File content ending with a newline.
    """
    return ''.join(f'*[{acronym}]: {entries[acronym]}\n' for acronym in sorted(entries))


def main(argv: list[str] | None = None) -> int:
    """Generate or check the abbreviations file.

    Args:
        argv: Command-line arguments (defaults to `sys.argv[1:]`).

    Returns:
        Process exit code: 0 on success, 1 when `--check` finds the file out of date.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='fail if the abbreviations file is out of date')
    args = parser.parse_args(argv)

    content = render(build_entries(parse_acronym_rows(GLOSSARY.read_text(encoding='utf-8'))))
    current = OUTPUT.read_text(encoding='utf-8') if OUTPUT.exists() else ''
    if args.check:
        if current != content:
            print(f'{OUTPUT.relative_to(REPO_ROOT)} is out of date: run make abbreviations')
            return 1
        print(f'{OUTPUT.relative_to(REPO_ROOT)} is up to date ({content.count(chr(10))} abbreviations)')
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding='utf-8')
    print(f'wrote {content.count(chr(10))} abbreviations to {OUTPUT.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
