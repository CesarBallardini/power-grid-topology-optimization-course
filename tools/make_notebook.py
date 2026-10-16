"""Build a chapter notebook from a plain-text script with ``# %% [markdown]`` and ``# %%`` cell markers.

Usage::

    python tools/make_notebook.py docs/part-X/chNN-slug/src/lab.pct.py docs/part-X/chNN-slug/src/lab.ipynb

The source is an ordinary text file in the percent format: a line ``# %% [markdown]`` starts a Markdown cell
(each following line is Markdown, with a leading ``# `` stripped), a line ``# %%`` starts a code cell. The
notebook is written without outputs, with the ``python3`` kernelspec, ready for ``tests/test_notebooks.py``.
"""

import sys
from pathlib import Path

import nbformat


def parse_percent(text: str) -> list[tuple[str, str]]:
    """Split percent-format text into ``(kind, source)`` cells, ``kind`` being ``markdown`` or ``code``."""
    cells: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('# %%'):
            kind = 'markdown' if '[markdown]' in stripped else 'code'
            cells.append((kind, []))
        elif cells:
            cells[-1][1].append(line)
    out = []
    for kind, lines in cells:
        if kind == 'markdown':
            lines = [ln[2:] if ln.startswith('# ') else ('' if ln.strip() == '#' else ln) for ln in lines]
        source = '\n'.join(lines).strip('\n')
        if source:
            out.append((kind, source))
    return out


def build(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """A v4 notebook with the python3 kernelspec and no outputs."""
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    for i, (kind, source) in enumerate(cells):
        cell = nbformat.v4.new_markdown_cell(source) if kind == 'markdown' else nbformat.v4.new_code_cell(source)
        cell['id'] = str(i)  # the ids nbstripout writes, so the pre-commit hook leaves the file unchanged
        nb.cells.append(cell)
    nbformat.validate(nb)
    return nb


def main(argv: list[str]) -> int:
    """Convert ``argv[1]`` (percent script) into ``argv[2]`` (notebook)."""
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    nbformat.write(build(parse_percent(src.read_text(encoding='utf-8'))), dst)
    print(f'wrote {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
