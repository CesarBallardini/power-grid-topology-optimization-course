"""Tests for tools/notebook_hooks.py: the import scan, the staging copy, link rewriting, math and buttons."""

import json
import posixpath
import re
import tomllib
from pathlib import Path

import pytest

import notebook_hooks as nh

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def write_notebook(path: Path, cells: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        'cells': [
            {'cell_type': kind, 'metadata': {}, 'source': source}
            | ({'outputs': [], 'execution_count': None} if kind == 'code' else {})
            for kind, source in cells
        ],
        'metadata': {},
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    path.write_text(json.dumps(nb), encoding='utf-8')
    return path


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """A two-chapter book: ch02 imports a module of ch01, which imports scipy."""
    root = tmp_path / 'docs'
    ch01 = root / 'part-1' / 'ch01-a' / 'src'
    ch02 = root / 'part-1' / 'ch02-b' / 'src'
    ch01.mkdir(parents=True)
    ch02.mkdir(parents=True)
    (ch01 / 'shared.py').write_text('import scipy.linalg\nimport json\n', encoding='utf-8')
    (ch01 / 'test_shared.py').write_text('import pytest\n', encoding='utf-8')
    (ch01 / 'data.csv').write_text('a,b\n1,2\n', encoding='utf-8')
    (ch01 / 'runs' / 'run_0').mkdir(parents=True)
    (ch01 / 'runs' / 'run_0' / 'res.json').write_text('{}', encoding='utf-8')
    (ch01 / '.hypothesis').mkdir()
    (ch01 / '.hypothesis' / 'db').write_text('', encoding='utf-8')
    (ch02 / 'local.py').write_text(
        'import sys\nfrom pathlib import Path\n'
        "sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ch01-a' / 'src'))\n"
        'from shared import *\n',
        encoding='utf-8',
    )
    (ch02 / 'helper_toop.py').write_text('import toop_engine_dc_solver\n', encoding='utf-8')
    write_notebook(
        ch01 / 'lab.ipynb',
        [('markdown', '# Lab\n\nSee [the chapter](../index.md#lab) and $x^2$.'), ('code', 'import numpy as np')],
    )
    write_notebook(ch02 / 'lab.ipynb', [('markdown', '# Lab'), ('code', 'from local import *\n%matplotlib inline')])
    write_notebook(ch02 / 'lab_toop.ipynb', [('code', 'import numpy')])
    write_notebook(ch02 / 'exercises.ipynb', [('code', 'import pulp')])
    return root


def test_imported_modules():
    source = 'import numpy as np\nimport os.path\nfrom scipy.sparse import csr_matrix\nfrom . import x\n'
    assert nh.imported_modules(source) == {'numpy', 'os', 'scipy'}


def test_notebook_code_skips_magics(docs):
    code = nh.notebook_code(docs / 'part-1' / 'ch02-b' / 'src' / 'lab.ipynb')
    assert code == 'from local import *'


def test_scan_follows_own_and_other_chapters_modules(docs):
    third_party, modules = nh.scan(docs / 'part-1' / 'ch02-b' / 'src' / 'lab.ipynb', docs)
    assert third_party == {'scipy'}
    assert {m.stem for m in modules} == {'local', 'shared'}


def test_why_not_runnable(docs, monkeypatch):
    src = docs / 'part-1' / 'ch02-b' / 'src'
    assert nh.why_not_runnable(src / 'lab.ipynb', docs) is None
    assert nh.why_not_runnable(src / 'lab_toop.ipynb', docs) == 'needs ToOp'
    assert nh.why_not_runnable(src / 'exercises.ipynb', docs) == 'imports pulp, not available in Pyodide'
    monkeypatch.setitem(nh.NOT_RUNNABLE, 'ch02-b/lab.ipynb', 'spawns a subprocess')
    assert nh.why_not_runnable(src / 'lab.ipynb', docs) == 'spawns a subprocess'
    assert not nh.runnable(src / 'lab.ipynb', docs)


def test_book_notebooks_classification():
    """Every chapter notebook is classified; the ones that need ToOp, labs or JAX are never runnable."""
    notebooks = sorted(DOCS.glob(nh.NOTEBOOK_GLOB))
    assert len(notebooks) > 100
    runnable = {nh.key(nb) for nb in notebooks if nh.runnable(nb)}
    assert 'ch01-complex-numbers/lab.ipynb' in runnable
    assert 'ch23-optimization-modeling/lab.ipynb' not in runnable  # pulp
    assert 'ch31-jax-gpu-fundamentals/exercises.ipynb' not in runnable  # jax
    assert not {k for k in runnable if k.endswith(('_toop.ipynb', '_labs.ipynb'))}
    for name in nh.NOT_RUNNABLE:
        assert any(DOCS.glob(f'part-*/{name.replace("/", "/src/")}')), f'NOT_RUNNABLE names a missing notebook: {name}'


def test_book_notebook_links_resolve():
    """Every relative link in a notebook's Markdown names a file of the book (the build fails otherwise)."""
    for nb in sorted(DOCS.glob(nh.NOTEBOOK_GLOB)):
        rel = nb.relative_to(DOCS).as_posix()
        for cell in json.loads(nb.read_text(encoding='utf-8'))['cells']:
            if cell['cell_type'] != 'markdown':
                continue
            for target in nh.LINK.findall(''.join(cell['source'])):
                path = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target.partition('#')[0]))
                assert (DOCS / path).is_file(), f'{rel} links to {target}'


def test_book_notebook_math_is_balanced():
    """Every ``$`` in a notebook's Markdown (outside code) opens or closes math, so the page typesets it.

    The rendered HTML no longer tells ``\\$`` from ``$``, so a literal dollar is written ``&#36;``.
    """
    for nb in sorted(DOCS.glob(nh.NOTEBOOK_GLOB)):
        for cell in json.loads(nb.read_text(encoding='utf-8'))['cells']:
            if cell['cell_type'] == 'markdown':
                text = re.sub(r'```.*?```|`[^`]*`', '', ''.join(cell['source']), flags=re.S)
                count = text.replace('$$', '$').count('$')
                assert count % 2 == 0, f'{nb}: unbalanced $ (write a literal dollar as &#36;)'


def test_absolute_links():
    rel = 'part-1/ch06-x/src/lab.ipynb'
    book = 'https://example.org/book/v1/'
    markdown = (
        '[a](../index.md) [b](../solutions.md#lab) [c](../../../part-3/ch15-y/index.md) [d](lab_toop.ipynb) '
        '[e](https://other.org/x.md) [f](#here)'
    )
    assert nh.absolute_links(markdown, rel, book) == (
        '[a](https://example.org/book/v1/part-1/ch06-x/) '
        '[b](https://example.org/book/v1/part-1/ch06-x/solutions/#lab) '
        '[c](https://example.org/book/v1/part-3/ch15-y/) '
        '[d](https://example.org/book/v1/part-1/ch06-x/src/lab_toop/) '
        '[e](https://other.org/x.md) [f](#here)'
    )


def test_stage(docs, tmp_path):
    out = tmp_path / 'contents'
    rel = nh.stage(out, 'https://example.org/', docs)
    assert rel == ['part-1/ch01-a/src/lab.ipynb', 'part-1/ch02-b/src/lab.ipynb']
    assert (out / 'runnable.txt').read_text(encoding='utf-8').split() == rel
    # ch01 is copied whole (data files and folders too); tests, hidden folders and ToOp files are left out, as are
    # unrunnable notebooks
    ch01_files = (out / 'part-1' / 'ch01-a' / 'src').rglob('*')
    assert sorted(p.relative_to(out).as_posix() for p in ch01_files if p.is_file()) == [
        'part-1/ch01-a/src/data.csv',
        'part-1/ch01-a/src/lab.ipynb',
        'part-1/ch01-a/src/runs/run_0/res.json',
        'part-1/ch01-a/src/shared.py',
    ]
    assert sorted(p.name for p in (out / 'part-1' / 'ch02-b' / 'src').iterdir()) == ['lab.ipynb', 'local.py']
    ch01 = json.loads((out / rel[0]).read_text(encoding='utf-8'))
    assert '[the chapter](https://example.org/part-1/ch01-a/#lab)' in ch01['cells'][0]['source']
    assert ch01['cells'][1]['source'].endswith('%pip install -q numpy')  # after the title cell
    ch02 = json.loads((out / rel[1]).read_text(encoding='utf-8'))
    assert ch02['cells'][1]['source'].endswith('%pip install -q scipy')


def test_install_cell_absent_without_third_party(docs, tmp_path):
    nb = write_notebook(docs / 'part-1' / 'ch01-a' / 'src' / 'examples.ipynb', [('code', 'import cmath')])
    target = tmp_path / 'examples.ipynb'
    nh.stage_notebook(nb, target, 'https://example.org/', docs)
    assert [c['source'] for c in json.loads(target.read_text(encoding='utf-8'))['cells']] == ['import cmath']


def test_install_cell_adds_runtime_dependencies():
    assert nh.install_cell({'networkx'})['source'].splitlines()[1:] == ['%pip install -q networkx scipy']
    lines = nh.install_cell({'pandapower', 'matplotlib'})['source'].splitlines()[1:]
    assert lines[-1] == f'%pip install -q --no-deps {nh.NO_DEPS["pandapower"]}'
    first = lines[0].split()
    assert {'matplotlib', 'pandas', 'tzdata', 'scipy', 'deepdiff==8.6.2'} <= set(first)
    assert 'pandapower' not in first


def test_no_deps_pins_match_the_lockfile():
    """The browser installs the same pandapower as the course environment."""
    lock = tomllib.loads((ROOT / 'uv.lock').read_text(encoding='utf-8'))
    locked = {p['name']: p['version'] for p in lock['package']}
    for requirement in nh.NO_DEPS.values():
        name, _, version = requirement.partition('==')
        assert locked[name] == version, f'{requirement} in tools/notebook_hooks.py, {locked[name]} in uv.lock'


def test_page_links():
    html = '<a href="../index.md#lab">x</a> <a class="anchor-link" href="#step">y</a> <a href="https://a.b/c.md">z</a>'
    seen = []

    def resolve(path, fragment):
        seen.append((path, fragment))
        return '../../' + ('#' + fragment if fragment else '')

    out = nh.page_links(html, 'part-1/ch06-x/src/lab.ipynb', resolve)
    assert seen == [('part-1/ch06-x/index.md', 'lab')]
    assert (
        out == '<a href="../../#lab">x</a> <a class="anchor-link" href="#step">y</a> <a href="https://a.b/c.md">z</a>'
    )


def test_typeset_math():
    html = '<p>Solve $J\\,\\Delta = -F$ and</p><p>$$x^2 + y^2$$</p><pre>cost = "$5"; $x$</pre><code>$y$</code> \\$3'
    assert nh.typeset_math(html) == (
        '<p>Solve <span class="arithmatex">\\(J\\,\\Delta = -F\\)</span> and</p>'
        '<p><span class="arithmatex">\\[x^2 + y^2\\]</span></p><pre>cost = "$5"; $x$</pre><code>$y$</code> \\$3'
    )


def test_buttons_and_title():
    plain = nh.buttons('lab.ipynb', None)
    assert 'href="lab.ipynb" download' in plain
    assert 'Run in your browser' not in plain
    run = nh.buttons('lab.ipynb', '../../../../lite/notebooks/index.html?path=p/c/src/lab.ipynb')
    assert 'href="../../../../lite/notebooks/index.html?path=p/c/src/lab.ipynb"' in run
    assert nh.insert_after_title('<h1>T</h1><p>x</p>', '<b/>') == '<h1>T</h1><b/><p>x</p>'
    assert nh.insert_after_title('<p>x</p>', '<b/>') == '<b/><p>x</p>'


def test_externalize_css():
    css = '.jupyter-wrapper{color:red}' * 100
    html = f'<style type="text/css">{css}</style><style type="text/css">.small{{}}</style><script src=""> </script>'
    store = {}
    out = nh.externalize_css(html, 'part-1/ch06-x/src/lab/', store)
    [(name, saved)] = store.items()
    assert saved == css
    assert out == f'<link rel="stylesheet" href="../../../../{name}"><style type="text/css">.small{{}}</style>'


def test_test_lite_uses_the_pyodide_of_the_jupyterlite_kernel():
    # make test-lite and test-lite.yml run the notebooks in the Pyodide pinned in lite/package-lock.json; the site
    # loads the one the JupyterLite Pyodide kernel names. A bump of either alone would test a different Python.
    constants = pytest.importorskip('jupyterlite_pyodide_kernel.constants')
    lock = json.loads((ROOT / 'lite' / 'package-lock.json').read_text(encoding='utf-8'))
    assert lock['packages']['node_modules/pyodide']['version'] == constants.PYODIDE_VERSION
