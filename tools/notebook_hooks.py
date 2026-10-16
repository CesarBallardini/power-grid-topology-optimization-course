"""MkDocs hooks for the chapter notebooks: book pages for every notebook, JupyterLite for the ones Pyodide can run.

mkdocs-jupyter turns each ``docs/part-*/chNN-slug/src/*.ipynb`` into a page (``.../src/lab/``). These hooks
finish the job, in every ``mkdocs build`` (``make docs``, the build workflow and ``mike deploy`` alike):

- **links**: a notebook's Markdown links (``../index.md#lab``, ``../solutions.md``) are relative to the
  ``.ipynb`` file and stay raw in the HTML that mkdocs-jupyter writes; they are rewritten to the pages' URLs. A
  link to a file that is not in the book fails the build, as ``--strict`` does for Markdown pages;
- **math**: ``$...$`` and ``$$...$$`` in Markdown cells are wrapped the way pymdownx.arithmatex wraps them, so
  the book's MathJax configuration typesets them;
- **buttons**: "Download the notebook" on every notebook page, and "Run in your browser" on the notebooks that
  Pyodide can run, linking to the JupyterLite site built into ``<site>/lite``;
- **size**: the 570 KB of JupyterLab CSS that mkdocs-jupyter inlines into every page is written once, as a file;
- **JupyterLite**: after the build, the runnable chapters' ``src/`` folders are staged (without tests, ToOp or
  labs files, with an install cell and absolute links to the book) and ``jupyter lite build`` writes the
  JupyterLite site. ``LITE=0`` skips this step, for a quick build; ``mkdocs serve`` skips it unless ``LITE=1``.

A notebook is runnable when every module it imports, directly or through the chapter modules it imports (also
those of other chapters, found on ``sys.path`` as the chapter code arranges), is in the standard library or in
``PYODIDE_OK``, and it is neither a ``_toop`` nor a ``_labs`` notebook nor listed in ``NOT_RUNNABLE``.

``python tools/notebook_hooks.py stage <dir>`` writes the staged contents alone, with ``runnable.txt``, for
``make test-lite``, which runs every runnable notebook in Pyodide under Node (``lite/run_in_pyodide.mjs``).
"""

import ast
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
LITE_CONFIG = ROOT / 'lite' / 'jupyter-lite.json'
LITE_DIR = 'lite'  # the JupyterLite site, under the book's site folder
NOTEBOOK_GLOB = 'part-*/*/src/*.ipynb'

# Third-party modules that run in Pyodide (the version the pinned JupyterLite Pyodide kernel loads). Every
# notebook that relies on them runs in `make test-lite`, so an entry here is checked, not assumed.
PYODIDE_OK = frozenset({'matplotlib', 'networkx', 'numpy', 'pandapower', 'pandas', 'pydantic', 'scipy'})

# What else to install with a module, because the import scan cannot see it: networkx imports scipy inside its
# linear-algebra functions, and pandas time zones need the tzdata package (Pyodide has no system time zone data).
# pandapower 3.5.6 pins pandas~=2.3 (Pyodide has 3.x) and deepdiff~=9.1, whose cachebox dependency is compiled
# and has no Pyodide build, so micropip cannot resolve it: its dependencies are listed here, with deepdiff 8, and
# pandapower itself is installed without them (NO_DEPS). make test-lite runs every pandapower notebook this way.
EXTRA: dict[str, frozenset[str]] = {
    'networkx': frozenset({'scipy'}),
    'pandas': frozenset({'tzdata'}),
    'pandapower': frozenset({
        'colorama==0.4.6', 'deepdiff==8.6.2', 'geojson==3.3.0', 'networkx', 'numpy', 'packaging', 'pandas',
        'pandera==0.34.1', 'scipy', 'tqdm', 'typing_extensions',
    }),
}  # fmt: skip
# Installed with --no-deps, under this requirement. Keep the pandapower pin equal to the one in uv.lock.
NO_DEPS: dict[str, str] = {'pandapower': 'pandapower==3.5.6'}

# Notebooks whose imports would pass but which still cannot run in the browser, keyed by
# '<chapter folder>/<notebook>', with the reason.
NOT_RUNNABLE: dict[str, str] = {}

log = logging.getLogger('mkdocs.hooks.notebooks')


# --- which notebooks run in Pyodide ------------------------------------------------------------------------


def imported_modules(source: str) -> set[str]:
    """Top-level names of the absolute imports in ``source`` (``import a.b`` and ``from a.b import c`` give a)."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split('.')[0])
    return names


def notebook_code(path: Path) -> str:
    """The code cells of a notebook, joined, with IPython magics (``%pip``, ``!cmd``) left out."""
    cells = json.loads(path.read_text(encoding='utf-8'))['cells']
    lines = []
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            lines += [line for line in source.splitlines() if not line.lstrip().startswith(('%', '!'))]
    return '\n'.join(lines)


def find_module(name: str, folder: Path, docs: Path = DOCS) -> Path | None:
    """The chapter module ``name``: in the notebook's own ``src/`` first, otherwise in exactly one other chapter."""
    own = folder / f'{name}.py'
    if own.is_file():
        return own
    found = [p for p in docs.glob(f'part-*/*/src/{name}.py') if p.parent != folder]
    return found[0] if len(found) == 1 else None


def scan(notebook: Path, docs: Path = DOCS) -> tuple[set[str], set[Path]]:
    """The third-party modules a notebook needs and the chapter modules it imports, transitively."""
    pending = imported_modules(notebook_code(notebook))
    seen: set[str] = set()
    third_party: set[str] = set()
    modules: set[Path] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        module = find_module(name, notebook.parent, docs)
        if module is not None:
            modules.add(module)
            pending |= imported_modules(module.read_text(encoding='utf-8'))
        elif name not in sys.stdlib_module_names:
            third_party.add(name)
    return third_party, modules


def key(notebook: Path) -> str:
    """``chNN-slug/name.ipynb``, the key of ``NOT_RUNNABLE``."""
    return f'{notebook.parents[1].name}/{notebook.name}'


def why_not_runnable(notebook: Path, docs: Path = DOCS) -> str | None:
    """Why the notebook cannot run in Pyodide, or ``None`` when it can."""
    if notebook.stem.endswith('_toop'):
        return 'needs ToOp'
    if notebook.stem.endswith('_labs'):
        return 'needs the labs container'
    if key(notebook) in NOT_RUNNABLE:
        return NOT_RUNNABLE[key(notebook)]
    missing = scan(notebook, docs)[0] - PYODIDE_OK
    if missing:
        return 'imports ' + ', '.join(sorted(missing)) + ', not available in Pyodide'
    return None


def runnable(notebook: Path, docs: Path = DOCS) -> bool:
    """True when the notebook runs in the browser (JupyterLite, Pyodide kernel)."""
    return why_not_runnable(notebook, docs) is None


# --- staging the JupyterLite contents ----------------------------------------------------------------------


def staged_file(path: Path) -> bool:
    """True for the ``src/`` files and folders JupyterLite gets: not tests, caches, ToOp or labs files."""
    name = path.name
    return not (
        name.startswith(('test_', '.'))
        or name == '__pycache__'
        or '_toop' in path.stem
        or '_labs' in path.stem
        or name.endswith('.pyc')
    )


LINK = re.compile(r'\]\((?!https?:|mailto:|#)([^)\s]+)\)')


def absolute_links(markdown: str, notebook_rel: str, book_url: str) -> str:
    """Markdown links relative to the notebook, as absolute URLs of the book pages they point to.

    ``notebook_rel`` is the notebook's path in ``docs/`` (``part-1-math/ch06-newton-raphson/src/lab.ipynb``).
    JupyterLite opens a relative link as a file of its own contents, so links to the book must be absolute.
    """

    def replace(match: re.Match) -> str:
        target, _, fragment = match.group(1).partition('#')
        path = posixpath.normpath(posixpath.join(posixpath.dirname(notebook_rel), target))
        if path.endswith('.ipynb'):
            path = path[: -len('.ipynb')] + '/'
        elif path.endswith('index.md'):
            path = path[: -len('index.md')]
        elif path.endswith('.md'):
            path = path[: -len('.md')] + '/'
        url = book_url + path + (f'#{fragment}' if fragment else '')
        return f']({url})'

    return LINK.sub(replace, markdown)


def install_cell(third_party: set[str]) -> dict:
    """The code cell that installs the notebook's libraries in the browser."""
    requirements: set[str] = set()
    pending = set(third_party)
    while pending:
        requirement = pending.pop()
        requirements.add(requirement)
        pending |= EXTRA.get(re.split(r'[=<>]', requirement)[0], frozenset()) - requirements
    plain = sorted(requirements - NO_DEPS.keys())
    lines = [f'%pip install -q {" ".join(plain)}'] if plain else []
    lines += [f'%pip install -q --no-deps {NO_DEPS[name]}' for name in sorted(requirements & NO_DEPS.keys())]
    header = '# JupyterLite only: install the libraries this notebook and its chapter modules use.'
    source = '\n'.join([header, *lines])
    return {'cell_type': 'code', 'execution_count': None, 'id': 'jupyterlite-install', 'metadata': {},
            'outputs': [], 'source': source}  # fmt: skip


def stage_notebook(notebook: Path, target: Path, book_url: str, docs: Path = DOCS) -> None:
    """Write the JupyterLite copy of ``notebook`` to ``target``: absolute links, the install cell first."""
    rel = notebook.relative_to(docs).as_posix()
    nb = json.loads(notebook.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            cell['source'] = absolute_links(source, rel, book_url)
    third_party = scan(notebook, docs)[0]
    if third_party:
        # after the title cell, when the notebook starts with one
        first = nb['cells'][0] if nb['cells'] else {}
        title = first.get('cell_type') == 'markdown' and ''.join(first['source']).startswith('# ')
        nb['cells'].insert(1 if title else 0, install_cell(third_party))
    target.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def stage(out: Path, book_url: str, docs: Path = DOCS) -> list[str]:
    """Copy the runnable notebooks' chapters to ``out``; return the notebooks' paths, relative to ``out``.

    Each chapter keeps its place (``part-*/chNN-slug/src/``), so chapter code that puts another chapter's
    ``src/`` on ``sys.path`` finds it; chapters reached that way are copied too, without their notebooks.
    """
    notebooks = [nb for nb in sorted(docs.glob(NOTEBOOK_GLOB)) if runnable(nb, docs)]
    folders = {nb.parent for nb in notebooks}
    for nb in notebooks:
        folders |= {module.parent for module in scan(nb, docs)[1]}
    for folder in sorted(folders):
        for path in sorted(folder.rglob('*')):
            rel = path.relative_to(folder)
            if path.is_file() and path.suffix != '.ipynb' and all(staged_file(Path(p)) for p in rel.parts):
                dest = out / folder.relative_to(docs) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    for nb in notebooks:
        stage_notebook(nb, out / nb.relative_to(docs), book_url, docs)
    rel = [nb.relative_to(docs).as_posix() for nb in notebooks]
    (out / 'runnable.txt').write_text(''.join(f'{r}\n' for r in rel), encoding='utf-8', newline='\n')
    return rel


def build_lite(site_dir: Path, book_url: str, docs: Path = DOCS) -> int:
    """Build the JupyterLite site into ``site_dir/lite``; return the number of runnable notebooks."""
    with tempfile.TemporaryDirectory(prefix='lite-') as tmp:
        work = Path(tmp)
        contents = work / 'contents'
        count = len(stage(contents, book_url, docs))
        (contents / 'runnable.txt').unlink()
        shutil.copy2(LITE_CONFIG, work / 'jupyter-lite.json')
        # Only the Pyodide kernel: the environment's other JupyterLab extensions (jupytext's, which mkdocs-jupyter
        # pulls in) would be copied into the site too.
        kernel = Path(sys.prefix) / 'share' / 'jupyter' / 'labextensions' / '@jupyterlite' / 'pyodide-kernel-extension'
        command = [
            sys.executable, '-m', 'jupyterlite_core', 'build',
            '--lite-dir', str(work), '--contents', str(contents), '--output-dir', str(site_dir / LITE_DIR),
            '--apps', 'notebooks', '--apps', 'tree', '--no-sourcemaps', '--no-unused-shared-packages',
            '--ignore-sys-prefix', f'--LiteBuildConfig.federated_extensions={kernel}',
        ]  # fmt: skip
        result = subprocess.run(command, cwd=work, capture_output=True, text=True, check=False)  # noqa: S603 -- fixed argv
        if result.returncode:
            from mkdocs.exceptions import PluginError

            raise PluginError(f'jupyter lite build failed:\n{result.stdout}\n{result.stderr}')
    return count


# --- rendering the notebook pages --------------------------------------------------------------------------

HREF = re.compile(r'(<a\s[^>]*?href=")([^"]*)(")')
PROTECTED = re.compile(r'(<(pre|code|script|style)\b.*?</\2>)', re.S)
DISPLAY_MATH = re.compile(r'\$\$(.+?)\$\$', re.S)
INLINE_MATH = re.compile(r'(?<![\\$])\$(?=\S)([^$]+?)(?<=\S)\$(?!\$)')
BIG_STYLE = re.compile(r'<style type="text/css">(.{2000,}?)</style>', re.S)
EMPTY_SCRIPT = re.compile(r'<script src="">\s*</script>')
CSS_DIR = 'assets/notebooks'


def is_notebook(page) -> bool:
    """True for the pages mkdocs-jupyter renders from a notebook."""
    return page.file.src_uri.endswith('.ipynb')


def page_links(html: str, src_uri: str, resolve) -> str:
    """Rewrite the relative links of a notebook page; ``resolve(docs_path, fragment)`` gives the new href.

    ``src_uri`` is the notebook's path in ``docs/``; links are relative to it. Links with a scheme, root-relative
    links and fragments alone are left as they are.
    """

    def replace(match: re.Match) -> str:
        href = match.group(2)
        if not href or href.startswith(('#', '/')) or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href):
            return match.group(0)
        target, _, fragment = href.partition('#')
        path = posixpath.normpath(posixpath.join(posixpath.dirname(src_uri), target))
        return match.group(1) + resolve(path, fragment) + match.group(3)

    return HREF.sub(replace, html)


def typeset_math(html: str) -> str:
    """Wrap ``$$...$$`` and ``$...$`` outside code as arithmatex spans, which the book's MathJax typesets."""
    parts = PROTECTED.split(html)
    out = []
    # PROTECTED has two groups, so split() yields text, protected block, tag name, text, ...
    for i in range(0, len(parts), 3):
        text = DISPLAY_MATH.sub(lambda m: f'<span class="arithmatex">\\[{m.group(1)}\\]</span>', parts[i])
        text = INLINE_MATH.sub(lambda m: f'<span class="arithmatex">\\({m.group(1)}\\)</span>', text)
        out.append(text)
        if i + 1 < len(parts):
            out.append(parts[i + 1])
    return ''.join(out)


def buttons(notebook_name: str, run_href: str | None) -> str:
    """The Download button, and the Run button when the notebook runs in the browser."""
    html = f'<p><a class="md-button" href="{notebook_name}" download>Download the notebook</a>'
    if run_href:
        html += (
            f' <a class="md-button md-button--primary" href="{run_href}" target="_blank" rel="noopener"'
            ' title="Opens JupyterLite: Python runs in this browser tab, nothing to install">'
            'Run in your browser</a>'
        )
    return html + '</p>\n'


def insert_after_title(html: str, snippet: str) -> str:
    """``snippet`` right after the first ``</h1>``, or at the top when there is no title."""
    index = html.find('</h1>')
    if index < 0:
        return snippet + html
    index += len('</h1>')
    return html[:index] + snippet + html[index:]


def externalize_css(html: str, page_url: str, store: dict[str, str]) -> str:
    """Replace the large inline ``<style>`` blocks by links to files named by their hash, kept in ``store``."""
    from mkdocs.utils import get_relative_url

    def replace(match: re.Match) -> str:
        css = match.group(1)
        name = f'{CSS_DIR}/{hashlib.sha256(css.encode()).hexdigest()[:12]}.css'
        store[name] = css
        return f'<link rel="stylesheet" href="{get_relative_url(name, page_url)}">'

    return EMPTY_SCRIPT.sub('', BIG_STYLE.sub(replace, html))


# --- MkDocs events ----------------------------------------------------------------------------------------

_state: dict = {'command': 'build', 'css': {}}


def on_startup(command, dirty):
    """Remember whether this is ``mkdocs serve``, which skips the JupyterLite build."""
    _state['command'] = command


def on_page_content(html, page, config, files):
    """Notebook pages: resolve links, typeset math, add the Download and Run buttons."""
    if not is_notebook(page):
        return html
    from mkdocs.exceptions import PluginError
    from mkdocs.utils import get_relative_url

    def resolve(path: str, fragment: str) -> str:
        target = files.get_file_from_path(path)
        if target is None:
            raise PluginError(f"Notebook '{page.file.src_uri}' links to '{path}', which is not in the book")
        return target.url_relative_to(page.file) + (f'#{fragment}' if fragment else '')

    html = typeset_math(page_links(html, page.file.src_uri, resolve))
    notebook = Path(page.file.abs_src_path)
    run = None
    if runnable(notebook, Path(config['docs_dir'])):
        run = get_relative_url(f'{LITE_DIR}/notebooks/index.html', page.url) + f'?path={page.file.src_uri}'
    return insert_after_title(html, buttons(notebook.name, run))


def on_post_page(output, page, config):
    """Notebook pages: link the shared JupyterLab CSS instead of inlining it."""
    if not is_notebook(page):
        return output
    return externalize_css(output, page.url, _state['css'])


def on_post_build(config):
    """Write the shared CSS files and build the JupyterLite site."""
    site_dir = Path(config['site_dir'])
    for name, css in _state['css'].items():
        (site_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (site_dir / name).write_text(css, encoding='utf-8', newline='\n')
    if os.environ.get('LITE', '0' if _state['command'] == 'serve' else '1') != '1':
        log.info('JupyterLite build skipped (LITE=1 builds it)')
        return
    version = os.environ.get('MIKE_DOCS_VERSION')  # set by `mike deploy`, which publishes one folder per version
    book_url = config['site_url'] + (f'{version}/' if version else '')
    count = build_lite(site_dir, book_url, Path(config['docs_dir']))
    log.info(f'JupyterLite: {count} notebooks run in the browser')


def main(argv: list[str]) -> int:
    """``stage <dir> [book_url]``: write the JupyterLite contents and runnable.txt, for ``make test-lite``."""
    if len(argv) not in (3, 4) or argv[1] != 'stage':
        print(main.__doc__)
        return 2
    out = Path(argv[2])
    if out.exists():
        shutil.rmtree(out)
    book_url = argv[3] if len(argv) == 4 else 'https://example.invalid/'
    for rel in stage(out, book_url):
        print(rel)
    for nb in sorted(DOCS.glob(NOTEBOOK_GLOB)):
        reason = why_not_runnable(nb)
        if reason:
            print(f'not runnable: {key(nb)}: {reason}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
