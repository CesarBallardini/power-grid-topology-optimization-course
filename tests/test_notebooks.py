"""Execute every chapter notebook (docs/part-*/chNN-slug/src/*.ipynb) headlessly; any failing cell fails the test.

Notebooks are thin drivers of the chapter's ``src`` modules, so each one runs with its own folder as working
directory. Notebooks whose name ends in ``_toop`` need ToOp: they carry the ``toop`` marker, skip when ToOp is not
importable and run under ``make test-toop``. Notebooks whose name ends in ``_labs`` need the labs container
(docker/labs): they carry the ``labs`` marker, skip unless the library named in ``LABS_LIBRARY`` for their chapter
imports, and run under ``make test-labs``. Committed notebooks have no outputs (nbstripout hook).
"""

import importlib.util
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob('docs/part-*/*/src/*.ipynb'))
HAS_TOOP = importlib.util.find_spec('toop_engine_dc_solver') is not None
# The library a chapter's ``_labs`` notebook needs (Grid2Op pulls Gymnasium and Stable-Baselines3 with it in the image)
LABS_LIBRARY = {'ch27-pareto-multi-objective': 'qdax', 'ch29-learning-based-control': 'grid2op'}


def _param(path: Path):
    marks = []
    if path.stem.endswith('_toop'):
        marks.append(pytest.mark.toop)
    if path.stem.endswith('_labs'):
        marks.append(pytest.mark.labs)
    return pytest.param(path, marks=marks, id=f'{path.parents[1].name}/{path.name}')


def test_notebooks_exist_and_are_stripped():
    assert NOTEBOOKS, 'no chapter notebooks found'
    for path in NOTEBOOKS:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                assert not cell.outputs, f'{path} has committed outputs: run nbstripout'
                assert cell.execution_count is None, f'{path} has execution counts: run nbstripout'


@pytest.mark.parametrize('path', [_param(p) for p in NOTEBOOKS])
def test_notebook_executes(path: Path):
    if path.stem.endswith('_toop') and not HAS_TOOP:
        pytest.skip('needs ToOp: run with make test-toop')
    if path.stem.endswith('_labs'):
        library = LABS_LIBRARY.get(path.parents[1].name, 'grid2op')
        if importlib.util.find_spec(library) is None:
            pytest.skip(f'needs {library}: run with make test-labs')
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(path.parent)}})
    client.execute()
