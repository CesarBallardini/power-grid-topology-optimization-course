"""Checks the chapter's N-1 flows against ToOp's NumPy reference (``packages/dc_solver_pkg/tests/numpy_reference.py``).

Runs only with ToOp's interpreter (`make test-toop`); the reference file is found through ``TOOP_REPO`` (default:
the ``ToOp`` folder next to this repository), as ``tools/check_toop_paths.py`` does.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
from lodf import ptdf_matrix  # noqa: E402
from n1_screening import case_data, lodf_matrix, n1_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = Path(os.environ.get('TOOP_REPO', ROOT.parent / 'ToOp')) / 'packages/dc_solver_pkg/tests/numpy_reference.py'


@pytest.fixture(scope='module')
def reference():
    if not REFERENCE.is_file():
        pytest.skip(f'{REFERENCE} not found: set TOOP_REPO')
    spec = importlib.util.spec_from_file_location('numpy_reference', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['case14', 'case118'])
def test_n1_flows_match_toop_reference(reference, name):
    frm, to, x, p, slack = case_data(name)
    h = ptdf_matrix(frm, to, x, len(p), slack)
    lodf = lodf_matrix(h, frm, to)
    ours = n1_matrix(lodf, h @ p)
    theirs, success = reference.contingency_analysis(h, frm, to, np.arange(len(frm)), p)
    assert (~success).sum() == np.isnan(lodf).all(axis=0).sum()  # the same bridges
    np.testing.assert_allclose(ours[success], theirs[success], atol=1e-9)
    for k in np.flatnonzero(success)[:5]:
        np.testing.assert_allclose(reference.calc_lodf(h, frm, to, int(k)), lodf[:, k], atol=1e-12)
