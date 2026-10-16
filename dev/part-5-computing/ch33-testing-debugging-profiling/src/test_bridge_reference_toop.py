"""ToOp's numpy reference raises on the bridge where the jitted kernel returns NaN (Chapter 33, Exercise 2).

Run with ToOp's interpreter from the repository root::

    ../ToOp/.venv/Scripts/python.exe -m pytest -m toop -o addopts="" -p no:cacheprovider -W ignore <this file>
"""

import os
import sys
from pathlib import Path

import bridge_debug
import dcgrid
import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

TOOP = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
sys.path.insert(0, str(TOOP / 'packages' / 'dc_solver_pkg' / 'tests'))
numpy_reference = pytest.importorskip('numpy_reference')


def test_reference_raises_value_error_for_the_bridge():
    fr, to, b, _ = bridge_debug.bridge_grid()
    ptdf = dcgrid.ptdf_matrix(fr, to, b, 6)
    with pytest.raises(ValueError, match='split the grid'):
        numpy_reference.calc_lodf(ptdf, fr, to, 5)


def test_reference_agrees_with_the_jitted_kernel_on_meshed_outages():
    fr, to, b, p = bridge_debug.bridge_grid()
    ptdf = dcgrid.ptdf_matrix(fr, to, b, 6)
    flows = bridge_debug.run_plain()
    f0 = ptdf @ p
    for outage in range(5):
        lodf = numpy_reference.calc_lodf(ptdf, fr, to, outage)
        np.testing.assert_allclose(flows[outage], f0 + lodf * f0[outage], atol=1e-12)
