"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
from compose_toop import composed_n1, grid_from_toop, load, reference_n1, topo_vect  # noqa: E402
from ieee14 import load_case14  # noqa: E402
from toop_engine_dc_solver.preprocess.convert_to_jax import run_initial_loadflow  # noqa: E402


@pytest.fixture(scope='module')
def case14(tmp_path_factory):
    si, nd = load(tmp_path_factory.mktemp('case14'))
    return si, nd


def test_unsplit_flows_agree_with_run_initial_loadflow(case14):
    si, nd = case14
    si, metrics = run_initial_loadflow(si)
    unsplit = -np.asarray(si.dynamic_information.unsplit_flow)[0]
    np.testing.assert_allclose(unsplit, load_case14().flows(), atol=1e-6)
    np.testing.assert_allclose(unsplit, grid_from_toop(nd).flows(), atol=1e-6)
    assert metrics == (0.0, 0.0)  # case14 has no ratings


def test_split_and_n1_screen_agree_with_numpy_reference(case14):
    _, nd = case14
    grid = grid_from_toop(nd)
    bus_a = [grid.names.index('L2-4'), grid.names.index('L4-5')]
    outages = [k for k in range(grid.n_branch) if nd.outaged_branch_mask[k]]
    assert len(outages) == 19  # T7-8, the stub, is excluded from N-1 by the example's masks
    flags = topo_vect(nd, 3, bus_a)
    assert flags.sum() == 3  # L3-4, T4-7 and T4-9 go to bus B
    ref, ok = reference_n1(nd, 3, bus_a, outages)
    assert ok.all()
    np.testing.assert_allclose(composed_n1(grid, 3, bus_a, outages), ref, atol=1e-9)
