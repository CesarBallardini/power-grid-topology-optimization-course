import numpy as np
import pytest
from ieee14 import load_case14
from modf import ISLAND_TOL, apply_modf, lodf_matrix, modf, modf_2x2, modf_flows, transfer, woodbury


@pytest.fixture(scope='module')
def grid():
    return load_case14()


@pytest.mark.parametrize('names', [['L2-4', 'L4-5'], ['L6-11', 'L9-14', 'T4-9'], ['L1-5']])
def test_modf_update_matches_a_recomputed_ptdf(grid, names):
    out = [grid.names.index(n) for n in names]
    h = grid.ptdf()
    h_new = apply_modf(h, grid.frm, grid.to, out)
    keep = [k for k in range(grid.n_branch) if k not in out]
    np.testing.assert_allclose(h_new[keep], grid.without(out).ptdf(), atol=1e-12)
    assert np.all(h_new[out] == 0)
    f_new = modf_flows(grid.flows(), h, grid.frm, grid.to, out)
    np.testing.assert_allclose(f_new[keep], grid.without(out).flows(), atol=1e-9)


def test_cut_set_gives_zero_determinant_but_no_bridge(grid):
    h = grid.ptdf()
    cut = [grid.names.index('L1-2'), grid.names.index('L1-5')]
    _, det = modf_2x2(h, grid.frm, grid.to, *cut)
    assert abs(det) < ISLAND_TOL
    for k in cut:
        assert abs(1 - transfer(h, grid.frm, grid.to, [k])[k, 0]) > 0.1  # each one alone leaves the grid connected
    assert np.all(np.isnan(modf(h, grid.frm, grid.to, cut)[0]))
    with pytest.raises(ValueError, match='islands'):
        apply_modf(h, grid.frm, grid.to, cut)


def test_closed_form_2x2_equals_the_general_solve(grid):
    h = grid.ptdf()
    out = [grid.names.index('L2-4'), grid.names.index('L4-5')]
    m, det = modf(h, grid.frm, grid.to, out)
    m2, det2 = modf_2x2(h, grid.frm, grid.to, *out)
    np.testing.assert_allclose(m2, m, atol=1e-12)
    assert det2 == pytest.approx(det)


def test_woodbury_matches_recomputation_and_alpha_minus_one_is_modf(grid):
    h = grid.ptdf()
    t = grid.names.index('T5-6')
    np.testing.assert_allclose(woodbury(h, grid.frm, grid.to, [t], [0.2]), grid.scaled(t, 1.2).ptdf(), atol=1e-12)
    out = [grid.names.index('L2-4'), grid.names.index('L4-5')]
    h_w = woodbury(h, grid.frm, grid.to, out, [-1.0, -1.0])
    np.testing.assert_allclose(h_w, apply_modf(h, grid.frm, grid.to, out), atol=1e-12)


def test_lodf_matrix_diagonal_and_bridge_column(grid):
    lodf = lodf_matrix(grid.ptdf(), grid.frm, grid.to)
    assert lodf.shape == (20, 20)
    np.testing.assert_allclose(np.diag(lodf), -1.0)
    stub = grid.names.index('T7-8')
    assert np.all(np.isnan(np.delete(lodf[:, stub], stub)))  # the only bridge of IEEE 14
    k = grid.names.index('L2-3')
    f = grid.flows()
    np.testing.assert_allclose(np.delete(f + lodf[:, k] * f[k], k), grid.without([k]).flows(), atol=1e-9)
