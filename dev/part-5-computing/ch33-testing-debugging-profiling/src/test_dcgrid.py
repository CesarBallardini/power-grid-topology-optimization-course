import dcgrid
import numpy as np
import pandapower.networks as pn


def test_ptdf_reproduces_pandapower_dc_flows_on_case14():
    data = dcgrid.pandapower_dc(pn.case14())
    h = dcgrid.ptdf_matrix(data.fr, data.to, data.b, data.n_bus, data.slack)
    np.testing.assert_allclose(h @ data.p, data.flows, rtol=0, atol=1e-9)
    assert np.all(h[:, data.slack] == 0.0)


def test_lodf_diagonal_and_bridge_denominator():
    fr, to, b = np.array([0, 1, 2, 0]), np.array([1, 2, 0, 3]), np.ones(4)
    h = dcgrid.ptdf_matrix(fr, to, b, 4)
    lodf, den = dcgrid.lodf_matrix(h, fr, to)
    assert np.all(np.diag(lodf) == -1.0)
    assert den[3] == 0.0  # branch 0-3 is a bridge
    np.testing.assert_allclose(den[:3], 1 / 3, atol=1e-12)


def test_without_drops_one_branch():
    data = dcgrid.pandapower_dc(pn.case14())
    smaller = data.without(0)
    assert len(smaller.fr) == len(data.fr) - 1
    assert smaller.fr[0] == data.fr[1]


def test_n1_mismatch_is_zero_only_with_the_diagonal_fixed():
    fr, to, b = np.array([0, 1, 2, 0]), np.array([1, 2, 0, 2]), np.array([1.0, 2.0, 1.0, 1.5])
    p = np.array([0.0, 1.0, -1.0])
    assert dcgrid.n1_mismatch(fr, to, b, p) < 1e-12
    assert dcgrid.n1_mismatch(fr, to, b, p, fix_diagonal=False) > 0.1
