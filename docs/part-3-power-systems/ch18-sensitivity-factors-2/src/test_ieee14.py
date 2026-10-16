import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest
from ieee14 import DcGrid, load_case14, ptdf, ratings


@pytest.fixture(scope='module')
def grid():
    return load_case14()


def test_flows_match_pandapower_dc_power_flow(grid):
    net = pn.case14()
    pp.rundcpp(net)
    ref = np.concatenate([net.res_line.p_from_mw.to_numpy(), net.res_trafo.p_hv_mw.to_numpy()])
    np.testing.assert_allclose(grid.flows(), ref, atol=1e-9)
    assert grid.n_bus == 14 and grid.n_branch == 20 and grid.names[0] == 'L1-2' and grid.names[15] == 'T4-7'


def test_ptdf_rows_sum_to_zero_for_slack_and_kcl_holds(grid):
    h = grid.ptdf()
    assert np.all(h[:, grid.slack] == 0)
    a = np.zeros((grid.n_branch, grid.n_bus))
    a[np.arange(grid.n_branch), grid.frm], a[np.arange(grid.n_branch), grid.to] = 1, -1
    inj = a.T @ grid.flows()
    np.testing.assert_allclose(inj, grid.p, atol=1e-9)


def test_split_adds_a_node_and_moves_the_other_branches(grid):
    bus_a = [grid.names.index('L2-4'), grid.names.index('L4-5')]
    g = grid.split(3, bus_a)
    assert g.n_bus == 15 and g.p[14] == 0
    assert set(g.branches_at(14)) == {grid.names.index(n) for n in ('L3-4', 'T4-7', 'T4-9')}
    assert set(g.branches_at(3)) == set(bus_a)


def test_isolated_node_gets_a_zero_column():
    frm, to, b = np.array([0, 1]), np.array([1, 2]), np.array([1.0, 1.0])
    h = ptdf(frm, to, b, 4)  # node 3 has no branch
    assert h.shape == (2, 4) and np.all(h[:, 3] == 0)
    np.testing.assert_allclose(h[:, 2], [-1.0, -1.0])  # one MW injected at node 2 flows to the slack, against 0->1->2


def test_without_and_scaled_and_moved(grid):
    g = grid.without([0]).scaled(0, 2.0).moved(1, 2, 10.0)
    assert g.n_branch == 19 and g.names[0] == 'L1-5' and g.b[0] == pytest.approx(2 * grid.b[1])
    assert g.p[1] == pytest.approx(grid.p[1] - 10) and g.p[2] == pytest.approx(grid.p[2] + 10)
    assert isinstance(g, DcGrid)


def test_ratings_have_a_floor_and_a_margin(grid):
    lim = ratings(grid)
    assert lim.min() == 40.0 and lim[0] == pytest.approx(1.3 * abs(grid.flows()[0]))
