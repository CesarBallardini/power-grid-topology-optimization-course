import numpy as np
import pytest
from compose import brute_force, busbar_outage, compose, coupler_row, moved_ends, n1_screen, split
from ieee14 import load_case14

STATION = 3  # bus 4


@pytest.fixture(scope='module')
def grid():
    return load_case14()


@pytest.fixture(scope='module')
def case(grid):
    bus_a = [grid.names.index('L2-4'), grid.names.index('L4-5')]
    return bus_a, grid.names.index('L6-13'), 8


def test_coupler_row_reproduces_kcl_at_bus_a(grid, case):
    bus_a, _, _ = case
    h = grid.ptdf()
    p_c = coupler_row(h, grid.frm, grid.to, STATION, bus_a)
    f = grid.flows()
    a_to = [k for k in bus_a if grid.to[k] == STATION]
    a_from = [k for k in bus_a if grid.frm[k] == STATION]
    assert p_c @ grid.p == pytest.approx(f[a_to].sum() - f[a_from].sum() + grid.p[STATION])


def test_split_matches_physically_split_grid(grid, case):
    bus_a, _, _ = case
    s, p_ext = split(grid.ptdf(), grid.frm, grid.to, grid.b, grid.p, STATION, bus_a)
    ref = grid.split(STATION, bus_a)
    np.testing.assert_allclose(s.h, ref.ptdf(), atol=1e-12)
    np.testing.assert_allclose(s.f, ref.flows(), atol=1e-9)
    np.testing.assert_allclose(grid.flows() + s.c * s.beta, s.h @ p_ext, atol=1e-9)
    frm, to = moved_ends(grid.frm, grid.to, STATION, bus_a, grid.n_bus)
    np.testing.assert_array_equal(frm, ref.frm)
    np.testing.assert_array_equal(to, ref.to)


def test_composition_in_toop_order_matches_brute_force(grid, case):
    bus_a, disconnect, busbar = case
    c = compose(grid, STATION, bus_a, disconnect, busbar)
    f_ref, n1_ref, f_bb_ref = brute_force(grid, STATION, bus_a, disconnect, busbar)
    keep = [k for k in range(grid.n_branch) if k != disconnect]
    np.testing.assert_allclose(c.f[keep], f_ref, atol=1e-9)
    finite = np.isfinite(c.n1).all(axis=1)  # the outage of the stub T7-8 is a bridge: NaN row from the LODF
    assert finite.sum() == len(keep) - 1
    np.testing.assert_allclose(c.n1[finite][:, keep], n1_ref[finite], atol=1e-9)
    np.testing.assert_allclose(c.f_bb[keep], f_bb_ref, atol=1e-9)
    assert c.f[disconnect] == 0 and np.all(c.h[disconnect] == 0)


def test_order_changes_beta_and_c_but_not_the_result(grid, case):
    bus_a, disconnect, busbar = case
    c1 = compose(grid, STATION, bus_a, disconnect, busbar)
    c2 = compose(grid, STATION, bus_a, disconnect, busbar, split_first=False)
    np.testing.assert_allclose(c1.h, c2.h, atol=1e-12)
    np.testing.assert_allclose(c1.f, c2.f, atol=1e-9)
    np.testing.assert_allclose(c1.n1, c2.n1, atol=1e-9)
    np.testing.assert_allclose(c1.f_bb, c2.f_bb, atol=1e-9)
    assert np.abs(c1.beta - c2.beta).max() > 0.01
    assert abs(c1.c - c2.c) > 1.0


def test_n1_screen_marks_bridges_with_nan(grid):
    h, f = grid.ptdf(), grid.flows()
    n1 = n1_screen(h, f, grid.frm, grid.to, list(range(grid.n_branch)))
    stub = grid.names.index('T7-8')
    assert np.all(np.isnan(np.delete(n1[stub], stub)))
    assert np.isfinite(np.delete(n1, stub, axis=0)).all()


def test_busbar_outage_retries_with_a_skeleton_branch(grid):
    h, f = grid.ptdf(), grid.flows()
    f_bb, branches = busbar_outage(h, f, grid.frm, grid.to, grid.p, 7)  # bus 8: only T7-8, a bridge
    assert branches == []  # the only branch stays as skeleton, only the injection is removed
    np.testing.assert_allclose(f_bb, f - h[:, 7] * grid.p[7], atol=1e-9)
    # bus 9 has four branches; outaging all four islands the busbar itself, so the first (L9-10) stays as skeleton
    f_bb9, branches9 = busbar_outage(h, f, grid.frm, grid.to, grid.p, 8)
    assert [grid.names[k] for k in branches9] == ['L9-14', 'T4-9', 'T7-9']
    assert np.isfinite(f_bb9).all() and np.all(f_bb9[branches9] == 0)
    assert f_bb9[grid.names.index('L9-10')] == pytest.approx(0.0, abs=1e-9)  # the skeleton feeds an empty busbar
