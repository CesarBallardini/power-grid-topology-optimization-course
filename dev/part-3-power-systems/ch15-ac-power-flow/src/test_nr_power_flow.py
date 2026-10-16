import numpy as np
import pandapower as pp
import pytest
from nr_power_flow import (
    CASE5,
    dc_start,
    make_ybus,
    max_loading,
    pandapower_net,
    pandapower_solution,
    pv_to_pq,
    scaled_loads,
    solve,
    without_branches,
)


def test_ybus_matches_pandapower_internal_matrix():
    net = pandapower_net(CASE5)
    pp.runpp(net)
    y_pp = net._ppc['internal']['Ybus'].toarray()
    np.testing.assert_allclose(make_ybus(CASE5), y_pp, atol=1e-9)


def test_quadratic_convergence_and_pandapower_agreement():
    theta, vm, hist, ok = solve(CASE5)
    assert ok and len(hist) == 4
    # the exponent of the mismatch roughly doubles at every step
    assert hist[2] < hist[1] ** 2 * 10 and hist[3] < hist[2] ** 2 * 10
    th_pp, vm_pp = pandapower_solution(CASE5)
    np.testing.assert_allclose(vm, vm_pp, atol=1e-8)
    np.testing.assert_allclose(theta, th_pp, atol=1e-8)


def test_loadability_limit_and_failure_signature():
    lam = max_loading(CASE5)
    assert lam == pytest.approx(5.2)
    _, vm, hist, ok = solve(scaled_loads(CASE5, lam + 0.05))
    assert not ok and len(hist) == 20
    assert vm.min() < 0.7  # the iteration wanders around the nose of the curve
    assert max_loading(pv_to_pq(CASE5, 2)) < lam  # without the voltage source the limit comes earlier


def test_island_makes_the_jacobian_singular():
    with pytest.raises(np.linalg.LinAlgError):
        solve(without_branches(CASE5, [4, 6]))


def test_starts_reach_the_same_solution():
    heavy = scaled_loads(CASE5, 5.0)
    flat = solve(heavy)
    dc = solve(heavy, dc_start(heavy))
    assert flat[3] and dc[3]
    np.testing.assert_allclose(flat[1], dc[1], atol=1e-7)
    previous = solve(heavy, solve(scaled_loads(CASE5, 4.9))[:2])
    assert len(previous[2]) < len(flat[2])
