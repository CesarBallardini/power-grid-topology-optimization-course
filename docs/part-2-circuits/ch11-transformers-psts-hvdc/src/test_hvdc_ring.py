import numpy as np
from hvdc_ring import LOAD, NAMES, ac_flows, best_setpoint, sensitivity


def test_base_case_balances_the_loads():
    f0 = ac_flows(0.0)
    np.testing.assert_allclose(f0[0] + f0[2], LOAD.sum())  # everything leaves bus 1 over L1-2 and L1-3
    np.testing.assert_allclose(f0[0] - f0[1], LOAD[1])  # KCL at bus 2
    assert NAMES[int(np.argmax(np.abs(f0)))] == 'L1-2'


def test_flows_are_affine_in_p_dc():
    f0, d = ac_flows(0.0), sensitivity()
    for p in (-0.5, 0.3, 1.2):
        np.testing.assert_allclose(ac_flows(p), f0 + p * d, atol=1e-12)
    np.testing.assert_allclose(d[0], d[1])  # the path 1-2-3 of the ring carries one share, L1-3 the other
    np.testing.assert_allclose(d[0] + d[2], -1.0)  # together they move the whole 1 pu off the AC lines


def test_minimax_setpoint():
    p_best, f_max = best_setpoint()
    np.testing.assert_allclose([p_best, f_max], [0.775, 0.3], atol=1e-6)
    flows = ac_flows(p_best)
    np.testing.assert_allclose(np.abs(flows).max(), f_max, atol=1e-6)
    for p in (p_best - 0.1, p_best + 0.1):
        assert np.abs(ac_flows(p)).max() > f_max
