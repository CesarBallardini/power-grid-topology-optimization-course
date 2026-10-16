import numpy as np
from dc_power_flow import dc_power_flow


def test_ring_splits_by_reactance():
    frm, to = np.array([0, 1, 0]), np.array([1, 2, 2])
    f = dc_power_flow(frm, to, np.array([0.1, 0.1, 0.2]), np.array([100.0, 0.0, -100.0]), slack=0)
    np.testing.assert_allclose(f, [50.0, 50.0, 50.0])


def test_kirchhoff_and_slack_independence():
    rng = np.random.default_rng(16)
    frm, to = np.array([0, 0, 1, 1, 2, 3]), np.array([1, 2, 2, 3, 3, 4])
    x = rng.uniform(0.05, 0.3, 6)
    p = rng.normal(size=5)
    p[0] -= p.sum()  # balanced injections
    f = dc_power_flow(frm, to, x, p, slack=0)
    a = np.zeros((6, 5))
    a[np.arange(6), frm], a[np.arange(6), to] = 1, -1
    np.testing.assert_allclose(a.T @ f, p, atol=1e-12)  # flows out of each node equal its injection
    np.testing.assert_allclose(dc_power_flow(frm, to, x, p, slack=3), f, atol=1e-12)
