import numpy as np
import pytest
from ac_jacobian import injections, jacobian_blocks


@pytest.fixture
def three_bus():
    rng = np.random.default_rng(3)
    yl = 1 / (0.02 + 0.2j)
    y = np.array([[2 * yl, -yl, -yl], [-yl, 2 * yl, -yl], [-yl, -yl, 2 * yl]]) + 0.05j * np.eye(3)
    theta = rng.uniform(-0.3, 0.3, 3)
    vm = rng.uniform(0.95, 1.05, 3)
    return theta, vm, y


def test_injections_balance_without_shunts():
    yl = 1 / (0.01 + 0.1j)
    y = np.array([[yl, -yl], [-yl, yl]])
    s = injections(np.array([0.0, -0.05]), np.array([1.0, 0.98]), y)
    assert s[1].real < 0 < s[0].real  # power flows from bus 0 to bus 1
    assert s.real.sum() > 0  # the series resistance dissipates the difference


def test_jacobian_blocks_match_finite_differences(three_bus):
    theta, vm, y = three_bus
    j11, j12, j21, j22 = jacobian_blocks(theta, vm, y)
    h = 1e-7
    for k in range(3):
        dth = np.zeros(3)
        dth[k] = h
        ds = (injections(theta + dth, vm, y) - injections(theta - dth, vm, y)) / (2 * h)
        np.testing.assert_allclose(j11[:, k], ds.real, atol=1e-6)
        np.testing.assert_allclose(j21[:, k], ds.imag, atol=1e-6)
        dv = np.zeros(3)
        dv[k] = h
        ds = (injections(theta, vm + dv, y) - injections(theta, vm - dv, y)) / (2 * h)
        np.testing.assert_allclose(j12[:, k], ds.real, atol=1e-6)
        np.testing.assert_allclose(j22[:, k], ds.imag, atol=1e-6)
