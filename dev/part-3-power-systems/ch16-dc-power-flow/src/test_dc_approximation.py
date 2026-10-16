import numpy as np
import pytest
from dc_approximation import angles_and_flows, error_bounds, relative_error


def test_relative_error_vanishes_for_flat_voltage_and_small_angle():
    assert relative_error(1.0, 1.0, 1e-6) == pytest.approx(0.0, abs=1e-9)
    assert relative_error(1.0, 1.0, np.deg2rad(10)) == pytest.approx(1 / np.sinc(10 / 180) - 1)


def test_bounds_are_set_by_the_voltage_corners():
    lo, hi = error_bounds()
    assert lo == pytest.approx(1 / 1.05**2 - 1, abs=1e-6)
    assert hi == pytest.approx(np.deg2rad(10) / (0.95**2 * np.sin(np.deg2rad(10))) - 1, abs=1e-6)


def test_flows_do_not_depend_on_the_slack_but_angles_do():
    frm, to = np.array([0, 0, 1, 2]), np.array([1, 2, 3, 3])
    x = np.array([0.1, 0.2, 0.1, 0.2])
    p = np.array([0.8, 0.2, -0.5, -0.5])
    th0, f0 = angles_and_flows(frm, to, x, p, 0)
    th3, f3 = angles_and_flows(frm, to, x, p, 3)
    np.testing.assert_allclose(f0, f3, atol=1e-12)
    shift = th3 - th0
    np.testing.assert_allclose(shift, shift[0], atol=1e-12)
    assert abs(shift[0]) > 1e-3
