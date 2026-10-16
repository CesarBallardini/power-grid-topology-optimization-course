import math

import numpy as np
from rl_load import correction_capacitor, impedance, load_power, power_factor


def test_exercise_1_numbers():
    z = impedance(10.0, 0.050, 50.0)
    i, s = load_power(230.0, z)
    np.testing.assert_allclose([z.real, z.imag], [10.0, 15.708], atol=1e-3)
    np.testing.assert_allclose(abs(i), 12.35, atol=5e-3)
    np.testing.assert_allclose([s.real, s.imag], [1526, 2396], atol=1)
    np.testing.assert_allclose(power_factor(s), 0.537, atol=5e-4)
    _, c = correction_capacitor(s, 0.95, 230.0, 50.0)
    np.testing.assert_allclose(c * 1e6, 114, atol=0.1)


def test_capacitor_reaches_target_power_factor():
    z = impedance(10.0, 0.050, 50.0)
    _, s = load_power(230.0, z)
    _, c = correction_capacitor(s, 0.95, 230.0, 50.0)
    y_total = 1 / z + 1j * 2 * math.pi * 50.0 * c
    _, s_new = load_power(230.0, 1 / y_total)
    np.testing.assert_allclose(power_factor(s_new), 0.95, atol=1e-9)
    np.testing.assert_allclose(s_new.real, s.real)  # the capacitor changes Q only
