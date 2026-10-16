import cmath
import math

import numpy as np
from three_phase import (
    A_MATRIX,
    A_OP,
    balanced_set,
    instantaneous_power,
    line_voltages,
    relay_impedance,
    symmetrical_components,
    wye_load,
)


def test_operator_and_line_voltages():
    np.testing.assert_allclose(1 + A_OP + A_OP**2, 0, atol=1e-12)
    v = balanced_set()
    v_ll = line_voltages(v)
    np.testing.assert_allclose(np.abs(v_ll), math.sqrt(3))
    np.testing.assert_allclose(math.degrees(cmath.phase(v_ll[0])), 30.0)
    np.testing.assert_allclose(v_ll.sum(), 0, atol=1e-12)


def test_balanced_currents_have_no_neutral_current():
    for phi in (0.0, 30.0, 45.0, 90.0):
        i = balanced_set(1.0, -phi)
        np.testing.assert_allclose(i.sum(), 0, atol=1e-12)
        np.testing.assert_allclose(math.degrees(cmath.phase(balanced_set()[1] / i[1])), phi)


def test_exercise_1_wye_load():
    i_a, s, z_delta = wye_load(30 + 40j, 400.0)
    np.testing.assert_allclose(abs(i_a), 400 / math.sqrt(3) / 50)
    np.testing.assert_allclose([s.real, s.imag], [1920.0, 2560.0])
    np.testing.assert_allclose(abs(s), math.sqrt(3) * 400 * abs(i_a))
    assert z_delta == 90 + 120j


def test_exercise_2_total_power_is_constant():
    _, p_phase, p_total = instantaneous_power(230.0, 10.0, 30.0)
    np.testing.assert_allclose(p_total, 3 * 2300 * math.cos(math.radians(30.0)), rtol=1e-9)
    assert p_phase[0].max() - p_phase[0].min() > 4000  # each phase alone swings by 2 S


def test_exercise_3_relay_impedance():
    z_abs, r, x = relay_impedance(380.0, 1000.0, 600.0, 200.0)
    np.testing.assert_allclose([z_abs, r, x], [219.4, 208.1, 69.4], atol=0.05)
    np.testing.assert_allclose(math.hypot(r, x), z_abs)


def test_exercise_5_symmetrical_components():
    v_012 = symmetrical_components(np.array([1.0, 0.9 * A_OP**2, A_OP]))
    np.testing.assert_allclose(np.abs(v_012), [0.0333, 0.9667, 0.0333], atol=5e-5)
    np.testing.assert_allclose(A_MATRIX @ v_012, [1.0, 0.9 * A_OP**2, A_OP], atol=1e-12)
    np.testing.assert_allclose(np.abs(symmetrical_components(balanced_set())), [0, 1, 0], atol=1e-12)
