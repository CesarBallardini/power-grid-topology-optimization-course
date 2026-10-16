import cmath
import math

import numpy as np
from pst_phasors import (
    phase_shift_deg,
    quadrature_booster,
    quadrature_magnitude_for_shift,
    series_voltage,
    symmetric_pst,
)


def test_quadrature_booster_changes_magnitude_symmetric_does_not():
    u = 1 + 0j
    for k in (0.05, 0.1, 0.2):
        w_q = quadrature_booster(u, k)
        alpha = math.degrees(cmath.phase(w_q))
        np.testing.assert_allclose(alpha, math.degrees(math.atan(k)))
        np.testing.assert_allclose(abs(w_q), math.sqrt(1 + k * k))
        np.testing.assert_allclose(abs(symmetric_pst(u, alpha)), 1.0)
        np.testing.assert_allclose(cmath.phase(symmetric_pst(u, alpha)), cmath.phase(w_q))


def test_general_step_matches_chapter_1_formula():
    for k, phi in ((0.05, 90.0), (0.05, 60.0), (0.2, 30.0), (0.1, 120.0)):
        w = series_voltage(1 + 0j, k, phi)
        np.testing.assert_allclose(math.degrees(cmath.phase(w)), phase_shift_deg(k, phi), atol=1e-12)
    np.testing.assert_allclose([phase_shift_deg(0.05, 90.0), phase_shift_deg(0.05, 60.0)], [2.86, 2.42], atol=0.005)


def test_exercise_5_same_shift_different_magnitudes():
    for alpha in (5.0, 10.0, 20.0):
        k = math.tan(math.radians(alpha))
        w_q = quadrature_booster(1 + 0j, k)
        np.testing.assert_allclose(math.degrees(cmath.phase(w_q)), alpha)
        np.testing.assert_allclose(abs(w_q), quadrature_magnitude_for_shift(alpha))
        np.testing.assert_allclose(abs(symmetric_pst(1 + 0j, alpha)), 1.0)
        assert abs(w_q) > 1.0
