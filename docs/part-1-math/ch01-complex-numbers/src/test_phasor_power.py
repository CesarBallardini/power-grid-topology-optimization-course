import cmath
import math

from phasor_power import complex_power, load_current


def test_current_magnitude_and_angle():
    i = load_current(230, 40 + 30j)
    assert math.isclose(abs(i), 4.6)
    assert math.isclose(math.degrees(cmath.phase(i)), -36.87, abs_tol=0.005)


def test_complex_power_components():
    s = complex_power(230, load_current(230, 40 + 30j))
    assert math.isclose(s.real, 846.4) and math.isclose(s.imag, 634.8)
    assert math.isclose(abs(s), 1058.0)
    assert math.isclose(abs(s), math.hypot(s.real, s.imag))
