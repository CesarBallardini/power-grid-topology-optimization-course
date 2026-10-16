import cmath
import math

import pytest
from complex_plane import (
    LOADS,
    atan_vs_atan2,
    exercise1,
    load_study,
    polar_deg,
    product_check,
    roots_of_unity,
    rotate,
    rotating_phasor,
    tap_phase_shift,
    wrap_deg,
)


def test_rotation_keeps_the_modulus_and_adds_the_angle():
    z = 3 + 4j
    for theta in (0, 30, 90, 180, 270, 359):
        mod, arg = polar_deg(rotate(z, theta))
        assert mod == pytest.approx(5.0)
        assert arg == pytest.approx(wrap_deg(polar_deg(z)[1] + theta), abs=1e-9)


def test_product_is_rotate_and_scale():
    c = product_check(3 + 4j, 1 + 0.5j)
    assert c['abs_p'] == pytest.approx(c['abs_z_abs_u']) == pytest.approx(math.sqrt(1 + 5.5**2))
    assert c['arg_p'] == pytest.approx(c['arg_z_plus_arg_u'])
    assert (3 + 4j) * (1 + 0.5j) == 1 + 5.5j


def test_wrap_deg_lands_in_the_half_open_interval():
    assert wrap_deg(190) == pytest.approx(-170)
    assert wrap_deg(-190) == pytest.approx(170)
    assert wrap_deg(180) == 180
    assert wrap_deg(540) == 180


def test_cube_roots_of_unity_sum_to_zero():
    roots = roots_of_unity(3)
    assert all(abs(r**3 - 1) < 1e-12 for r in roots)
    assert abs(sum(roots)) < 1e-12
    a = roots[1]
    assert a == pytest.approx(cmath.exp(2j * math.pi / 3))
    assert abs(1 + a + a**2) < 1e-12


def test_three_loads_and_the_sign_of_q():
    rows = {r['load']: r for r in load_study(230, LOADS)}
    for r in rows.values():
        assert r['abs_s'] == pytest.approx(r['hypot'])
        assert r['s'] == pytest.approx(230 * r['i'].conjugate())
    assert rows['inductive']['s'].imag > 0 and rows['inductive']['q'].startswith('Q > 0')
    assert rows['capacitive']['s'].imag < 0 and rows['capacitive']['q'].startswith('Q < 0')
    assert rows['resistive']['s'].imag == pytest.approx(0) and rows['resistive']['q'] == 'Q = 0'
    assert rows['resistive']['s'].real == pytest.approx(230**2 / 50)


def test_exercise_1_by_hand():
    z = exercise1()
    assert z == pytest.approx(0.5 - 3.5j)
    mod, arg = polar_deg(z)
    assert mod == pytest.approx(3.536, abs=5e-4)
    assert arg == pytest.approx(-81.87, abs=5e-3)


def test_atan_fails_in_the_third_quadrant():
    by_atan, by_atan2 = atan_vs_atan2(-1 - 1j)
    assert by_atan == pytest.approx(45.0)
    assert by_atan2 == pytest.approx(-135.0)
    assert by_atan2 == pytest.approx(polar_deg(-1 - 1j)[1])


def test_tap_phase_shift_two_ways():
    for phi, expected in ((90, 2.86), (60, 2.42)):
        by_arg, by_atan2 = tap_phase_shift(400, 5, 1, phi)
        assert by_arg == pytest.approx(by_atan2, abs=1e-12)
        assert by_arg == pytest.approx(expected, abs=5e-3)


def test_rotating_phasor_projects_to_a_cosine():
    omega = 2 * math.pi
    for t in (0, 0.1, 0.25, 0.5, 0.9):
        u, x = rotating_phasor(omega, t)
        assert abs(u) == pytest.approx(1)
        assert x == pytest.approx(math.cos(omega * t))
