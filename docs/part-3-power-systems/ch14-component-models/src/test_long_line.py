import numpy as np
from long_line import abcd, exact_pi, nominal_pi, series_admittance_error


def test_abcd_is_reciprocal_and_symmetric():
    m = abcd(300.0)
    assert np.isclose(np.linalg.det(m), 1.0)
    assert np.isclose(m[0, 0], m[1, 1])


def test_short_line_is_the_nominal_pi():
    (z_nom, y_nom), (z_ex, y_ex) = nominal_pi(1.0), exact_pi(1.0)
    assert np.isclose(z_nom, z_ex, rtol=1e-6) and np.isclose(y_nom, y_ex, rtol=1e-6)


def test_error_grows_with_the_square_of_the_length():
    e100, e200 = series_admittance_error(np.array([100.0, 200.0]))
    assert 0 < e100 < e200
    assert np.isclose(e200 / e100, 4.0, rtol=0.05)


def test_exact_pi_reproduces_the_abcd_matrix():
    series, shunt = exact_pi(300.0)
    m = abcd(300.0)
    assert np.isclose(1 + series * shunt / 2, m[0, 0])
    assert np.isclose(shunt * (1 + series * shunt / 4), m[1, 0])
