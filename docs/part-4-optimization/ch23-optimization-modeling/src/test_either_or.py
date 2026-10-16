import numpy as np
import pytest
from either_or import BOX, M_BOX, smallest_valid_m, solve


def test_smallest_valid_big_ms():
    assert smallest_valid_m() == (pytest.approx(2.0), pytest.approx(6.0))
    assert M_BOX == (16.0, 34.0)
    assert M_BOX[0] == 2 * BOX - 4 and M_BOX[1] == 4 * BOX - 6


def test_integer_optimum_is_six_for_every_valid_m():
    for m1, m2 in (smallest_valid_m(), M_BOX, (1000.0, 1000.0)):
        z, x, y = solve(m1, m2, relax=False)
        assert z == pytest.approx(6.0)
        np.testing.assert_allclose(x, [0.0, 6.0], atol=1e-6)
        assert y == pytest.approx(1.0)  # y = 1 switches the first constraint off


def test_too_small_an_m_cuts_feasible_points_off():
    z, _, _ = solve(1.0, 6.0, relax=False)  # M1 = 1 < 2 forbids (0, 6)
    assert z < 6.0
    z, _, _ = solve(2.0, 5.0, relax=False, objective=(1.0, 0.0))  # M2 = 5 < 6 forbids (4, 0)
    assert z < 4.0


def test_relaxation_bound_grows_with_m_towards_the_box():
    tight = solve(*smallest_valid_m(), relax=True)[0]
    box = solve(*M_BOX, relax=True)[0]
    huge = solve(10 * M_BOX[0], 10 * M_BOX[1], relax=True)[0]
    assert tight == pytest.approx(6.0) and tight <= box <= huge <= 2 * BOX + 1e-9
    assert huge == pytest.approx(2 * BOX)
