import numpy as np
import pytest
from lp_duality import A, B, exercise_1, objective_ranging, rhs_ranging, solve_lab_lp, vertices


def test_vertices_and_optimum():
    vs = vertices()
    assert len(vs) == 5
    best = max(vs, key=lambda t: t[1])
    np.testing.assert_allclose(best[0], [2.0, 6.0])
    assert best[1] == pytest.approx(36.0)


def test_duals_satisfy_strong_duality_and_complementary_slackness():
    z, x, duals = solve_lab_lp()
    assert z == pytest.approx(36.0)
    np.testing.assert_allclose(x, [2.0, 6.0])
    np.testing.assert_allclose(duals, [0.0, 1.5, 1.0])
    assert B @ duals == pytest.approx(z)
    np.testing.assert_allclose(duals * (B - A @ x), 0.0)


def test_shadow_price_of_the_third_constraint_holds_between_12_and_24():
    rows = {b3: price for b3, _, _, price in rhs_ranging()}
    assert rows[12.0] == rows[18.0] == rows[21.0] == pytest.approx(1.0)
    assert rows[10.0] == pytest.approx(2.5)  # x1 at zero: one more MW of b3 buys half a unit of x2, worth 5
    assert rows[24.0] in (pytest.approx(0.0), pytest.approx(1.0))  # degenerate vertex: either price is valid
    assert rows[27.0] == pytest.approx(0.0)  # constraint 3 no longer binds


def test_objective_ranging_jumps_at_c1_equals_7_5():
    rows = dict(objective_ranging())
    np.testing.assert_allclose(rows[7.0], [2.0, 6.0])
    np.testing.assert_allclose(rows[8.0], [4.0, 3.0])


def test_exercise_1_primal_equals_dual():
    zp, xp, zd, yd = exercise_1()
    assert zp == pytest.approx(3.5) and zd == pytest.approx(3.5)
    np.testing.assert_allclose(xp, [2.5, 1.5])
    np.testing.assert_allclose(yd, [0.5, 1.5])
