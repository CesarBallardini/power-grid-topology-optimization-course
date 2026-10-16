import numpy as np
from newton import jacobian, residual
from newton_exercises import (
    SQRT2,
    angle_iterates,
    arctan_critical_point,
    arctan_iterates,
    broyden_good,
    double_root_iterates,
    newton_history,
    sqrt2_iterates,
)


def test_exercise_1_sqrt2_errors_are_squared():
    xs, errs = sqrt2_iterates()
    np.testing.assert_allclose(xs, [1.0, 1.5, 1.41667, 1.414216], atol=5e-6)
    np.testing.assert_allclose(errs[1:], [8.6e-2, 2.5e-3, 2.1e-6], rtol=0.03)
    ratios = [b / a**2 for a, b in zip(errs[:-1], errs[1:], strict=True)]
    assert all(0.3 < q < 0.5 for q in ratios)  # e_{k+1} ≈ e_k² / (2 √2)
    assert abs(ratios[-1] - 1 / (2 * SQRT2)) < 0.01


def test_exercise_2_arctan_threshold_and_damping():
    xc = arctan_critical_point()
    assert abs(xc - 1.3917) < 1e-4
    assert abs(arctan_iterates(xc, 1)[1] + xc) < 1e-9  # x1 = -x0
    assert abs(arctan_iterates(1.3)[-1]) < 1e-10
    assert abs(arctan_iterates(1.5)[-1]) > 1e3  # divergent
    damped = arctan_iterates(1.5, 12, 0.5)
    assert abs(damped[-1]) < 1e-4
    tail = np.abs(damped[-6:])
    assert np.allclose(tail[1:] / tail[:-1], 0.5, atol=0.01)  # linear, ratio 1/2


def test_exercise_3_double_root_error_halves():
    xs, errs = double_root_iterates()
    np.testing.assert_allclose(xs, [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375])
    np.testing.assert_allclose(np.array(errs[1:]) / np.array(errs[:-1]), 0.5)


def test_exercise_4_two_bus_angle():
    assert abs(angle_iterates(0.5)[-1] - np.arcsin(0.5)) < 1e-12
    hist = angle_iterates(0.99)
    assert abs(hist[-1] - np.arcsin(0.99)) < 1e-9 and len(hist) > 5
    div = angle_iterates(1.2, 12)
    assert len(div) == 13 and not np.allclose(div[-1], div[-2], atol=1e-3)  # no fixed point to settle on
    assert all(abs(np.sin(d) - 1.2) > 0.2 for d in div)  # the residual never gets small


def test_exercise_5_broyden_versus_newton():
    x0 = [1.0, -1.0]
    xn, hn = newton_history(x0)
    xb, hb = broyden_good(residual, x0, jacobian(np.asarray(x0)))
    np.testing.assert_allclose(xb, xn, atol=1e-9)
    assert len(hn) - 1 == 5 and len(hb) - 1 == 7
    assert abs(hb[1] - hn[1]) < 1e-12  # the first Broyden step is the Newton step (B0 = J(x0))
    _, hi = broyden_good(residual, x0, np.eye(2))
    assert len(hi) - 1 == 13 and hi[-1] < 1e-10
