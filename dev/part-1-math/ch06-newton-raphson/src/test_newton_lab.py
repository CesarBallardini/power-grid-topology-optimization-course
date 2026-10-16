import numpy as np
from newton import jacobian, residual
from newton_lab import (
    ROOT_A,
    ROOT_B,
    basin_counts,
    basins,
    convergence_order,
    failures_near_singular_curve,
    newton_solve,
    scipy_comparison,
)


def test_newton_solve_returns_solution_flag_and_history():
    x, ok, hist = newton_solve(residual, jacobian, [1.0, -1.0])
    assert ok and len(hist) == 6  # start plus five steps
    np.testing.assert_allclose(x, ROOT_A, atol=1e-6)
    assert np.allclose(hist[1:5], [8.6e-1, 4.0e-2, 1.0e-4, 8.4e-10], rtol=0.1)
    x, ok, _ = newton_solve(residual, jacobian, [-1.0, 1.0])
    assert ok
    np.testing.assert_allclose(x, ROOT_B, atol=1e-6)


def test_singular_jacobian_is_a_failure_not_an_exception():
    x, ok, hist = newton_solve(residual, jacobian, [0.0, 0.0])
    assert not ok and len(hist) == 1 and np.all(x == 0)


def test_max_iter_without_convergence_is_a_failure():
    _, ok, hist = newton_solve(residual, jacobian, [1.0, -1.0], max_iter=2)
    assert not ok and len(hist) == 3


def test_convergence_order_is_about_two_in_the_tail():
    _, _, hist = newton_solve(residual, jacobian, [1.0, -1.0])
    orders = convergence_order(hist)
    assert all(1.8 < p < 3.0 for p in orders[-2:])


def test_fixed_damping_converges_linearly_backtracking_keeps_quadratic():
    _, ok, half = newton_solve(residual, jacobian, [1.0, -1.0], damping=0.5)
    assert ok and 30 <= len(half) - 1 <= 40
    ratios = np.array(half[-10:]) / np.array(half[-11:-1])
    assert np.allclose(ratios, 0.5, atol=0.02)  # error halves each step
    _, ok, back = newton_solve(residual, jacobian, [1.0, -1.0], backtrack=True)
    assert ok and len(back) == 6


def test_basins_cover_the_grid_and_backtracking_removes_every_failure():
    plain = basins()
    counts = basin_counts(plain)
    assert plain.shape == (200, 200) and sum(counts.values()) == 40000
    assert counts == {'A': 12117, 'B': 27673, 'failed': 210}
    assert failures_near_singular_curve(plain) == 139
    assert basin_counts(basins(backtrack=True))['failed'] == 0
    # the two known starting points land where newton_solve says they do: (1, -1) -> A, (-1, 1) -> B
    s = np.linspace(-3, 3, 200)
    i_plus, i_minus = int(np.argmin(abs(s - 1))), int(np.argmin(abs(s + 1)))
    assert plain[i_minus, i_plus] == 0 and plain[i_plus, i_minus] == 1


def test_scipy_comparison_counts():
    out = scipy_comparison()
    assert out['newton'] == (5, 6)
    assert out['hybr'][0] is None and out['hybr'][1] == 13
    assert out['broyden1'] == (16, 27)
