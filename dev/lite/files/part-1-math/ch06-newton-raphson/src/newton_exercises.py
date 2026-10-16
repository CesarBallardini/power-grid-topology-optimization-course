"""Scalar Newton iterations, the arctan divergence threshold, the two-bus angle equation and Broyden's method.

Numeric checks behind Exercises 1–5 of Chapter 6. The 2 × 2 system of Exercise 5 is the worked example of
``newton.py``.
"""

import numpy as np
from newton import jacobian, residual
from newton_lab import newton_solve
from scipy import optimize

SQRT2 = np.sqrt(2.0)


# --8<-- [start:scalar]
def newton_scalar(f, df, x0: float, steps: int = 3, damping: float = 1.0, stop: float = 0.0) -> list[float]:
    """The iterates x_0, x_1, ..., x_steps of x ← x − damping · f(x)/f'(x); stops early once |f(x)| ≤ ``stop``."""
    xs = [float(x0)]
    for _ in range(steps):
        x = xs[-1]
        if abs(f(x)) <= stop:
            break
        xs.append(x - damping * f(x) / df(x))
    return xs


# --8<-- [end:scalar]


def sqrt2_iterates(steps: int = 3) -> tuple[list[float], list[float]]:
    """Exercise 1: Newton on x² − 2 from 1, with the errors |x_k − √2|."""
    xs = newton_scalar(lambda x: x * x - 2, lambda x: 2 * x, 1.0, steps)
    return xs, [abs(x - SQRT2) for x in xs]


# --8<-- [start:arctan]
def arctan_critical_point() -> float:
    """Exercise 2: the x₀ > 0 for which Newton on arctan x gives x₁ = −x₀, i.e. 2x = (1 + x²) arctan x."""
    return float(optimize.brentq(lambda x: 2 * x - (1 + x * x) * np.arctan(x), 1.0, 2.0, xtol=1e-14))


def arctan_iterates(x0: float, steps: int = 6, damping: float = 1.0) -> list[float]:
    """Newton (optionally damped) on f(x) = arctan x, f'(x) = 1/(1 + x²), from ``x0``."""
    return newton_scalar(np.arctan, lambda x: 1 / (1 + x * x), x0, steps, damping)


# --8<-- [end:arctan]


# --8<-- [start:double_root]
def double_root_iterates(steps: int = 6) -> tuple[list[float], list[float]]:
    """Exercise 3: Newton on (x − 1)² from 0, with the errors |x_k − 1|; each error is half the previous one."""
    xs = newton_scalar(lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1), 0.0, steps)
    return xs, [abs(x - 1) for x in xs]


# --8<-- [end:double_root]


# --8<-- [start:angle]
def angle_iterates(px: float, steps: int = 12) -> list[float]:
    """Exercise 4: Newton on sin δ − PX from δ₀ = 0 (the lossless two-bus power flow, |V₁| = |V₂| = 1)."""
    return newton_scalar(lambda d: np.sin(d) - px, np.cos, 0.0, steps)


# --8<-- [end:angle]


# --8<-- [start:broyden]
def broyden_good(f, x0, j0, tol: float = 1e-10, max_iter: int = 50) -> tuple[np.ndarray, list[float]]:
    """Broyden's 'good' method with the inverse Jacobian kept by Sherman–Morrison; returns (x, residual norms).

    From B₀ = J(x₀) the step is Δ = −H F(x), H = B⁻¹. With s = Δ and y = F(x + Δ) − F(x) the good update is
    B⁺ = B + (y − B s) sᵀ / (sᵀ s), whose inverse (Sherman–Morrison) is H⁺ = H + (s − H y) sᵀ H / (sᵀ H y).
    The history holds ‖F(x_k)‖ for k = 0, 1, ...
    """
    x = np.asarray(x0, float)
    h = np.linalg.inv(np.asarray(j0, float))
    fx = f(x)
    history = [float(np.linalg.norm(fx))]
    for _ in range(max_iter):
        if history[-1] < tol:
            break
        s = -h @ fx
        x = x + s
        fx_new = f(x)
        y = fx_new - fx
        hy = h @ y
        h = h + np.outer(s - hy, s @ h) / (s @ hy)
        fx = fx_new
        history.append(float(np.linalg.norm(fx)))
    return x, history


# --8<-- [end:broyden]


def newton_history(x0, tol: float = 1e-10, max_iter: int = 50) -> tuple[np.ndarray, list[float]]:
    """Plain Newton on the worked example from ``x0`` (the comparison for Exercise 5); returns (x, residual norms)."""
    x = np.asarray(x0, float)
    history = [float(np.linalg.norm(residual(x)))]
    for _ in range(max_iter):
        if history[-1] < tol:
            break
        x = x + np.linalg.solve(jacobian(x), -residual(x))
        history.append(float(np.linalg.norm(residual(x))))
    return x, history


def main() -> None:
    """Print the numbers quoted in the solutions of Exercises 1–5."""
    xs, errs = sqrt2_iterates()
    print('Ex 1  iterates', [f'{x:.6f}' for x in xs], ' errors', [f'{e:.1e}' for e in errs])
    print('      error ratios e_{k+1}/e_k^2', [f'{b / a**2:.3f}' for a, b in zip(errs[:-1], errs[1:], strict=True)])
    xc = arctan_critical_point()
    print(f'Ex 2  critical x0 = {xc:.4f}; x1 from it = {arctan_iterates(xc, 1)[1]:.4f}')
    print('      from 1.3     ', [f'{x:.3g}' for x in arctan_iterates(1.3)])
    print('      from 1.5     ', [f'{x:.3g}' for x in arctan_iterates(1.5)])
    print('      from 1.5, damping 0.5', [f'{x:.3g}' for x in arctan_iterates(1.5, 12, 0.5)])
    _, ok, hist = newton_solve(np.arctan, lambda v: np.array([[1 / (1 + v[0] ** 2)]]), [1.5], backtrack=True)
    print(f'      from 1.5, backtracking: success = {ok} in {len(hist) - 1} steps')
    xs, errs = double_root_iterates()
    print('Ex 3  iterates', [f'{x:.5f}' for x in xs], ' errors', [f'{e:.5f}' for e in errs])
    for px in (0.5, 0.99, 1.2):
        d = angle_iterates(px)
        print(
            f'Ex 4  PX = {px}: ',
            ' '.join(f'{v:.4f}' for v in d),
            f' (asin = {np.degrees(np.arcsin(px)):.2f} deg)' if px <= 1 else '',
        )
    x0 = [1.0, -1.0]
    xn, hn = newton_history(x0)
    xb, hb = broyden_good(residual, x0, jacobian(np.asarray(x0)))
    print(
        f'Ex 5  Newton  {len(hn) - 1} iterations, x = {np.round(xn, 6)}, residuals', ' '.join(f'{r:.1e}' for r in hn)
    )
    print(
        f'      Broyden {len(hb) - 1} iterations, x = {np.round(xb, 6)}, residuals', ' '.join(f'{r:.1e}' for r in hb)
    )
    xb2, hb2 = broyden_good(residual, x0, np.eye(2))
    print(f'      Broyden from B0 = I: {len(hb2) - 1} iterations, x = {np.round(xb2, 6)}')


if __name__ == '__main__':
    main()
