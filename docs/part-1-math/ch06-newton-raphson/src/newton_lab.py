"""A reusable Newton solver with damping, basins of attraction and a SciPy comparison (Chapter 6 lab).

The system is the worked example of ``newton.py`` (x² + y² = 4, eˣ + y = 1); its two roots are ``ROOTS``.
"""

import numpy as np
from newton import jacobian, residual
from scipy import optimize

ROOT_A = np.array([1.004168, -1.729637])  # reached from (1, -1)
ROOT_B = np.array([-1.816264, 0.837368])  # reached from (-1, 1)
ROOTS = np.array([ROOT_A, ROOT_B])
MAX_HALVINGS = 10


# --8<-- [start:newton_solve]
def newton_solve(f, jac, x0, tol: float = 1e-10, max_iter: int = 50, damping: float = 1.0, backtrack: bool = False):
    """Newton's method for F(x) = 0; returns (x, success, residual norms).

    Each step solves J(x) Δ = −F(x) and sets x ← x + damping · Δ. With ``backtrack`` the fraction starts at
    ``damping`` and is halved (at most ten times) until ‖F‖ decreases. The history holds ‖F(x_k)‖ for
    k = 0, 1, ... including the starting point. A singular Jacobian (``np.linalg.LinAlgError``) or a non-finite
    iterate ends the run with ``success = False`` instead of an exception; so does reaching ``max_iter``
    above ``tol``.
    """
    x = np.asarray(x0, float)
    history = [float(np.linalg.norm(f(x)))]
    for _ in range(max_iter):
        if history[-1] < tol:
            return x, True, history
        try:
            step = np.linalg.solve(jac(x), -f(x))
        except np.linalg.LinAlgError:
            return x, False, history
        alpha = damping
        for _ in range(MAX_HALVINGS + 1):
            with np.errstate(over='ignore', invalid='ignore'):
                xn = x + alpha * step
                r = float(np.linalg.norm(f(xn)))
            if not backtrack or (np.isfinite(r) and r < history[-1]):
                break
            alpha /= 2
        if not np.isfinite(r):
            return xn, False, history
        x = xn
        history.append(r)
    return x, history[-1] < tol, history


# --8<-- [end:newton_solve]


# --8<-- [start:order]
def convergence_order(history) -> list[float]:
    """The ratios log r_{k+1} / log r_k for consecutive residual norms below 1 (about 2 for quadratic convergence)."""
    r = np.asarray(history, float)
    r = r[(r > 0) & (r < 1)]
    return [float(np.log(b) / np.log(a)) for a, b in zip(r[:-1], r[1:], strict=True)]


# --8<-- [end:order]


def _newton_step_grid(x, y):
    """Newton step (dx, dy) for every grid point, with the 2 × 2 Jacobian inverted by Cramer's rule."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        f1, f2 = x**2 + y**2 - 4, np.exp(x) + y - 1
        det = 2 * x - 2 * y * np.exp(x)  # determinant of J = [[2x, 2y], [eˣ, 1]]
        dx = -(f1 - 2 * y * f2) / det
        dy = -(2 * x * f2 - np.exp(x) * f1) / det
    return dx, dy


def _residual_grid(x, y):
    """‖F‖ for every grid point (inf where it overflows)."""
    with np.errstate(over='ignore', invalid='ignore'):
        return np.hypot(x**2 + y**2 - 4, np.exp(x) + y - 1)


# --8<-- [start:basins]
def basins(n: int = 200, damping: float = 1.0, backtrack: bool = False, max_iter: int = 50) -> np.ndarray:
    """Which root Newton reaches from each point of an n × n grid over [−3, 3]²: 0 or 1 for ``ROOTS``, −1 for failure.

    All grid points are iterated together, with the same step rule as ``newton_solve`` (fixed fraction
    ``damping``, or backtracking from it). A point whose Jacobian is singular or whose iterate leaves the
    floating-point range is marked as failed and frozen. Returns the label matrix indexed as
    ``label[row = y index, column = x index]`` for ``plt.imshow``.
    """
    s = np.linspace(-3, 3, n)
    x, y = (a.ravel().copy() for a in np.meshgrid(s, s))
    alive = np.ones(x.size, bool)
    r_old = _residual_grid(x, y)
    for _ in range(max_iter):
        dx, dy = _newton_step_grid(x, y)
        alpha = np.full(x.size, damping)
        xn, yn = x + alpha * dx, y + alpha * dy
        r_new = _residual_grid(xn, yn)
        if backtrack:
            for _ in range(MAX_HALVINGS):
                retry = ~(np.isfinite(r_new) & (r_new < r_old))
                if not retry.any():
                    break
                alpha = np.where(retry, alpha / 2, alpha)
                xn, yn = x + alpha * dx, y + alpha * dy
                r_new = _residual_grid(xn, yn)
        ok = alive & np.isfinite(r_new) & (np.abs(xn) < 1e6) & (np.abs(yn) < 1e6)
        alive &= ok
        x, y = np.where(ok, xn, x), np.where(ok, yn, y)
        r_old = np.where(ok, r_new, r_old)
    pts = np.column_stack([x, y])
    dist = np.linalg.norm(pts[:, None, :] - ROOTS[None, :, :], axis=2)
    label = np.where(dist.min(axis=1) < 1e-6, dist.argmin(axis=1), -1)
    label[~alive] = -1
    return label.reshape(n, n)


# --8<-- [end:basins]


def basin_counts(label: np.ndarray) -> dict[str, int]:
    """Number of grid points per outcome: 'A', 'B' and 'failed'."""
    return {'A': int((label == 0).sum()), 'B': int((label == 1).sum()), 'failed': int((label == -1).sum())}


def failures_near_singular_curve(label: np.ndarray, n: int = 200, band: float = 0.1) -> int:
    """How many failed grid points lie within ``band`` (vertically) of the curve det J = 0, i.e. y = x e⁻ˣ."""
    s = np.linspace(-3, 3, n)
    x, y = np.meshgrid(s, s)
    near = np.abs(y - x * np.exp(-x)) < band
    return int(((label == -1) & near).sum())


# --8<-- [start:scipy]
def scipy_comparison(x0=(1.0, -1.0)) -> dict[str, tuple[int | None, int]]:
    """(iterations, function evaluations) of Newton, SciPy ``hybr`` and SciPy ``broyden1`` from ``x0``.

    Newton evaluates F once per iteration plus once at the start. ``hybr`` (MINPACK) reports only ``nfev``,
    which includes the finite-difference Jacobian (two evaluations per rebuild), so its iteration count is
    ``None``; ``broyden1`` reports ``nit`` and ``nfev``.
    """
    _, _, hist = newton_solve(residual, jacobian, x0)
    out: dict[str, tuple[int | None, int]] = {'newton': (len(hist) - 1, len(hist))}
    for method in ('hybr', 'broyden1'):
        sol = optimize.root(residual, x0, method=method, tol=1e-10)
        out[method] = (int(sol.nit) if 'nit' in sol else None, int(sol.nfev))
    return out


# --8<-- [end:scipy]


def main() -> None:
    """Print the numbers quoted in the lab solution."""
    for x0 in ([1.0, -1.0], [-1.0, 1.0], [2.0, 2.0], [0.0, 0.0]):
        x, ok, hist = newton_solve(residual, jacobian, x0)
        print(f'x0 = {x0}: success = {ok}, {len(hist) - 1} steps, x = {np.round(x, 6)}')
        print('   residuals', ' '.join(f'{r:.1e}' for r in hist))
        print('   order    ', ' '.join(f'{p:.2f}' for p in convergence_order(hist)))
    plain = basins()
    print(
        'basins, full steps      :', basin_counts(plain), 'failed near det J = 0:', failures_near_singular_curve(plain)
    )
    print('basins, damping 0.5     :', basin_counts(basins(damping=0.5)))
    print('basins, backtracking    :', basin_counts(basins(backtrack=True)))
    for x0 in ([1.0, -1.0], [-1.0, 1.0]):
        n_fixed = len(newton_solve(residual, jacobian, x0, damping=0.5)[2]) - 1
        n_back = len(newton_solve(residual, jacobian, x0, backtrack=True)[2]) - 1
        print(f'from {x0}: damping 0.5 takes {n_fixed} steps, backtracking {n_back}')
    for method, (nit, nfev) in scipy_comparison().items():
        print(f'{method:9s} iterations = {nit!s:>4s}  function evaluations = {nfev:3d}')


if __name__ == '__main__':
    main()
