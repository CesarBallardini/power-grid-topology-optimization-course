"""Newton-Raphson for a system of two nonlinear equations (Chapter 6)."""

import numpy as np


def residual(x: np.ndarray) -> np.ndarray:
    """F(x, y) = (x² + y² − 4, eˣ + y − 1)."""
    return np.array([x[0] ** 2 + x[1] ** 2 - 4, np.exp(x[0]) + x[1] - 1])


def jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobian of ``residual``."""
    return np.array([[2 * x[0], 2 * x[1]], [np.exp(x[0]), 1.0]])


def newton(f, jac, x0, tol: float = 1e-12, max_iter: int = 20) -> list[tuple[np.ndarray, float]]:
    """Iterate x ← x − J(x)⁻¹ F(x) by solving J Δ = −F; return the iterates with their residual norms.

    ``f`` and ``jac`` are functions of the vector x. Never forms the inverse: each step is one linear solve.
    Stops when the residual norm drops below ``tol``.
    """
    x = np.asarray(x0, float)
    history = []
    for _ in range(max_iter):
        x = x + np.linalg.solve(jac(x), -f(x))
        r = float(np.linalg.norm(f(x)))
        history.append((x.copy(), r))
        if r < tol:
            break
    return history


def main() -> None:
    """Print the iterates from (1, -1): the residual exponent roughly doubles each step."""
    for k, (x, r) in enumerate(newton(residual, jacobian, [1.0, -1.0])):
        print(k + 1, x, f'{r:.1e}')


if __name__ == '__main__':
    main()
