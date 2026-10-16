"""Two-bus power flow: Newton-Raphson by hand and the power transfer curve (Chapter 15, Exercises 1 and 2).

Slack bus 1 at 1 pu and zero angle, PQ bus 2 with load P + jQ behind a lossless line of reactance X. With
v = |V2| and d = theta_2, the injections at bus 2 are P2 = (v / X) sin d and Q2 = (v² - v cos d) / X, and the
specified values are -P and -Q (a load), so the solution has d < 0.
"""

import numpy as np


def injections_2bus(d: float, v: float, x: float) -> np.ndarray:
    """(P2, Q2) at the load bus for angle ``d`` (rad), magnitude ``v`` and line reactance ``x``."""
    return np.array([v * np.sin(d) / x, (v**2 - v * np.cos(d)) / x])


def jacobian_2bus(d: float, v: float, x: float) -> np.ndarray:
    """The 2 x 2 Jacobian [[dP2/dd, dP2/dv], [dQ2/dd, dQ2/dv]]."""
    return np.array([[v * np.cos(d) / x, np.sin(d) / x], [v * np.sin(d) / x, (2 * v - np.cos(d)) / x]])


def newton_steps(p: float, q: float, x: float, n_steps: int = 2) -> list[dict]:
    """The first ``n_steps`` Newton iterations from the flat start, with mismatch, Jacobian, step and new state."""
    d, v = 0.0, 1.0
    out = []
    for _ in range(n_steps):
        mismatch = np.array([-p, -q]) - injections_2bus(d, v, x)
        jac = jacobian_2bus(d, v, x)
        step = np.linalg.solve(jac, mismatch)
        d, v = d + step[0], v + step[1]
        out.append({'mismatch': mismatch, 'jacobian': jac, 'step': step, 'state': np.array([d, v])})
    return out


def exact_solution(p: float, q: float, x: float) -> tuple[float, float]:
    """Closed form (angle in rad, magnitude): the high-voltage root of v⁴ + (2QX - 1) v² + X²(P² + Q²) = 0."""
    b = 2 * q * x - 1
    c = x**2 * (p**2 + q**2)
    v = np.sqrt((-b + np.sqrt(b**2 - 4 * c)) / 2)
    return float(-np.arcsin(p * x / v)), float(v)


def transfer(v1: float, v2: float, x: float, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P = |V1||V2| sin(delta) / X and Q2 = (|V1||V2| cos(delta) - |V2|²) / X for the angles ``delta`` (rad)."""
    return v1 * v2 * np.sin(delta) / x, (v1 * v2 * np.cos(delta) - v2**2) / x


def main() -> None:
    """Exercise 1 (two iterations, exact solution) and Exercise 2 (maximum transfer, Jacobian at 90 degrees)."""
    x, p, q = 0.1, 1.0, 0.5
    for k, it in enumerate(newton_steps(p, q, x), start=1):
        print(
            f'iteration {k}: mismatch {np.round(it["mismatch"], 4)}, Jacobian {np.round(it["jacobian"], 4).tolist()}'
        )
        d, v = it['state']
        print(f'  step {np.round(it["step"], 4)} -> theta2 = {d:.4f} rad ({np.rad2deg(d):.3f} deg), |V2| = {v:.4f}')
    d, v = exact_solution(p, q, x)
    print(f'exact: theta2 = {d:.4f} rad ({np.rad2deg(d):.3f} deg), |V2| = {v:.4f}')
    more = newton_steps(p, q, x, 5)
    print('mismatch inf-norm per iteration:', ' '.join(f'{np.abs(it["mismatch"]).max():.1e}' for it in more))
    delta = np.deg2rad(np.linspace(0, 90, 7))
    pt, qt = transfer(1.0, 1.0, x, delta)
    print('delta (deg):', np.round(np.rad2deg(delta)).astype(int), ' P:', np.round(pt, 3), ' Q2:', np.round(qt, 3))
    for deg in (60, 85, 89, 90):
        print(f'dP/d(delta) at {deg} deg: {np.cos(np.deg2rad(deg)) / x:.4f} pu/rad')


if __name__ == '__main__':
    main()
