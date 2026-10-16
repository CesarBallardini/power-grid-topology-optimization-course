"""Error of the DC branch flow and independence of the slack (Chapter 16, Exercises 1 and 2)."""

import numpy as np
from dc_power_flow import dc_power_flow


def relative_error(vf: float, vt: float, dtheta: float) -> float:
    """(p_DC - p_AC) / p_AC for a lossless branch: |Vf||Vt| sin(d) / d is the AC flow per unit of DC flow."""
    return dtheta / (vf * vt * np.sin(dtheta)) - 1


def error_bounds(v_lo: float = 0.95, v_hi: float = 1.05, max_deg: float = 10.0) -> tuple[float, float]:
    """Smallest and largest relative error over |V| in [v_lo, v_hi] at both ends and 0 < |d| <= max_deg."""
    d = np.deg2rad(np.linspace(1e-6, max_deg, 1001))
    errors = [relative_error(vf, vt, d) for vf in (v_lo, v_hi) for vt in (v_lo, v_hi)]
    return float(np.min(errors)), float(np.max(errors))


def angles_and_flows(
    frm: np.ndarray, to: np.ndarray, x: np.ndarray, p: np.ndarray, slack: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bus angles (rad, slack at 0) and branch flows of a DC power flow, both for the chosen slack."""
    n, m = len(p), len(frm)
    a = np.zeros((m, n))
    a[np.arange(m), frm], a[np.arange(m), to] = 1.0, -1.0
    bbus = a.T @ np.diag(1 / x) @ a
    keep = np.arange(n) != slack
    theta = np.zeros(n)
    theta[keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], p[keep])
    return theta, dc_power_flow(frm, to, x, p, slack)


def main() -> None:
    """Exercise 1 bounds and the slack comparison of Exercise 2 on the four-bus grid of the tests."""
    lo, hi = error_bounds()
    print(f'Exercise 1: relative error of the DC flow between {100 * lo:+.2f} % and {100 * hi:+.2f} %')
    sinc = np.sin(np.deg2rad(10)) / np.deg2rad(10)
    print(f'  sinc(10 deg) = {sinc:.5f}; 1/0.95^2 - 1 = {1 / 0.95**2 - 1:+.4f}; 1/1.05^2 - 1 = {1 / 1.05**2 - 1:+.4f}')
    print(f'  angle alone at 10 deg (|V| = 1): {100 * relative_error(1, 1, np.deg2rad(10)):+.2f} %')
    frm, to = np.array([0, 0, 1, 2]), np.array([1, 2, 3, 3])
    x = np.array([0.1, 0.2, 0.1, 0.2])
    p = np.array([0.8, 0.2, -0.5, -0.5])  # pu, like the reactances
    th0, f0 = angles_and_flows(frm, to, x, p, slack=0)
    th3, f3 = angles_and_flows(frm, to, x, p, slack=3)
    print('Exercise 2: flows (pu) with slack 0:', np.round(f0, 4), ' with slack 3:', np.round(f3, 4))
    print('  angles (deg) slack 0:', np.round(np.rad2deg(th0), 3), ' slack 3:', np.round(np.rad2deg(th3), 3))
    print('  difference of the angle vectors (constant):', np.round(np.rad2deg(th3 - th0), 3))


if __name__ == '__main__':
    main()
