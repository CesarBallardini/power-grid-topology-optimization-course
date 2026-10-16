"""DC power flow of a branch list: nodal susceptance matrix, angles and branch flows (Chapter 16)."""

import numpy as np


def dc_power_flow(frm: np.ndarray, to: np.ndarray, x: np.ndarray, p: np.ndarray, slack: int) -> np.ndarray:
    """Branch flows (from -> to, same unit as ``p``) for reactances ``x`` and nodal injections ``p``."""
    n, m = len(p), len(frm)
    a = np.zeros((m, n))
    a[np.arange(m), frm] = 1.0
    a[np.arange(m), to] = -1.0
    b = 1.0 / x
    bbus = a.T @ np.diag(b) @ a
    keep = np.arange(n) != slack
    theta = np.zeros(n)
    theta[keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], p[keep])
    return np.diag(b) @ a @ theta


def main() -> None:
    """Three-bus ring: 100 MW from bus 0 to bus 2 split in inverse proportion to the path reactances."""
    frm, to = np.array([0, 1, 0]), np.array([1, 2, 2])
    x = np.array([0.1, 0.1, 0.2])
    p = np.array([100.0, 0.0, -100.0])
    print(dc_power_flow(frm, to, x, p, slack=0))  # [50. 50. 50.]


if __name__ == '__main__':
    main()
