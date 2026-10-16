"""PTDF matrix, LODF columns and the N-1 flow matrix of a DC grid (Chapter 17)."""

import numpy as np


def ptdf_matrix(frm: np.ndarray, to: np.ndarray, x: np.ndarray, n_bus: int, slack: int = 0) -> np.ndarray:
    """H (branches x buses): flow on every branch per unit injected at a bus and withdrawn at the slack."""
    m = len(frm)
    a = np.zeros((m, n_bus))
    a[np.arange(m), frm] = 1.0
    a[np.arange(m), to] = -1.0
    bf = a / x[:, None]
    bbus = a.T @ bf
    keep = np.arange(n_bus) != slack
    h = np.zeros_like(a)
    h[:, keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], bf[:, keep].T).T
    return h


def lodf_column(h: np.ndarray, frm: np.ndarray, to: np.ndarray, k: int) -> np.ndarray:
    """Flow change on every branch per MW that branch ``k`` carried before its outage (-1 on ``k`` itself)."""
    col = h[:, frm[k]] - h[:, to[k]]
    den = 1.0 - col[k]
    if abs(den) < 1e-11:  # bridge: islanding
        raise ValueError('outage splits the grid')
    lodf = col / den
    lodf[k] = -1.0
    return lodf


def n1_flows(h: np.ndarray, frm: np.ndarray, to: np.ndarray, p: np.ndarray) -> np.ndarray:
    """N-1 flow matrix (outage, branch): row ``k`` holds the flows after the outage of branch ``k``."""
    f0 = h @ p
    return np.array([f0 + lodf_column(h, frm, to, k) * f0[k] for k in range(len(frm))])


def main() -> None:
    """N-1 flows of a four-bus grid with one generator and one load."""
    frm, to = np.array([0, 0, 1, 1, 2]), np.array([1, 2, 2, 3, 3])
    x = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
    p = np.array([100.0, 0.0, 0.0, -100.0])
    h = ptdf_matrix(frm, to, x, 4)
    print(np.round(h @ p, 1))
    print(np.round(n1_flows(h, frm, to, p), 1))


if __name__ == '__main__':
    main()
