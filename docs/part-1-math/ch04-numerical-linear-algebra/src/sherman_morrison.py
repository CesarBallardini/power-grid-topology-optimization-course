"""Removing one branch from a resistor network with a rank-1 update of the inverse (Chapter 4)."""

import numpy as np

fr = np.array([0, 0, 1, 1, 2, 3, 4])
to = np.array([1, 2, 2, 3, 4, 4, 5])
g = np.array([10, 5, 8, 4, 6, 3, 2.0])
m, n = len(fr), 6


def reduced_incidence(fr: np.ndarray, to: np.ndarray, n: int, ground: int = 0) -> np.ndarray:
    """Incidence matrix without the column of the grounded node."""
    a = np.zeros((len(fr), n))
    a[np.arange(len(fr)), fr] = 1
    a[np.arange(len(fr)), to] = -1
    return a[:, np.arange(n) != ground]


def remove_branch(minv: np.ndarray, ar: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
    """Inverse of the reduced Laplacian after removing branch ``k``, by the Sherman-Morrison formula.

    ``minv`` is the inverse before the removal. Raises ``ZeroDivisionError`` when branch ``k`` is a bridge.
    """
    a = ar[k]
    u = minv @ a
    den = 1 - g[k] * (a @ u)
    if abs(den) < 1e-12:
        raise ZeroDivisionError(f'branch {k} is a bridge: removing it islands the network')
    return minv + g[k] * np.outer(u, u) / den  # Gr is symmetric, so Minv a a^T Minv = u u^T


def bridge_indicator(minv: np.ndarray, ar: np.ndarray, g: np.ndarray, k: int) -> float:
    """1 − gₖ R_eff,k: zero exactly when branch ``k`` is the only path between its ends."""
    return 1 - g[k] * (ar[k] @ minv @ ar[k])


def main() -> None:
    """Compare the rank-1 update with a fresh inverse for branch (1, 3), and detect the bridge (4, 5)."""
    ar = reduced_incidence(fr, to, n)  # ground node 0
    gr = ar.T @ np.diag(g) @ ar
    minv = np.linalg.inv(gr)
    k = 3  # branch (1, 3)
    minv_sm = remove_branch(minv, ar, g, k)
    keep = np.arange(m) != k
    gk = ar[keep].T @ np.diag(g[keep]) @ ar[keep]
    print(np.max(np.abs(minv_sm - np.linalg.inv(gk))))  # ~1e-16
    print(bridge_indicator(minv, ar, g, 6))  # bridge (4, 5): 0


if __name__ == '__main__':
    main()
