"""PTDF and LODF of a three-bus triangle with equal reactances, slack at bus 3 (Chapter 17, Exercise 1).

Branches: 0 = 1-2, 1 = 1-3, 2 = 2-3 (buses numbered 1-3 in the statement, 0-2 in the code). With x = 1 on every
branch, one MW injected at bus 1 and withdrawn at bus 3 has the direct path 1-3 (x = 1) and the detour 1-2-3
(x = 2): two thirds take the direct path, one third the detour.
"""

import numpy as np
from lodf import lodf_column, ptdf_matrix

FRM, TO, X = np.array([0, 0, 1]), np.array([1, 2, 2]), np.ones(3)
NAMES = ('1-2', '1-3', '2-3')


def ptdf() -> np.ndarray:
    """PTDF with slack at bus 3 (index 2)."""
    return ptdf_matrix(FRM, TO, X, 3, slack=2)


def lodf() -> np.ndarray:
    """Full LODF matrix (branch, outage); no branch of a triangle is a bridge."""
    h = ptdf()
    return np.column_stack([lodf_column(h, FRM, TO, k) for k in range(3)])


def main() -> None:
    """Print both matrices and the current-divider check."""
    np.set_printoptions(precision=4, suppress=True)
    h = ptdf()
    print('PTDF (rows 1-2, 1-3, 2-3; columns bus 1, bus 2, bus 3):')
    print(h)
    print('LODF (rows: monitored branch; columns: outaged branch):')
    print(lodf())
    print('current divider after the outage of 1-2: its flow must go 1-3 then 3-2 in series,')
    print('  so LODF[1-3, 1-2] = 1 and LODF[2-3, 1-2] = -1')
    print('h_kk of every branch (1 MW from f_k to t_k with k in service):', np.diag(h[:, FRM] - h[:, TO]))
    p = np.array([1.0, 0.0, -1.0])
    f0 = h @ p
    print('base flows for 1 MW from bus 1 to bus 3:', f0, ' after the outage of 1-2:', f0 + lodf()[:, 0] * f0[0])


if __name__ == '__main__':
    main()
