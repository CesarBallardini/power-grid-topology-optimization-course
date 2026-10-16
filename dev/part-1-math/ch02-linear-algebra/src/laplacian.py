"""Incidence matrix, weighted Laplacian and nodal solve of a small resistor network (Chapter 2)."""

import numpy as np

fr = np.array([0, 1, 2, 2])  # from nodes
to = np.array([1, 2, 0, 3])  # to nodes
g = np.array([1.0, 2.0, 1.0, 0.5])  # conductances in S


def incidence(fr: np.ndarray, to: np.ndarray, n: int) -> np.ndarray:
    """Branch-node incidence matrix A (branches x nodes): +1 at the from node, -1 at the to node."""
    a = np.zeros((len(fr), n))
    a[np.arange(len(fr)), fr] = 1.0
    a[np.arange(len(fr)), to] = -1.0
    return a


def laplacian(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Weighted Laplacian G = Aᵀ diag(g) A, the nodal conductance matrix."""
    return a.T @ np.diag(g) @ a  # same as np.einsum("ki,k,kj->ij", a, g, a)


def nodal_voltages(gmat: np.ndarray, i: np.ndarray, ground: int = 0) -> np.ndarray:
    """Solve G v = i with node ``ground`` held at 0 V (its row and column are dropped)."""
    keep = np.arange(len(i)) != ground
    v = np.zeros(len(i))
    v[keep] = np.linalg.solve(gmat[np.ix_(keep, keep)], i[keep])
    return v


def main() -> None:
    """Build G for the four-node example, show its null space and solve one injection pattern."""
    a = incidence(fr, to, 4)
    gmat = laplacian(a, g)
    w, vecs = np.linalg.eigh(gmat)
    print(np.round(w, 4))  # [0. 0.6106 3.091 5.2984]: one zero eigenvalue
    print(vecs[:, 0])  # +-[0.5 0.5 0.5 0.5]: the constant vector
    i = np.array([1.0, 0.0, 0.0, -1.0])  # inject 1 A at node 0, withdraw at node 3
    v = nodal_voltages(gmat, i)  # ground node 0
    print(v, g * (a @ v))  # node voltages and branch currents


if __name__ == '__main__':
    main()
