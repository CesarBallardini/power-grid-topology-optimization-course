"""The nodal solver of the Chapter 8 lab with susceptances: the DC power flow used by Chapter 11.

B = Aᵀ diag(b) A is the nodal susceptance matrix; with the ground (slack) angle fixed at 0, B θ = p gives the
angles and the branch flows are b ⊙ (A θ). Conductances, voltages and currents of Chapter 8 become
susceptances, angles and flows; the algebra is the same.
"""

import numpy as np


def incidence(fr: np.ndarray, to: np.ndarray, n: int) -> np.ndarray:
    """Branch-node incidence matrix (branches x nodes): +1 at the from node, −1 at the to node."""
    a = np.zeros((len(fr), n))
    a[np.arange(len(fr)), fr] = 1.0
    a[np.arange(len(fr)), to] = -1.0
    return a


def nodal_solve(
    fr: np.ndarray, to: np.ndarray, b: np.ndarray, inj: np.ndarray, n: int, ground: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Node angles (``ground`` held at 0) and branch flows b ⊙ (A θ) for the injections ``inj`` (ground's is free)."""
    fr, to, b, inj = (np.asarray(v, dtype=float) for v in (fr, to, b, inj))
    a = incidence(fr.astype(int), to.astype(int), n)
    bmat = a.T @ np.diag(b) @ a
    keep = np.arange(n) != ground
    theta = np.zeros(n)
    theta[keep] = np.linalg.solve(bmat[np.ix_(keep, keep)], inj[keep])
    return theta, b * (a @ theta)


def main() -> None:
    """Solve the Chapter 8 worked example (two nodes and ground) as a check."""
    theta, flows = nodal_solve([1, 1, 2], [0, 2, 0], [1.0, 2.0, 4.0], [0.0, 3.0, 0.0], 3)
    print(np.round(theta, 4), np.round(flows, 4))  # [0. 1.2857 0.4286] and [1.2857 1.7143 1.7143]


if __name__ == '__main__':
    main()
