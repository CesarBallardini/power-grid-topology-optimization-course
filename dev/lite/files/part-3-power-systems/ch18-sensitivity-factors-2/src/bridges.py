"""Exercise 6 of Chapter 18: how often is 1 - h_kk exactly zero for a radial branch of a random meshed grid?

Each grid is a random spanning tree on ``n`` nodes with extra random branches, plus one leaf node hung on a single
radial branch k. Mathematically h_kk = 1 for a bridge, so 1 - h_kk = 0; in floating point the PTDF comes out of
an LU solve and the difference is round-off instead. The chapter lists ToOp's islanding tolerances; the exercise
asks which of them would notice this bridge.
"""

import numpy as np
from ieee14 import ptdf

TOLERANCES = {
    'LODF: |1 - h_kk| > 1e-11 means connected': 1e-11,
    'BSDF: |d| >= 1e-5': 1e-5,
    '2x2, 3x3 and LU solves: |det| > 1e-10': 1e-10,
    '1x1 solve_and_check_det: a != 0 (exact)': 0.0,
}


def random_grid(n: int, extra: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """From/to/susceptance arrays of a meshed grid on ``n`` nodes plus a leaf; returns the radial branch index."""
    frm, to = [], []
    for node in range(1, n):  # random spanning tree: each new node attaches to an earlier one
        frm.append(int(rng.integers(0, node)))
        to.append(node)
    while len(frm) < n - 1 + extra:  # extra meshing branches between distinct nodes
        i, j = rng.integers(0, n, size=2)
        if i != j:
            frm.append(int(i))
            to.append(int(j))
    frm.append(int(rng.integers(0, n)))  # the radial branch to the leaf node n
    to.append(n)
    b = rng.uniform(2.0, 20.0, size=len(frm))
    return np.array(frm), np.array(to), b, len(frm) - 1


def one_minus_hkk(frm: np.ndarray, to: np.ndarray, b: np.ndarray, k: int) -> float:
    """1 - h_kk of branch ``k``, with h_kk the flow on k when one MW is sent from its from-end to its to-end."""
    h = ptdf(frm, to, b, int(max(frm.max(), to.max())) + 1)
    return float(1.0 - (h[k, frm[k]] - h[k, to[k]]))


def survey(trials: int, n: int = 12, extra: int = 8, seed: int = 0) -> np.ndarray:
    """1 - h_kk of the radial branch for ``trials`` random grids."""
    rng = np.random.default_rng(seed)
    return np.array([one_minus_hkk(*random_grid(n, extra, rng)) for _ in range(trials)])


def main() -> None:
    """Survey 2 000 grids and say which tolerance catches the bridge in every one of them."""
    values = survey(2000)
    exact = np.mean(values == 0.0)
    print(f'{len(values)} random grids, 12 meshed nodes + 1 leaf: 1 - h_kk is exactly zero in {100 * exact:.1f} %')
    print(f'  |1 - h_kk| median {np.median(np.abs(values)):.1e}, max {np.abs(values).max():.1e}')
    for name, tol in TOLERANCES.items():
        caught = np.mean(np.abs(values) <= tol) if tol > 0 else exact
        print(f'  {name:45s} catches the bridge in {100 * caught:5.1f} % of the grids')


if __name__ == '__main__':
    main()
