"""Exercise 2 of Chapter 18: the Woodbury susceptance update on IEEE 118 against recomputing the PTDF.

``check(grid, k)`` verifies the update for k simultaneous susceptance changes; ``timing(grid, ks)`` times both
routes. Timings depend on the machine and on the BLAS build, the crossover point less so.
"""

import timeit
from functools import partial

import numpy as np
import pandapower.networks as pn
from ieee14 import DcGrid, from_pandapower, ptdf
from modf import woodbury


def changed(grid: DcGrid, branches: list[int], alpha: list[float]) -> DcGrid:
    """The grid with the susceptances of ``branches`` multiplied by ``1 + alpha``."""
    b = grid.b.copy()
    b[branches] *= 1 + np.asarray(alpha)
    return DcGrid(frm=grid.frm, to=grid.to, b=b, p=grid.p, names=grid.names, slack=grid.slack)


def check(grid: DcGrid, k: int, seed: int = 0) -> float:
    """Largest difference between the Woodbury update and the recomputed PTDF for ``k`` random changes."""
    rng = np.random.default_rng(seed)
    branches = sorted(rng.choice(grid.n_branch, size=k, replace=False).tolist())
    alpha = rng.uniform(-0.5, 0.5, size=k).tolist()
    h_w = woodbury(grid.ptdf(), grid.frm, grid.to, branches, alpha)
    return float(np.abs(h_w - changed(grid, branches, alpha).ptdf()).max())


def timing(grid: DcGrid, ks: tuple[int, ...], number: int = 20) -> list[tuple[int, float, float]]:
    """(k, Woodbury milliseconds, recompute milliseconds) per k, best of five repeats of ``number`` runs."""
    h = grid.ptdf()
    rows = []
    for k in ks:
        branches = list(range(k))
        alpha = [0.2] * k
        update = partial(woodbury, h, grid.frm, grid.to, branches, alpha)
        g = changed(grid, branches, alpha)
        recompute = partial(ptdf, g.frm, g.to, g.b, g.n_bus, g.slack)
        t_w = min(timeit.repeat(update, number=number, repeat=5))
        t_r = min(timeit.repeat(recompute, number=number, repeat=5))
        rows.append((k, 1e3 * t_w / number, 1e3 * t_r / number))
    return rows


def main() -> None:
    """Verify k = 1, 2, 3 on IEEE 118 and time the two routes up to k = 150 of its 186 branches."""
    grid = from_pandapower(pn.case118())
    print(f'IEEE 118: {grid.n_bus} buses, {grid.n_branch} branches')
    for k in (1, 2, 3):
        print(f'  k = {k}: max |H_woodbury - H_recomputed| = {check(grid, k):.1e}')
    print('  k   Woodbury ms   recompute ms')
    for k, t_w, t_r in timing(grid, (1, 2, 3, 5, 10, 20, 50, 100, 150)):
        print(f'  {k:3d}   {t_w:8.3f}      {t_r:8.3f}   {"Woodbury" if t_w < t_r else "recompute"} wins')


if __name__ == '__main__':
    main()
