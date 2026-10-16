"""Bus splits on IEEE 14: BSDF against a recomputed PTDF, cross-coupler flows, reassignment and a split scan.

Lab steps 1 and 2 of Chapter 18 (``check_split``, ``reassign``), Exercise 3 (``beta_sum``) and Exercise 4
(``scan_station``: every admissible two-way split of one station, with the cross-coupler flow and the worst
post-split loading).
"""

from itertools import combinations

import numpy as np
from compose import moved_ends, split
from ieee14 import DcGrid, load_case14, ratings


def check_split(grid: DcGrid, station: int, bus_a: list[int]) -> dict[str, float]:
    """Largest differences between the BSDF update and the PTDF of the physically split grid (Lab step 1)."""
    s, p_ext = split(grid.ptdf(), grid.frm, grid.to, grid.b, grid.p, station, bus_a)
    ref = grid.split(station, bus_a)
    h_ref = ref.ptdf()
    nb = grid.n_bus
    return {
        'ptdf': float(np.abs(s.h - h_ref).max()),
        'flows': float(np.abs(s.f - h_ref @ p_ext).max()),
        'beta_vs_pedf': float(np.abs(s.beta - (h_ref[:, station] - h_ref[:, nb])).max()),
        'f_plus_c_beta_vs_h_p': float(np.abs(grid.flows() + s.c * s.beta - s.h @ p_ext).max()),
    }


def reassign(grid: DcGrid, station: int, bus_a: list[int], mw: float) -> dict[str, float]:
    """Move ``mw`` of injection from bus A to bus B after the split and compare delta F with -beta x (Lab step 2)."""
    s, p_ext = split(grid.ptdf(), grid.frm, grid.to, grid.b, grid.p, station, bus_a)
    nb = grid.n_bus
    p_moved = p_ext.copy()
    p_moved[station] -= mw
    p_moved[nb] += mw
    delta = s.h @ p_moved - s.f
    ref = grid.split(station, bus_a).moved(station, nb, mw).flows()
    return {
        'delta_f_vs_minus_beta_x': float(np.abs(delta + s.beta * mw).max()),
        'vs_recomputed': float(np.abs(s.h @ p_moved - ref).max()),
        'largest_change_mw': float(np.abs(delta).max()),
    }


def beta_sum(grid: DcGrid, station: int, bus_a: list[int]) -> float:
    """Exercise 3: sum of the bus-A branches' beta entries oriented out of bus A (expected: exactly one)."""
    s, _ = split(grid.ptdf(), grid.frm, grid.to, grid.b, grid.p, station, bus_a)
    sign = np.array([1.0 if grid.frm[k] == station else -1.0 for k in bus_a])
    return float(sign @ s.beta[bus_a])


def admissible_splits(branches: list[int], minimum: int = 2) -> list[list[int]]:
    """Two-way splits with at least ``minimum`` branches per busbar; a split and its mirror image count once."""
    first, rest = branches[0], branches[1:]
    out = []
    for r in range(minimum - 1, len(branches) - minimum):  # bus A holds r + 1 branches, bus B the rest
        out += [[first, *c] for c in combinations(rest, r)]  # the first branch is always on bus A
    return out


def scan_station(grid: DcGrid, station: int, limits: np.ndarray) -> list[tuple[list[str], float, float]]:
    """Exercise 4: (bus-A branches, |c| in MW, worst post-split loading in %) for every admissible split."""
    h = grid.ptdf()
    rows = []
    for bus_a in admissible_splits(grid.branches_at(station)):
        s, _ = split(h, grid.frm, grid.to, grid.b, grid.p, station, bus_a)
        rows.append(([grid.names[k] for k in bus_a], abs(s.c), float((100 * np.abs(s.f) / limits).max())))
    return rows


def main() -> None:
    """Split bus 2 of IEEE 14 (Lab steps 1-2, Exercise 3) and scan every split of bus 4 (Exercise 4)."""
    grid = load_case14()
    station = 1  # bus 2: generator 40 MW and load 21.7 MW; branches L1-2, L2-3, L2-4, L2-5
    bus_a = [grid.names.index('L1-2'), grid.names.index('L2-5')]
    print('bus 2 split, bus A = L1-2 + L2-5 (with the injections), bus B = L2-3 + L2-4')
    for key, value in check_split(grid, station, bus_a).items():
        print(f'  max |difference| {key:22s} {value:.2e}')
    s, _ = split(grid.ptdf(), grid.frm, grid.to, grid.b, grid.p, station, bus_a)
    print(f'  cross-coupler flow c = {s.c:.2f} MW from bus A to bus B')
    frm2, to2 = moved_ends(grid.frm, grid.to, station, bus_a, grid.n_bus)
    for k, name in enumerate(grid.names):
        if station in (frm2[k], to2[k]) or grid.n_bus in (frm2[k], to2[k]):
            print(f'  {name:6s} flow {grid.flows()[k]:8.2f} -> {s.f[k]:8.2f} MW, beta {s.beta[k]:+.4f}')
    for key, value in reassign(grid, station, bus_a, 40.0).items():
        print(f'  generator (40 MW) to bus B: {key:26s} {value:.4g}')
    print(f'  Exercise 3: sum of beta over bus-A branches, out of A = {beta_sum(grid, station, bus_a):.12f}')
    print('Exercise 4: every admissible split of bus 4 (L2-4, L3-4, L4-5, T4-7, T4-9), limits 1.3 x base, 40 MW floor')
    limits = ratings(grid)
    print(f'  unsplit worst loading {(100 * np.abs(grid.flows()) / limits).max():.1f} %')
    for names, c, loading in sorted(scan_station(grid, 3, limits), key=lambda r: r[1]):
        print(f'  bus A = {"+".join(names):22s} |c| = {c:6.1f} MW  worst loading {loading:6.1f} %')


if __name__ == '__main__':
    main()
