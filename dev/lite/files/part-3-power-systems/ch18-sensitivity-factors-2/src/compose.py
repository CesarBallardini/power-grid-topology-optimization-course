"""Compose ToOp's low-rank updates on a DC grid: bus split, disconnection, N-1 screen and busbar outage (Chapter 18).

The functions take the arrays of an ``ieee14.DcGrid`` (or of ToOp's ``NetworkData``) and follow the order of
``compute_symmetric_batch``: bus splits, then disconnections, then the contingency factors on the fully updated
PTDF; on the flow side the cross-coupler flows, then the disconnection update, then the contingency flows.
"""

from dataclasses import dataclass, replace

import numpy as np
from bsdf import bsdf_split
from modf import ISLAND_TOL, apply_modf, lodf_matrix, modf, modf_flows


def coupler_row(h: np.ndarray, frm: np.ndarray, to: np.ndarray, i: int, bus_a: list[int]) -> np.ndarray:
    """The row p_c with c = p_c P: KCL at bus A, counting to-end branches positive (``i`` is not the slack)."""
    a_from = [k for k in bus_a if frm[k] == i]
    a_to = [k for k in bus_a if to[k] == i]
    p_c = h[a_to].sum(0) - h[a_from].sum(0)
    p_c[i] += 1.0
    return p_c


@dataclass(frozen=True)
class Split:
    """The outcome of one bus split: updated PTDF, flows, beta, the coupler row and the cross-coupler flow."""

    h: np.ndarray
    f: np.ndarray
    beta: np.ndarray
    p_c: np.ndarray
    c: float


def split(h: np.ndarray, frm: np.ndarray, to: np.ndarray, b: np.ndarray, p: np.ndarray, i: int, bus_a: list[int]):
    """Split node ``i``; the new node is appended as the last column of the PTDF and the last injection (zero).

    ``h`` is a PTDF with as many columns as ``p`` has entries. Returns a ``Split`` plus the extended injections.
    """
    nb = h.shape[1]
    h_ext = np.hstack([h, h[:, [i]]])
    p_ext = np.append(p, 0.0)
    p_c = coupler_row(h_ext, frm, to, i, bus_a)
    h_new, f_new = bsdf_split(h_ext, frm, to, b, i, nb, bus_a, p_ext)
    return Split(h=h_new, f=f_new, beta=h_new[:, i] - h_new[:, nb], p_c=p_c, c=float(p_c @ p_ext)), p_ext


def moved_ends(frm: np.ndarray, to: np.ndarray, i: int, bus_a: list[int], nb: int) -> tuple[np.ndarray, np.ndarray]:
    """From/to vectors after the split: the branches of ``i`` that are not in ``bus_a`` now end at ``nb``."""
    frm, to = frm.copy(), to.copy()
    for k in range(len(frm)):
        if k not in bus_a:
            if frm[k] == i:
                frm[k] = nb
            elif to[k] == i:
                to[k] = nb
    return frm, to


def n1_screen(h: np.ndarray, f: np.ndarray, frm: np.ndarray, to: np.ndarray, outages: list[int]) -> np.ndarray:
    """Post-contingency flows (outages x branches) from the LODF matrix; a bridge outage gives a NaN row."""
    lodf = lodf_matrix(h, frm, to)
    return f[None, :] + lodf[:, outages].T * f[outages][:, None]


def busbar_outage(h, f, frm, to, p, node) -> tuple[np.ndarray, list[int]]:
    """Flows after losing a busbar: its injection goes first (PTDF column), then all its branches (MODF).

    If the MODF of all branches islands the grid, retry without the first branch, which stays as a skeleton
    branch (ToOp's ``perform_outage_single_busbar``). Returns the flows and the branches actually outaged; NaN
    flows if both attempts fail.
    """
    at_node = [k for k in range(len(frm)) if frm[k] == node or to[k] == node]
    f_hat = f - h[:, node] * p[node]
    for branches in (at_node, at_node[1:]):
        if not branches:
            return f_hat, []
        if abs(modf(h, frm, to, branches)[1]) >= ISLAND_TOL:
            return modf_flows(f_hat, h, frm, to, branches), branches
    return np.full_like(f, np.nan), at_node


@dataclass(frozen=True)
class Composed:
    """Everything the composed computation produces, for comparison with the brute-force references."""

    h: np.ndarray
    f: np.ndarray
    n1: np.ndarray
    f_bb: np.ndarray
    beta: np.ndarray
    c: float


def compose(grid, station: int, bus_a: list[int], disconnect: int, busbar: int, split_first: bool = True) -> Composed:
    """Bus split, then a disconnection, then an N-1 screen with a busbar outage, in ToOp's order (or swapped).

    ``grid`` is an ``ieee14.DcGrid``. With ``split_first=False`` the disconnection is applied before the split,
    the alternative order of Lab step 5. The N-1 screen outages every branch except the disconnected one.
    """
    h, p, frm, to = grid.ptdf(), grid.p, grid.frm, grid.to
    nb = grid.n_bus
    if split_first:
        s, p = split(h, frm, to, grid.b, p, station, bus_a)
        frm, to = moved_ends(frm, to, station, bus_a, nb)
        h = apply_modf(s.h, frm, to, [disconnect])
        f = modf_flows(s.f, s.h, frm, to, [disconnect])
    else:
        h_d = apply_modf(h, frm, to, [disconnect])
        s, p = split(h_d, frm, to, grid.b, p, station, bus_a)
        frm, to = moved_ends(frm, to, station, bus_a, nb)
        h, f = s.h, s.f
        f[disconnect] = 0.0
    frm, to = frm.copy(), to.copy()
    frm[disconnect] = to[disconnect] = -1  # ToOp marks a disconnected branch with an invalid node index
    outages = [k for k in range(grid.n_branch) if k != disconnect]
    n1 = n1_screen(h, f, frm, to, outages)
    f_bb, _ = busbar_outage(h, f, frm, to, p, busbar)
    return Composed(h=h, f=f, n1=n1, f_bb=f_bb, beta=s.beta, c=s.c)


def brute_force(grid, station: int, bus_a: list[int], disconnect: int, busbar: int):
    """The same N-0, N-1 and busbar-outage flows from PTDFs recomputed on the physically modified grids."""
    base = grid.split(station, bus_a).without([disconnect])
    f = base.flows()
    n1 = [np.insert(base.without([k]).flows(), k, 0.0) for k in range(base.n_branch)]
    at_busbar = base.branches_at(busbar)
    p = base.p.copy()
    p[busbar] = 0.0
    f_bb = np.zeros(base.n_branch)
    keep = [k for k in range(base.n_branch) if k not in at_busbar]
    f_bb[keep] = replace(base.without(at_busbar), p=p).flows()
    return f, np.array(n1), f_bb


def main() -> None:
    """Lab steps 4 and 5 on IEEE 14: split bus 4, disconnect L6-13, screen N-1 with the outage of busbar 9."""
    from ieee14 import load_case14

    grid = load_case14()
    station = 3  # bus 4 (0-based 3): L2-4, L3-4, L4-5, T4-7, T4-9
    bus_a = [grid.names.index('L2-4'), grid.names.index('L4-5')]
    disconnect, busbar = grid.names.index('L6-13'), 8  # bus 9 (0-based 8): L9-10, L9-14, T4-9, T7-9
    c1 = compose(grid, station, bus_a, disconnect, busbar)
    f_ref, n1_ref, f_bb_ref = brute_force(grid, station, bus_a, disconnect, busbar)
    keep = [k for k in range(grid.n_branch) if k != disconnect]
    print('step 4, ToOp order (split, disconnect, screen) against brute force:')
    print('  N-0 flows      max diff', np.abs(c1.f[keep] - f_ref).max(), 'MW')
    print('  N-1 flows      max diff', np.nanmax(np.abs(c1.n1[:, keep] - n1_ref)), 'MW')
    print('  busbar outage  max diff', np.abs(c1.f_bb[keep] - f_bb_ref).max(), 'MW')
    worst = np.unravel_index(np.nanargmax(np.abs(c1.n1)), c1.n1.shape)
    print(
        f'  cross-coupler flow c = {c1.c:.2f} MW; worst N-1 flow {np.nanmax(np.abs(c1.n1)):.1f} MW on '
        f'{grid.names[worst[1]]} after {grid.names[keep[worst[0]]]} out'
    )
    c2 = compose(grid, station, bus_a, disconnect, busbar, split_first=False)
    print('step 5, disconnection before the split:')
    print('  final PTDF     max diff', np.abs(c1.h - c2.h).max())
    print('  final flows    max diff', np.abs(c1.f - c2.f).max(), 'MW')
    print('  N-1 flows      max diff', np.nanmax(np.abs(c1.n1 - c2.n1)), 'MW')
    print(f'  beta           max diff {np.abs(c1.beta - c2.beta).max():.4f}')
    print(f'  cross-coupler flow c: {c1.c:.2f} MW (split first) vs {c2.c:.2f} MW (disconnection first)')
    h, f = grid.ptdf(), grid.flows()
    for node in (8, 7):
        f_bb, branches = busbar_outage(h, f, grid.frm, grid.to, grid.p, node)
        skeleton = [grid.names[k] for k in grid.branches_at(node) if k not in branches]
        print(
            f'busbar outage of bus {node + 1} on the unsplit grid: outaged {[grid.names[k] for k in branches]}, '
            f'skeleton {skeleton} carrying {[round(float(f_bb[grid.names.index(s)]), 6) for s in skeleton]} MW'
        )


if __name__ == '__main__':
    main()
