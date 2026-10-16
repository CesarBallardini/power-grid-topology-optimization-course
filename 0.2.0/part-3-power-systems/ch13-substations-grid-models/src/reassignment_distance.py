"""Vectorized reassignment distance of the worked example's station, with and without the A/B swap (Chapter 13).

The station is the chain of four busbars and three couplers of the worked example, with six line bays: lines 0
and 1 on busbar 0, line 2 on busbar 1, line 3 on busbar 2, lines 4 and 5 on busbar 3. A candidate is a pair of
Boolean rows (bus A, bus B) over the assets; a target is one Boolean row, True meaning "on bus B". The distance of
a candidate is the number of asset-busbar connections that differ, counted on bus A against the complement of the
target and on bus B against the target; ToOp also tries the target with A and B swapped and keeps the smaller sum.
"""

import numpy as np
from coupler_splits import COUPLERS, two_node_configurations

N_BUSBARS = 4
ASSET_BUSBAR = np.array([0, 0, 1, 2, 3, 3])  # busbar of each of the six line bays
ASSET_NAMES = [f'L{i}' for i in range(len(ASSET_BUSBAR))]


def asset_table(asset_busbar: np.ndarray = ASSET_BUSBAR, n_busbars: int = N_BUSBARS) -> np.ndarray:
    """Boolean table of shape (busbars, assets): True where the asset's bay is closed onto that busbar."""
    return np.arange(n_busbars)[:, None] == asset_busbar[None, :]


def candidates(couplers: list[tuple[int, int]] = COUPLERS, asset_busbar: np.ndarray = ASSET_BUSBAR) -> np.ndarray:
    """All two-component coupler states as a Boolean array of shape (candidates, 2, assets).

    Row 0 of each candidate marks the assets whose busbar lies in the first component, row 1 those in the second.
    """
    n_busbars = int(asset_busbar.max()) + 1
    table = asset_table(asset_busbar, n_busbars)
    out = []
    for _, parts in two_node_configurations(couplers, n_busbars):
        out.append(np.stack([table[sorted(part)].any(axis=0) for part in parts]))
    return np.array(out)


def reassignment_distance(cands: np.ndarray, target: np.ndarray, allow_swap: bool = True) -> np.ndarray:
    """Reassignment distance of every candidate to a target, one vectorized expression.

    Args:
        cands: Boolean array (candidates, 2, assets), bus A first, bus B second.
        target: Boolean array (assets,), True for the assets that should end on bus B.
        allow_swap: Also compare with the roles of A and B exchanged and keep the smaller distance, as ToOp does.
    """
    direct = (cands[:, 0] ^ ~target).sum(axis=1) + (cands[:, 1] ^ target).sum(axis=1)
    if not allow_swap:
        return direct
    swapped = (cands[:, 0] ^ target).sum(axis=1) + (cands[:, 1] ^ ~target).sum(axis=1)
    return np.minimum(direct, swapped)


def describe(cand: np.ndarray) -> str:
    """The two busbar groups of a candidate as asset names, for printing."""
    a = ' '.join(n for n, on in zip(ASSET_NAMES, cand[0], strict=True) if on)
    b = ' '.join(n for n, on in zip(ASSET_NAMES, cand[1], strict=True) if on)
    return f'A = {{{a}}}, B = {{{b}}}'


def main() -> None:
    """Print the candidates, the distances to two targets, and the example where ignoring the swap overstates."""
    cands = candidates()
    print('candidates (bus A | bus B):')
    for k, cand in enumerate(cands):
        print(f'  {k}: {describe(cand)}')
    target = np.array([False, False, False, True, True, True])  # L3, L4, L5 on bus B
    print('target: L3 L4 L5 on bus B')
    print('  with swap   ', reassignment_distance(cands, target))
    print('  without swap', reassignment_distance(cands, target, allow_swap=False))
    mirror = ~target  # the same split, with the names of the two buses exchanged
    print('mirror target: L0 L1 L2 on bus B')
    print('  with swap   ', reassignment_distance(cands, mirror))
    print('  without swap', reassignment_distance(cands, mirror, allow_swap=False))


if __name__ == '__main__':
    main()
