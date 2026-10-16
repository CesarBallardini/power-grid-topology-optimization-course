"""Six branches, one of them a bridge: which splits the bridge lookup removes (Chapter 22, Exercise 1)."""

import numpy as np
from action_count import assignments

D = 6
IS_BRIDGE = np.array([True, False, False, False, False, False])  # branch 0 is the bridge


def passes_bridge_lookup(assignment: np.ndarray, is_bridge: np.ndarray) -> bool:
    """A split is valid when every busbar in use keeps at least two non-bridge branches."""
    for on_busbar in (~assignment, assignment):
        if on_busbar.any() and int((on_busbar & ~is_bridge).sum()) < 2:
            return False
    return True


def describe(assignment: np.ndarray) -> str:
    """The branches moved to busbar B, as ``'b0+b3'``; ``'unsplit'`` when none."""
    moved = [f'b{i}' for i in np.flatnonzero(assignment)]
    return '+'.join(moved) if moved else 'unsplit'


def funnel(d: int = D, is_bridge: np.ndarray = IS_BRIDGE) -> tuple[int, int, int, list[str]]:
    """Counts after enumeration, after isolation removal and after the bridge lookup, plus the removed splits."""
    enumerated = 2 ** (d - 1)
    kept = assignments(d)
    survivors = np.array([passes_bridge_lookup(a, is_bridge) for a in kept])
    removed = [describe(a) for a in kept[~survivors]]
    return enumerated, len(kept), int(survivors.sum()), removed


def main() -> None:
    """The three counts and the list of assignments that the bridge lookup removes."""
    enumerated, after_isolation, after_bridge, removed = funnel()
    print(f'enumerated {enumerated}, after isolation removal {after_isolation}, after bridge lookup {after_bridge}')
    print('removed by the bridge lookup:', removed)


if __name__ == '__main__':
    main()
