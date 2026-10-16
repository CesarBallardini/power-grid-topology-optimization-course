"""How many busbar assignments ToOp keeps for a substation with d branches (Chapter 22)."""

import numpy as np


def assignments(d: int) -> np.ndarray:
    """Assignments of d branches to two busbars (one row each, True = busbar B), up to a swap, no branch alone."""
    r = np.arange(2 ** (d - 1))
    repo = ((r[:, None] >> np.arange(d)) & 1).astype(bool)
    flip = repo.sum(axis=1) > d / 2
    repo[flip] = ~repo[flip]
    on_b = repo.sum(axis=1)
    return repo[(on_b != 1) & (on_b != d - 1)]


def n_actions(d: int) -> int:
    """Count the distinct assignments, the unsplit configuration included."""
    return len(np.unique(assignments(d), axis=0))


def main() -> None:
    """Compare the enumeration with the closed form 2^(d-1) - d (the unsplit configuration included)."""
    print([(d, n_actions(d), 2 ** (d - 1) - d) for d in range(4, 9)])
    # [(4, 4, 4), (5, 11, 11), (6, 26, 26), (7, 57, 57), (8, 120, 120)]


if __name__ == '__main__':
    main()
