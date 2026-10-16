"""From behaviour descriptors to a MAP-Elites cell index (Chapter 25)."""

import numpy as np


def cell_index(split_subs: np.ndarray, switching_distance: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    """Flat cell index of each (split substations, switching distance) pair; values beyond ``dims`` are clipped."""
    return np.ravel_multi_index((split_subs, switching_distance), dims=dims, mode='clip')


def main() -> None:
    """Three topologies in a 5 x 45 archive."""
    print(cell_index(np.array([1, 7, 2]), np.array([60, 3, 44]), dims=(5, 45)))  # [ 89 183 134]


if __name__ == '__main__':
    main()
