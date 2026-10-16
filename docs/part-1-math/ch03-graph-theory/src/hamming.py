"""Hamming distance between two busbar-assignment vectors (Chapter 3, Exercise 5).

A topology of a station is a Boolean vector: ``True`` puts an asset on busbar A, ``False`` on busbar B. The labels
A and B are arbitrary, so a vector and its inverse describe the same electrical split; the switching distance
takes the smaller of the two distances, as ``per_station_switching_distance`` in
``dc_solver/preprocess/helpers/switching_distance.py`` does.
"""

import numpy as np


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Number of positions where the two Boolean vectors differ."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f'shapes differ: {a.shape} and {b.shape}')
    return int(np.count_nonzero(a != b))


def switching_distance(current: np.ndarray, target: np.ndarray) -> tuple[int, bool]:
    """Smallest Hamming distance to the target or to its inverse, and whether the inverse was the closer one."""
    plain = hamming_distance(current, target)
    inverted = hamming_distance(current, ~np.asarray(target, dtype=bool))
    return min(plain, inverted), inverted < plain


def main() -> None:
    """Print the distances quoted in the solution of Exercise 5."""
    current = np.array([True, True, False, False, True, False])
    target = np.array([False, False, True, True, False, False])
    print('plain Hamming distance   :', hamming_distance(current, target))
    print('to the inverted target   :', hamming_distance(current, ~target))
    print('switching distance, invert:', switching_distance(current, target))


if __name__ == '__main__':
    main()
