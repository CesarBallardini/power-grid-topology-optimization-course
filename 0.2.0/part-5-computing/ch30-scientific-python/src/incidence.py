"""Branch-bus incidence matrix of a graph as a type-annotated NumPy function (Chapter 30, Exercise 1)."""

import numpy as np


def incidence_matrix(from_bus: np.ndarray, to_bus: np.ndarray, n_bus: int) -> np.ndarray:
    """Dense incidence matrix with one row per branch: +1 at the from bus, -1 at the to bus.

    Args:
        from_bus: Integer array with the from bus of every branch.
        to_bus: Integer array with the to bus of every branch, same length.
        n_bus: Number of buses, which fixes the number of columns.

    Returns:
        Float array of shape (n_branch, n_bus). Every row sums to zero because each branch leaves one bus and
        enters another; A.T @ A is the graph Laplacian (Chapter 2).

    Raises:
        ValueError: If the two bus arrays differ in length or a branch connects a bus to itself.
    """
    from_bus, to_bus = np.asarray(from_bus, dtype=int), np.asarray(to_bus, dtype=int)
    if from_bus.shape != to_bus.shape:
        raise ValueError('from_bus and to_bus must have the same length')
    if np.any(from_bus == to_bus):
        raise ValueError('a branch cannot start and end at the same bus')
    rows = np.arange(len(from_bus))
    a = np.zeros((len(from_bus), n_bus))
    a[rows, from_bus] = 1.0
    a[rows, to_bus] = -1.0  # fancy indexing: one assignment per branch, no Python loop
    return a


def main() -> None:
    """The four-bus, five-branch grid of the Chapter 31 worked example and its Laplacian."""
    a = incidence_matrix([0, 0, 1, 1, 2], [1, 2, 2, 3, 3], 4)
    print(a)
    print('row sums:', a.sum(axis=1))
    print('Laplacian A.T @ A:')
    print(a.T @ a)


if __name__ == '__main__':
    main()
