"""Sparse against dense linear solves on random symmetric matrices, timed with timeit (Chapter 30, Exercise 2)."""

import timeit
from functools import partial

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def random_symmetric(n: int, density: float = 0.02, seed: int = 0) -> sp.csc_matrix:
    """Random sparse symmetric matrix, made strictly diagonally dominant so that it is never singular.

    Args:
        n: Matrix size.
        density: Fraction of nonzero entries of the random part before symmetrization.
        seed: Seed of the NumPy generator.

    Returns:
        The matrix M + M.T + 2 n I in CSC format, the format spsolve prefers.
    """
    rng = np.random.default_rng(seed)
    m = sp.random(n, n, density=density, format='csc', random_state=rng)
    return (m + m.T + 2 * n * sp.identity(n, format='csc')).tocsc()


def dense_solve(a: sp.csc_matrix, b: np.ndarray) -> np.ndarray:
    """Solve a x = b with the dense LAPACK solver of NumPy, conversion to a dense array included."""
    return np.linalg.solve(a.toarray(), b)


def solve_both(a: sp.csc_matrix, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a x = b with the sparse LU of spsolve and with the dense solver."""
    return spsolve(a, b), dense_solve(a, b)


def timings(sizes: tuple[int, ...], density: float = 0.02, repeat: int = 3) -> list[tuple[int, float, float]]:
    """Best-of-repeat run time in seconds of spsolve and of numpy.linalg.solve for each size.

    The dense timing includes the conversion to a dense array, since that is what the dense route costs when the
    matrix arrives sparse. The sparse run time grows roughly with the fill-in of the LU factors; the dense one
    grows as n**3.
    """
    out = []
    for n in sizes:
        a = random_symmetric(n, density)
        b = np.ones(n)
        t_sparse = min(timeit.repeat(partial(spsolve, a, b), number=1, repeat=repeat))
        t_dense = min(timeit.repeat(partial(dense_solve, a, b), number=1, repeat=repeat))
        out.append((n, t_sparse, t_dense))
    return out


def main() -> None:
    """Check the two solutions agree at n = 200, then print the timing table for the plot."""
    a = random_symmetric(200)
    x_sparse, x_dense = solve_both(a, np.ones(200))
    print(
        f'n = 200: {a.nnz} nonzeros of {200 * 200}, max |x_sparse - x_dense| = {np.abs(x_sparse - x_dense).max():.2e}'
    )
    print(f'{"n":>6s} {"spsolve [ms]":>14s} {"dense solve [ms]":>18s} {"dense/sparse":>13s}')
    for n, t_sparse, t_dense in timings((200, 400, 800, 1600, 3200)):
        print(f'{n:6d} {1e3 * t_sparse:14.2f} {1e3 * t_dense:18.2f} {t_dense / t_sparse:13.1f}')


if __name__ == '__main__':
    main()
