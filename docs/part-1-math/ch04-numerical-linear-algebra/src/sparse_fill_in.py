"""Fill-in of a sparse LU factorization with and without column reordering (Chapter 4, Exercise 5).

The reduced conductance matrix of a ``side × side`` grid network is factored with ``scipy.sparse.linalg.splu``
under the natural ordering, COLAMD and MMD_AT_PLUS_A, and the number of nonzeros of L + U is compared with that of the
matrix itself.
"""

import pandas as pd
import scipy.sparse.linalg as spla
from rank_one_lab import grid_network, sparse_reduced_laplacian


def fill_in_table(side: int = 40, seed: int = 0) -> pd.DataFrame:
    """Number of nonzeros of Gr and of its LU factors under three column orderings, with the fill ratio.

    NATURAL and COLAMD are the two the exercise asks for; MMD_AT_PLUS_A is SuperLU's ordering for matrices with a
    symmetric pattern, added for comparison with Octave's ``symamd``.
    """
    gr = sparse_reduced_laplacian(*grid_network(side, seed))
    rows = [{'ordering': 'matrix G_r', 'nnz': int(gr.nnz), 'fill_ratio': 1.0}]
    for spec in ('NATURAL', 'COLAMD', 'MMD_AT_PLUS_A'):
        lu = spla.splu(gr, permc_spec=spec)
        nnz = int(lu.L.nnz + lu.U.nnz - gr.shape[0])  # the unit diagonal of L is stored but carries no information
        rows.append({'ordering': spec, 'nnz': nnz, 'fill_ratio': nnz / gr.nnz})
    return pd.DataFrame(rows)


def main() -> None:
    """Print the fill-in table of the 40 × 40 grid quoted in the solution of Exercise 5."""
    print(fill_in_table().to_string(index=False, float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()
