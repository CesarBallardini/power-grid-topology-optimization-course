import numpy as np
import scipy.sparse.linalg as spla
from rank_one_lab import grid_network, sparse_reduced_laplacian
from sparse_fill_in import fill_in_table


def test_fill_in_of_the_40_by_40_grid():
    table = fill_in_table(40).set_index('ordering')
    assert table.loc['matrix G_r', 'nnz'] == 7835  # 1599 diagonal + 2 * 3118 off-diagonal entries
    assert table.loc['NATURAL', 'nnz'] == 126319
    assert table.loc['COLAMD', 'nnz'] == 63291
    assert table.loc['COLAMD', 'fill_ratio'] < table.loc['NATURAL', 'fill_ratio'] / 1.9
    assert table.loc['MMD_AT_PLUS_A', 'nnz'] < table.loc['COLAMD', 'nnz']


def test_factors_reproduce_the_matrix():
    gr = sparse_reduced_laplacian(*grid_network(10))
    for spec in ('NATURAL', 'COLAMD'):
        lu = spla.splu(gr, permc_spec=spec)
        pr = spla.splu(gr, permc_spec=spec)  # a second factorization is identical: no randomness in the ordering
        assert lu.L.nnz == pr.L.nnz
        x = lu.solve(gr @ np.ones(gr.shape[0]))
        assert abs(x - 1).max() < 1e-10
