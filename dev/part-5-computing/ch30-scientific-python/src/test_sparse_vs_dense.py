import numpy as np
from sparse_vs_dense import random_symmetric, solve_both, timings


def test_matrix_is_symmetric_and_sparse():
    a = random_symmetric(200)
    assert a.shape == (200, 200) and a.nnz < 0.1 * 200 * 200
    assert abs(a - a.T).max() == 0.0


def test_sparse_and_dense_solutions_agree():
    a = random_symmetric(300)
    b = np.arange(300, dtype=float)
    x_sparse, x_dense = solve_both(a, b)
    np.testing.assert_allclose(x_sparse, x_dense, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a @ x_sparse, b, rtol=1e-10, atol=1e-9)


def test_timings_returns_positive_times_per_size():
    rows = timings((100, 200), repeat=1)
    assert [n for n, _, _ in rows] == [100, 200]
    assert all(t_sparse > 0 and t_dense > 0 for _, t_sparse, t_dense in rows)
