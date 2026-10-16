import numpy as np
import pytest
from ac_jacobian import jacobian_blocks
from jacobian_timing import case_state, finite_difference_error, jacobian_blocks_sparse, time_both


@pytest.fixture(scope='module')
def ieee14():
    return case_state('case14')


def test_analytic_jacobian_matches_central_differences(ieee14):
    theta, vm, y = ieee14
    assert finite_difference_error(theta, vm, y.toarray()) < 1e-6


def test_sparse_blocks_equal_dense_blocks(ieee14):
    theta, vm, y = ieee14
    dense = jacobian_blocks(theta, vm, y.toarray())
    sparse = jacobian_blocks_sparse(theta, vm, y)
    for d, s in zip(dense, sparse, strict=True):
        np.testing.assert_allclose(s.toarray(), d, atol=1e-12)
        assert s.nnz <= y.nnz


def test_timing_returns_positive_milliseconds(ieee14):
    dense, sparse = time_both(*ieee14, number=2)
    assert dense > 0 and sparse > 0
