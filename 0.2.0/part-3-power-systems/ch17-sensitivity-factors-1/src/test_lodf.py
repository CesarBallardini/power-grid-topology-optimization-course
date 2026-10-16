import numpy as np
import pytest
from lodf import lodf_column, n1_flows, ptdf_matrix

FRM, TO = np.array([0, 0, 1, 1, 2]), np.array([1, 2, 2, 3, 3])
X = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
P = np.array([100.0, 0.0, 0.0, -100.0])


def test_ptdf_columns_sum_rule():
    h = ptdf_matrix(FRM, TO, X, 4)
    assert np.allclose(h[:, 0], 0)  # the slack column is zero
    f = h @ P
    a = np.zeros((5, 4))
    a[np.arange(5), FRM], a[np.arange(5), TO] = 1, -1
    np.testing.assert_allclose(a.T @ f, P, atol=1e-12)


@pytest.mark.parametrize('k', range(5))
def test_lodf_matches_resolving_without_the_branch(k):
    h = ptdf_matrix(FRM, TO, X, 4)
    keep = np.arange(5) != k
    f_ref = np.zeros(5)
    f_ref[keep] = ptdf_matrix(FRM[keep], TO[keep], X[keep], 4) @ P
    np.testing.assert_allclose(n1_flows(h, FRM, TO, P)[k], f_ref, atol=1e-9)


def test_bridge_raises():
    frm, to, x = np.array([0, 1]), np.array([1, 2]), np.array([0.1, 0.1])
    h = ptdf_matrix(frm, to, x, 3)
    with pytest.raises(ValueError, match='splits'):
        lodf_column(h, frm, to, 0)
