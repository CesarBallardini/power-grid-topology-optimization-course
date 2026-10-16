import numpy as np
import pytest
from sherman_morrison import bridge_indicator, fr, g, m, n, reduced_incidence, remove_branch, to


@pytest.fixture
def network():
    ar = reduced_incidence(fr, to, n)
    return ar, np.linalg.inv(ar.T @ np.diag(g) @ ar)


@pytest.mark.parametrize('k', [0, 1, 2, 3, 4, 5])
def test_rank_one_update_matches_fresh_inverse(network, k):
    ar, minv = network
    keep = np.arange(m) != k
    fresh = np.linalg.inv(ar[keep].T @ np.diag(g[keep]) @ ar[keep])
    np.testing.assert_allclose(remove_branch(minv, ar, g, k), fresh, atol=1e-12)


def test_bridge_is_detected(network):
    ar, minv = network
    assert abs(bridge_indicator(minv, ar, g, 6)) < 1e-12  # branch (4, 5) is the only path to node 5
    assert bridge_indicator(minv, ar, g, 3) > 0.1
    with pytest.raises(ZeroDivisionError):
        remove_branch(minv, ar, g, 6)
