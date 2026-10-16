import itertools

import numpy as np
import pytest
import woodbury as wb
from sherman_morrison import fr, g, n, reduced_incidence, remove_branch, to


@pytest.fixture(scope='module')
def network():
    ar = reduced_incidence(fr, to, n)
    return ar, np.linalg.inv(ar.T @ np.diag(g) @ ar)


def test_solve2x2_matches_numpy_and_flags_singular():
    a = np.array([[3.0, 1.0], [2.0, 5.0]])
    b = np.array([1.0, -2.0])
    x, ok = wb.solve2x2(a, b)
    assert ok and np.allclose(x, np.linalg.solve(a, b))
    x2, ok2 = wb.solve2x2(a, np.eye(2))
    assert ok2 and np.allclose(x2, np.linalg.inv(a))
    _, singular = wb.solve2x2(np.array([[1.0, 2.0], [2.0, 4.0]]), b)
    assert not singular


def test_woodbury_matches_fresh_inverse_for_every_valid_pair(network):
    ar, minv = network
    islanding = {(0, 1), (3, 4), (3, 5), (4, 5)}  # ground cut off; {3, 4, 5} cut off; node 3 alone; {4, 5} alone
    valid = [pair for pair in itertools.combinations(range(6), 2) if pair not in islanding]
    assert len(valid) == 11
    for j, k in valid:
        updated, ok = wb.remove_two_branches(minv, ar, g, j, k)
        assert ok
        np.testing.assert_allclose(updated, wb.fresh_inverse(ar, g, (j, k)), atol=1e-12)


def test_pairs_that_island_are_flagged(network):
    ar, minv = network
    for j, k in ((0, 1), (3, 4), (3, 5), (4, 5)):
        _, ok = wb.remove_two_branches(minv, ar, g, j, k)
        assert not ok
    for j in range(6):  # any pair with the bridge
        _, ok = wb.remove_two_branches(minv, ar, g, j, 6)
        assert not ok


def test_two_sequential_rank_one_updates_equal_one_rank_two_update(network):
    ar, minv = network
    sequential = remove_branch(remove_branch(minv, ar, g, 2), ar, g, 5)
    together, ok = wb.remove_two_branches(minv, ar, g, 2, 5)
    assert ok
    np.testing.assert_allclose(sequential, together, atol=1e-12)
