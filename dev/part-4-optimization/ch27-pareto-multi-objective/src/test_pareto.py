import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pareto import (
    POINTS,
    achievement_scalarizing,
    crowding_distance,
    dominates,
    epsilon_constraint,
    epsilon_indicator,
    fast_non_dominated_sort,
    hypervolume_2d,
    igd_plus,
    knee_points,
    non_dominated_mask,
    pseudo_weights,
    supported_mask,
    weighted_sum_argmin,
    weighted_sum_sweep,
)


def brute_force_ranks(points):
    """Peel fronts one at a time with the definition of dominance."""
    ranks = np.full(len(points), -1)
    rank = 0
    while (ranks < 0).any():
        remaining = np.flatnonzero(ranks < 0)
        for i in remaining:
            if not any(dominates(points[j], points[i]) for j in remaining if j != i):
                ranks[i] = rank
        rank += 1
    return ranks


def test_exercise_1_points():
    assert non_dominated_mask(POINTS).all()
    np.testing.assert_array_equal(supported_mask(POINTS), [True, True, False, True, True])
    assert weighted_sum_sweep(POINTS) == {0, 1, 3, 4}  # (4, 3.5) never wins
    assert epsilon_constraint(POINTS, 0, np.array([0, 3.6])) == 2  # but the eps-constraint reaches it
    assert achievement_scalarizing(POINTS, np.array([4.0, 3.0]), rho=0) == 2  # and so does a reference point near it


def test_zero_weight_gives_weakly_pareto_optimum():
    weak = np.array([[1, 2], [1, 3], [2, 1]], dtype=float)
    np.testing.assert_array_equal(weighted_sum_argmin(weak, np.array([1, 0])), [0, 1])
    assert not non_dominated_mask(weak)[1]
    assert weighted_sum_argmin(weak, np.array([0.9, 0.1])).tolist() == [0]


def test_sort_matches_brute_force_in_three_objectives():
    rng = np.random.default_rng(seed=3)
    points = rng.random((300, 3))
    ranks = fast_non_dominated_sort(points)
    np.testing.assert_array_equal(ranks, brute_force_ranks(points))
    np.testing.assert_array_equal(ranks == 0, non_dominated_mask(points))


def test_sort_handles_duplicates_and_many_points():
    points = np.array([[1, 1], [1, 1], [2, 2], [0, 3]], dtype=float)
    np.testing.assert_array_equal(fast_non_dominated_sort(points), [0, 0, 1, 0])
    cloud = np.random.default_rng(seed=27).random((1000, 3))
    ranks = fast_non_dominated_sort(cloud)
    assert ranks.min() == 0 and (ranks >= 0).all()


def test_crowding_distance():
    points = np.array([[0, 3], [1, 1], [2, 0.5], [3, 0]], dtype=float)
    ranks = np.zeros(4, dtype=int)
    d = crowding_distance(points, ranks)
    assert np.isinf(d[0]) and np.isinf(d[3])
    assert d[1] == pytest.approx(2 / 3 + 2.5 / 3)
    assert d[2] == pytest.approx(2 / 3 + 1 / 3)


def test_hypervolume_rectangles():
    front = np.array([[1, 3], [2, 2], [3, 1]], dtype=float)
    assert hypervolume_2d(front, np.array([4, 4])) == pytest.approx(1 * 1 + 1 * 2 + 1 * 3)
    assert hypervolume_2d(POINTS, POINTS.max(axis=0)) == pytest.approx(37.0)
    assert hypervolume_2d(POINTS, 2 * POINTS.max(axis=0)) == pytest.approx(262.0)
    assert hypervolume_2d(front, np.array([1, 1])) == 0.0


@settings(max_examples=50, deadline=None)
@given(st.lists(st.tuples(st.floats(0, 1), st.floats(0, 1)), min_size=1, max_size=15))
def test_hypervolume_is_monotone_in_the_reference_point(raw):
    front = np.array(raw)
    assert hypervolume_2d(front, np.array([1.1, 1.1])) <= hypervolume_2d(front, np.array([2.0, 2.0])) + 1e-12


def test_indicators():
    front = np.array([[1, 3], [2, 2], [3, 1]], dtype=float)
    assert igd_plus(front, front) == 0.0
    assert epsilon_indicator(front, front) == 0.0
    worse = front + 0.5
    assert igd_plus(worse, front) == pytest.approx(0.5 * np.sqrt(2))
    assert epsilon_indicator(worse, front) == pytest.approx(0.5)
    assert epsilon_indicator(front, worse) == pytest.approx(-0.5)


def test_knee_and_pseudo_weights():
    front = np.array([[0, 1], [0.1, 0.1], [1, 0]], dtype=float)
    np.testing.assert_array_equal(knee_points(front), [1])
    w = pseudo_weights(front)
    np.testing.assert_allclose(w.sum(axis=1), 1)
    np.testing.assert_allclose(w[0], [1, 0])
    np.testing.assert_allclose(w[2], [0, 1])
    assert knee_points(np.array([[0, 1], [0.5, 0.5], [1, 0]], dtype=float)).size == 0  # a straight front has no knee
