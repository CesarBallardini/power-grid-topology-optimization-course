import numpy as np
import pytest

pytestmark = pytest.mark.labs
pytest.importorskip('pymoo')

from pymoo_sorting_labs import compare, pymoo_crowding, pymoo_ranks  # noqa: E402


def test_exercise_3_matches_pymoo():
    result = compare(np.random.default_rng(27).random((1000, 3)))
    assert result == {
        'fronts': 20,
        'first_front': 36,
        'ranks_equal': True,
        'crowding_equal': True,
        'boundary_first_front': 5,
    }


def test_small_fronts():
    points = np.array([[1, 3], [2, 2], [3, 1], [2, 3], [3, 3]], dtype=float)
    ranks = pymoo_ranks(points)
    np.testing.assert_array_equal(ranks, [0, 0, 0, 1, 2])
    crowd = pymoo_crowding(points, ranks)
    assert np.isinf(crowd[[0, 2]]).all() and crowd[1] == pytest.approx(2.0)
