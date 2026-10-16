import numpy as np
import pytest
from hamming import hamming_distance, switching_distance


def test_distance_counts_differing_positions():
    a = np.array([True, True, False, False, True, False])
    b = np.array([False, False, True, True, False, False])
    assert hamming_distance(a, a) == 0
    assert hamming_distance(a, b) == 5
    assert hamming_distance(a, ~a) == 6


def test_inverted_vector_is_the_same_split():
    a = np.array([True, True, False, False, True, False])
    b = np.array([False, False, True, True, False, False])
    assert switching_distance(a, b) == (1, True)
    assert switching_distance(a, a) == (0, False)
    assert switching_distance(a, ~a) == (0, True)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match='shapes differ'):
        hamming_distance(np.ones(3, dtype=bool), np.ones(4, dtype=bool))


def test_distance_is_a_metric_on_random_vectors():
    rng = np.random.default_rng(3)
    x, y, z = rng.integers(0, 2, size=(3, 40)).astype(bool)
    assert hamming_distance(x, y) == hamming_distance(y, x)
    assert hamming_distance(x, z) <= hamming_distance(x, y) + hamming_distance(y, z)
