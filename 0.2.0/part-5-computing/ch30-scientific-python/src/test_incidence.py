import numpy as np
import pytest
from incidence import incidence_matrix

GRAPHS = [
    ([0, 1], [1, 2], 3),  # a path of three buses
    ([0, 1, 2], [1, 2, 0], 3),  # a triangle
    ([0, 0, 1, 1, 2], [1, 2, 2, 3, 3], 4),  # the four-bus grid of Chapter 31
]


@pytest.mark.parametrize(('from_bus', 'to_bus', 'n_bus'), GRAPHS)
def test_every_row_sums_to_zero(from_bus, to_bus, n_bus):
    a = incidence_matrix(from_bus, to_bus, n_bus)
    assert a.shape == (len(from_bus), n_bus)
    np.testing.assert_array_equal(a.sum(axis=1), np.zeros(len(from_bus)))


def test_triangle_laplacian():
    a = incidence_matrix([0, 1, 2], [1, 2, 0], 3)
    np.testing.assert_array_equal(a.T @ a, [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])


def test_bad_input_raises():
    with pytest.raises(ValueError, match='same length'):
        incidence_matrix([0, 1], [1], 2)
    with pytest.raises(ValueError, match='same bus'):
        incidence_matrix([0, 1], [1, 1], 2)
