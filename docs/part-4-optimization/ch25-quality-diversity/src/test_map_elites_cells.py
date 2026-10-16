import numpy as np
from map_elites_cells import cell_index


def test_worked_example_and_clipping():
    cells = cell_index(np.array([1, 7, 2]), np.array([60, 3, 44]), dims=(5, 45))
    np.testing.assert_array_equal(cells, [89, 183, 134])
    assert cells[0] == 1 * 45 + 44  # 60 is clipped to the last column
    assert cells[1] == 4 * 45 + 3  # 7 is clipped to the last row
