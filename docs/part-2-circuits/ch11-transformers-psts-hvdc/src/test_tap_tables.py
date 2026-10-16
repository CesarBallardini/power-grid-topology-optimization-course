import numpy as np
from tap_tables import TAPS, constant_x_table, growing_x_table, is_linear_pst, shift_deg


def test_linearity_test_like_get_linear_pst():
    assert is_linear_pst(constant_x_table())
    assert not is_linear_pst(growing_x_table())
    assert len(constant_x_table()) == 33 and TAPS[0] == -16 and TAPS[-1] == 16


def test_shift_is_odd_and_not_proportional():
    table = constant_x_table().set_index('tap')
    negative = table.alpha_deg.loc[-16:-1].to_numpy()  # taps -16 ... -1
    positive = table.alpha_deg.loc[1:16].to_numpy()  # taps 1 ... 16
    np.testing.assert_allclose(negative[::-1], -positive)
    np.testing.assert_allclose(table.alpha_deg[1], np.degrees(np.arctan(0.01)))
    assert table.alpha_deg[16] < 16 * table.alpha_deg[1]  # atan grows slower than its argument
    np.testing.assert_allclose(table.alpha_deg[16], 9.09, atol=0.005)
    np.testing.assert_allclose(shift_deg(np.array([5]), 0.01, 60.0), [2.42], atol=0.005)  # Chapter 1, Exercise 4


def test_growing_table_reactance():
    table = growing_x_table()
    np.testing.assert_allclose(table.x_pu[table.tap == 0], 0.1)
    np.testing.assert_allclose(table.x_pu[table.tap == 16], 0.116)
    np.testing.assert_allclose(table.x_pu[table.tap == -16], 0.116)
