import math

import numpy as np
from per_unit import BRANCHES, branch_table, change_base, i_base, p_limit_mw, z_base


def test_worked_example_numbers():
    np.testing.assert_allclose(p_limit_mw(1000.0, 380.0), 658.2, atol=0.1)
    np.testing.assert_allclose(z_base(220.0), 484.0)
    np.testing.assert_allclose(40 / z_base(220.0), 0.0826, atol=5e-5)
    np.testing.assert_allclose(i_base(220.0), 262.4, atol=0.1)
    np.testing.assert_allclose(change_base(0.10, 1.0, 50.0, 1.0, 100.0), 0.20)


def test_bases_are_consistent():
    # Z_base = V_base / (sqrt(3) I_base) with V in kV phase... i.e. V_LL² / S = V_LL / (sqrt(3) I_base) in kV/kA
    v, s = 380.0, 100.0
    np.testing.assert_allclose(z_base(v, s), 1e3 * v / (math.sqrt(3) * i_base(v, s)))
    # a pf below 1 lowers the MW limit in proportion
    np.testing.assert_allclose(p_limit_mw(1000.0, 380.0, 0.95), 0.95 * p_limit_mw(1000.0, 380.0))


def test_branch_table_lab_values():
    table = branch_table().set_index('name')
    np.testing.assert_allclose(table.z_base_ohm, [1444.0, 484.0, 121.0])
    np.testing.assert_allclose(table.x_pu, [0.30 * 150 / 1444, 0.40 * 100 / 484, 0.12 * 20 / 121])
    np.testing.assert_allclose(table.r_pu, [0.03 * 150 / 1444, 0.06 * 100 / 484, 0.05 * 20 / 121])
    np.testing.assert_allclose(table.p_limit_mw, [1316.4, 381.1, 114.3], atol=0.05)
    assert list(BRANCHES.columns) == ['name', 'v_kv', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km', 'i_max_a']


def test_toop_formula_matches_table():
    table = branch_table()
    toop = table.i_max_a * table.v_kv * 1e-3 * math.sqrt(3)  # value * limit_voltage * 1e-3 * sqrt(3)
    np.testing.assert_allclose(table.p_limit_mw, toop)


def test_exercise_4_change_of_base_is_independent_of_side():
    np.testing.assert_allclose(change_base(0.15, 1.0, 1600.0, 1.0, 100.0), 0.009375)
    # the same ohmic reactance referred to either side gives the same per-unit value
    x_hv_ohm = 0.15 * z_base(380.0, 1600.0)
    x_lv_ohm = x_hv_ohm / (380.0 / 110.0) ** 2
    np.testing.assert_allclose(x_hv_ohm / z_base(380.0, 100.0), 0.009375)
    np.testing.assert_allclose(x_lv_ohm / z_base(110.0, 100.0), 0.009375)
