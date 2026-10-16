import numpy as np
from glsk_sensitivity import ZONE_A_KEYS, compare, glsk_with_zone_a


def test_glsk_columns_still_sum_to_one():
    for keys in ZONE_A_KEYS.values():
        np.testing.assert_allclose(glsk_with_zone_a(keys).sum(axis=0), 1.0)


def test_reference_flows_do_not_depend_on_the_glsk_but_margins_do():
    table = compare()
    f_ref = table.f_ref.unstack('glsk')
    assert (f_ref.nunique(axis=1) == 1).all()
    ram = table.ram.unstack('glsk')
    assert (ram.nunique(axis=1) == 3).all()
    ptdf_ab = table.ptdf_ab.unstack('glsk')
    assert (ptdf_ab.nunique(axis=1) == 3).all()


def test_worked_example_column_is_unchanged():
    sub = compare().loc['(0.6, 0.4)']
    np.testing.assert_allclose(sub.ptdf_ab, [0.188, 0.412, 0.362, 0.138, 0.225], atol=5e-4)
    np.testing.assert_allclose(sub.ram, [99.4, 130.6, 218.1, 186.9, 221.2], atol=0.05)
