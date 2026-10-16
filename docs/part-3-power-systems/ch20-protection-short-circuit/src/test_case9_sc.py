import pandapower.shortcircuit as sc
import pytest
from case9_sc import endtemp_effect, max_min_table, with_sc_data


def test_case9_has_the_short_circuit_columns_calc_sc_needs():
    net = with_sc_data()
    for col in ('s_sc_max_mva', 's_sc_min_mva', 'rx_max', 'rx_min'):
        assert col in net.ext_grid
    for col in ('vn_kv', 'sn_mva', 'xdss_pu', 'rdss_ohm', 'cos_phi'):
        assert col in net.gen
    assert 'endtemp_degree' in net.line
    assert (net.gen.vn_kv == 345.0).all()


def test_minimum_is_below_maximum_everywhere_and_most_at_the_feeder_bus():
    table = max_min_table()
    assert len(table) == 9
    assert (table['min'] < table['max']).all()
    assert table['min/max'].idxmin() == 0  # bus 1 sits at the external grid, where S''kQ and c change most
    assert table.loc[0, 'max'] == pytest.approx(9.087, abs=0.01)
    assert table.loc[0, 'min'] == pytest.approx(5.673, abs=0.01)


def test_hotter_lines_lower_the_minimum_current_a_little():
    s = endtemp_effect()
    assert s[20.0] > s[80.0] > s[160.0]
    assert (s[20.0] - s[160.0]) / s[20.0] < 0.02


def test_minimum_case_needs_the_end_temperature():
    net = with_sc_data()
    net.line = net.line.drop(columns='endtemp_degree')
    with pytest.raises(UserWarning, match='end temperature'):
        sc.calc_sc(net, case='min', branch_results=False)
