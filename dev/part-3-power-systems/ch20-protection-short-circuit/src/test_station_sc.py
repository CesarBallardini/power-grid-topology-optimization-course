import pytest
from fault_levels import worked_example
from station_sc import fault_levels, recoupling_angle, station, station_table, verdict


def test_hand_calculation_grid_reproduces_the_worked_example():
    table = station_table(extra_paths=False, r_ohm_per_km=0.0)
    hand = worked_example()
    assert table.loc[('max', 'closed'), 'S1'] == pytest.approx(hand['ik_closed_ka'], abs=0.02)
    assert table.loc[('max', 'closed'), 'S2'] == pytest.approx(hand['ik_closed_ka'], abs=0.02)
    assert table.loc[('max', 'open'), 'S1'] == pytest.approx(hand['ik_open_busbar1_ka'], abs=0.02)
    assert table.loc[('max', 'open'), 'S2'] == pytest.approx(hand['ik_open_busbar2_ka'], abs=0.02)


def test_extra_paths_raise_the_split_fault_levels_but_not_the_coupled_one():
    full = station_table()
    hand = station_table(extra_paths=False, r_ohm_per_km=0.0)
    assert full.loc[('max', 'closed'), 'S1'] == pytest.approx(hand.loc[('max', 'closed'), 'S1'], abs=0.05)
    assert full.loc[('max', 'open'), 'S1'] > hand.loc[('max', 'open'), 'S1'] + 2
    assert full.loc[('max', 'open'), 'S2'] > hand.loc[('max', 'open'), 'S2'] + 2
    assert (full.loc['min'] < full.loc['max']).all().all()


def test_peak_current_and_power_columns():
    table = fault_levels(station(), 'max')
    assert set(table.columns) == {'ikss_ka', 'skss_mw', 'ip_ka'}
    assert (table.ip_ka > table.ikss_ka * 1.4).all()  # kappa sqrt(2) > 1.4 for every bus
    assert table.loc['S1', 'skss_mw'] == pytest.approx(3**0.5 * 220 * table.loc['S1', 'ikss_ka'], rel=1e-6)


def test_recoupling_angle_is_zero_closed_and_small_open():
    closed, opened = recoupling_angle()
    assert closed == pytest.approx(0.0, abs=1e-9)
    assert 0.1 < opened < 1.0


def test_verdict_strings():
    assert verdict(7.0, 6.0, 8.0, 5.0) == 'acceptable'
    assert 'breaker' in verdict(9.0, 6.0, 8.0, 5.0)
    assert 'protection' in verdict(7.0, 4.0, 8.0, 5.0)
