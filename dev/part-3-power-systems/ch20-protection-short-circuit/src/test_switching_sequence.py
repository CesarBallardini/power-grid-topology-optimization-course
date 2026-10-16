import pytest
from switching_sequence import SEQUENCE, Station, run


def test_the_sequence_is_feasible_and_ends_split_on_bb2():
    station = Station.double_busbar()
    log = run(station)
    assert len(log) == len(SEQUENCE) == 8
    assert not station.busbars_coupled()
    assert station.devices['Q2'].closed and not station.devices['Q1'].closed and station.devices['CB'].closed
    parallel = [row['step'] for row in log if row['line on both busbars']]
    assert parallel == ['close CBC', 'close Q2', 'open Q1']  # the line sees both busbars only while coupled
    assert [row['device'] for row in log if row['step'].endswith('CBC')] == ['breaker', 'breaker']


def test_disconnectors_never_switch_load_current():
    log = run(Station.double_busbar())
    for row in log:
        if row['device'] == 'disconnector':
            assert row['allowed because'] in (
                'parallel path, no voltage across the disconnector',
                'one side dead, no current',
            )


def test_wrong_orders_are_refused():
    with pytest.raises(ValueError, match='close Q2'):
        run(Station.double_busbar(), [('close', 'Q2', ''), ('open', 'Q1', '')])
    with pytest.raises(ValueError, match='open Q1'):
        run(Station.double_busbar(), [('open', 'Q1', '')])
    station = Station.double_busbar()
    run(station, SEQUENCE[:3])  # coupler closed
    with pytest.raises(ValueError, match='open QC1'):
        run(station, [('open', 'QC1', '')])  # the coupler disconnector carries the coupler current


def test_breaker_first_then_disconnector_is_the_only_way_to_drop_a_bay():
    station = Station.double_busbar()
    run(station, [('open', 'CB', ''), ('open', 'Q1', ''), ('open', 'Q9', '')])  # bay dead after the breaker
    assert not any(d.closed for n, d in station.devices.items() if n in ('CB', 'Q1', 'Q9'))
