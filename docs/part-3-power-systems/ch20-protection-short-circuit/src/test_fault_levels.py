import numpy as np
import pytest
from fault_levels import (
    FIVE_BUS,
    Grid,
    feeder_impedance,
    ikss_ka,
    skss_mva,
    slg_current,
    split_comparison,
    three_phase_current,
    worked_example,
)


def test_worked_example_numbers_of_the_chapter():
    r = worked_example()
    assert r['zq1_ohm'] == pytest.approx(6.66, abs=0.01) and r['zq2_ohm'] == pytest.approx(8.87, abs=0.01)
    assert r['z1_ohm'] == pytest.approx(22.66, abs=0.01) and r['z2_ohm'] == pytest.approx(32.87, abs=0.01)
    assert r['zk_closed_ohm'] == pytest.approx(13.41, abs=0.01)
    assert r['ik_closed_ka'] == pytest.approx(10.4, abs=0.05)
    assert r['ik_open_busbar1_ka'] == pytest.approx(6.2, abs=0.05)
    assert r['ik_open_busbar2_ka'] == pytest.approx(4.3, abs=0.05)


def test_iec_formulas_are_consistent():
    zq = feeder_impedance(8000.0)
    ik = ikss_ka(zq)
    assert skss_mva(ik) == pytest.approx(8000.0, rel=1e-9)  # sqrt(3) Un c Un / (sqrt(3) Zq) = c Un^2 / Zq = S''kQ


def test_slg_exceeds_three_phase_only_when_z0_is_below_z1():
    e, z1 = 1.0, 0.2j
    for z0, expected in ((0.1j, 1.2), (0.2j, 1.0), (0.6j, 0.6)):
        ratio = abs(slg_current(e, z1, z1, z0)) / abs(three_phase_current(e, z1))
        assert ratio == pytest.approx(expected)


def test_zbus_diagonal_is_the_thevenin_impedance():
    g = Grid(branches=[(0, 1, 4.0)], sources={0: 1.0}, n_bus=2)
    np.testing.assert_allclose(np.diag(g.zbus()), [1.0, 5.0])
    assert g.fault_currents(110.0)[1] == pytest.approx(ikss_ka(5.0, 110.0))


def test_split_changes_the_split_bus_and_its_neighbour_most():
    before, after, change = split_comparison()
    assert len(before) == FIVE_BUS.n_bus and len(after) == FIVE_BUS.n_bus + 1
    assert np.all(after[: FIVE_BUS.n_bus] <= before + 1e-9)  # removing paths never raises a fault level
    assert set(np.flatnonzero(np.abs(change) > 10).tolist()) == {1, 2}
    assert abs(change[0]) < 1 and abs(change[4]) < 1  # the infeed buses barely notice
