import numpy as np
import pytest
from ieee14 import load_case14, ratings
from splits import admissible_splits, beta_sum, check_split, reassign, scan_station


@pytest.fixture(scope='module')
def grid():
    return load_case14()


BUS2_SPLIT = (1, ['L1-2', 'L2-5'])


def test_bsdf_on_bus_2_matches_the_physically_split_grid(grid):
    station, names = BUS2_SPLIT
    bus_a = [grid.names.index(n) for n in names]
    for value in check_split(grid, station, bus_a).values():
        assert value < 1e-9


def test_reassignment_changes_flows_by_minus_beta_x(grid):
    station, names = BUS2_SPLIT
    bus_a = [grid.names.index(n) for n in names]
    r = reassign(grid, station, bus_a, 40.0)
    assert r['delta_f_vs_minus_beta_x'] < 1e-9 and r['vs_recomputed'] < 1e-9 and r['largest_change_mw'] > 10


@pytest.mark.parametrize('station', [1, 3, 4, 5])
def test_bus_a_betas_sum_to_one(grid, station):
    for bus_a in admissible_splits(grid.branches_at(station)):
        assert beta_sum(grid, station, bus_a) == pytest.approx(1.0, abs=1e-9)


def test_admissible_splits_count_is_2_to_the_d_minus_1_minus_d_minus_1():
    for d in (4, 5, 6):
        assert len(admissible_splits(list(range(d)))) == 2 ** (d - 1) - d - 1


def test_scan_of_bus_4_has_ten_rows_and_a_wide_range(grid):
    rows = scan_station(grid, 3, ratings(grid))
    assert len(rows) == 10
    cs = np.array([c for _, c, _ in rows])
    loads = np.array([ld for _, _, ld in rows])
    assert cs.min() < 10 and cs.max() > 60 and loads.min() < 90 and loads.max() > 150
