import numpy as np
import pytest
from map_elites import (
    CELLS,
    N_POINTS,
    STEP,
    deceptive_fitness,
    evaluate,
    ga_1d,
    load_case30,
    map_elites,
    map_elites_1d,
    step,
)


@pytest.fixture(scope='module')
def case():
    return load_case30()


def test_descriptors_of_the_unswitched_grid_and_of_an_islanding_set(case):
    value, cell = evaluate(case, np.array([41, 41, 41]))
    assert value == pytest.approx(-145.351, abs=1e-3) and cell == (0, 2)  # 108.3 % N-0 loading: bin 100-110
    value, cell = evaluate(case, np.array([30, 32, 40]))
    assert value == pytest.approx(-14.391, abs=1e-3) and cell == (3, 0)  # 85.7 %: below 90
    assert evaluate(case, np.array([15, 41, 41])) == (-np.inf, None)


def test_archive_of_seed_zero_is_reproducible(case):
    archive = map_elites(case, seed=0)
    assert archive.n_evaluations == 1152 and archive.cells == CELLS
    assert archive.coverage == pytest.approx(0.70) and -archive.best == pytest.approx(14.391, abs=1e-3)
    assert archive.qd_score == pytest.approx(399.6, abs=0.1)
    assert archive.history[-1][1:] == (archive.coverage, archive.qd_score, archive.best)
    assert np.isnan(archive.table()[0, 0]) and archive.table()[0, 2] == pytest.approx(145.351, abs=1e-3)
    again = map_elites(case, seed=0)
    np.testing.assert_array_equal(again.fitness, archive.fitness)


def test_deceptive_landscape_separates_the_two_algorithms():
    assert deceptive_fitness(20) == pytest.approx(10.0) and deceptive_fitness(85) == pytest.approx(20.0)
    assert deceptive_fitness(70) < 1e-3
    rng = np.random.default_rng(0)
    assert all(0 <= step(x, rng) < N_POINTS for x in (0, 50, N_POINTS - 1))  # clipped at the ends
    assert all(0 < abs(step(50, rng) - 50) <= STEP for _ in range(100))
    assert [ga_1d(seed) for seed in range(5)] == [20] * 5
    assert [map_elites_1d(seed) for seed in range(5)] == [85] * 5
