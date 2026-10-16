import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from line_switching import load_case30  # noqa: E402
from switching_ga import Evaluator, crossover, genetic_algorithm, plot_convergence, repair, seed_table  # noqa: E402

OPTIMUM = -14.3915  # fitness of {L22-24, L24-25, L6-28}, the |D| <= 3 brute-force optimum, rounded down


@pytest.fixture(scope='module')
def case():
    return load_case30()


def test_crossover_samples_unique_lines_from_the_union():
    rng = np.random.default_rng(0)
    a, b, n = np.array([3, 9, 41]), np.array([9, 20, 30]), 41
    for _ in range(50):
        child = crossover(a, b, n, rng)
        lines = child[child < n]
        assert len(child) == 3 and len(np.unique(lines)) == len(lines)
        assert set(lines.tolist()) <= {3, 9, 20, 30}
    assert np.all(crossover(np.array([41, 41, 41]), np.array([41, 41, 41]), n, rng) == 41)


def test_repair_makes_the_grid_connected(case):
    rng = np.random.default_rng(1)
    fixed = repair(case, np.array([15, 32, 40]), rng)  # L12-13 islands bus 13
    assert 15 not in fixed and Evaluator(case)(fixed) > -np.inf


def test_seeded_runs_are_reproducible_and_counted(case):
    run = genetic_algorithm(case, seed=9, budget=300)
    again = genetic_algorithm(case, seed=9, budget=300)
    assert run.evaluator.n_evaluations == 300 and run.best == again.best
    np.testing.assert_array_equal(run.evaluator.best_genome, again.evaluator.best_genome)
    assert run.evaluations_to_reach(OPTIMUM) == 45  # seed 9 is the lucky one of the lab table


def test_seed_table_and_plot(case):
    runs = [genetic_algorithm(case, seed, budget=200) for seed in (0, 1)]
    table = seed_table(runs, OPTIMUM, case)
    assert list(table.index) == [0, 1] and 'not reached' in table['evals to optimum'].astype(str).tolist()
    fig, ax = plt.subplots()
    plot_convergence(runs, OPTIMUM, ax)
    assert len(ax.get_lines()) == 3
    plt.close(fig)
