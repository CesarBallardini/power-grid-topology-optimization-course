import numpy as np
import pytest
from annealing import (
    TOY_N,
    anneal_case,
    count_neighbors,
    hill_climb,
    neighbors,
    simulated_annealing,
    toy_fitness,
    toy_runs,
)
from genome_mutation import normalize
from line_switching import load_case30


def test_neighbor_count_matches_the_formula():
    n = 41
    assert len(neighbors(normalize(np.array([0, 1, 2]), n), n)) == count_neighbors(n, 3, 3) == 117
    assert len(neighbors(normalize(np.array([0, 1, n]), n), n)) == count_neighbors(n, 3, 2) == 119
    assert len(neighbors(normalize(np.array([n, n, n]), n), n)) == count_neighbors(n, 3, 0) == 41


def test_toy_hill_climbing_is_stuck_and_annealing_escapes():
    start = normalize(np.array([0, 1]), TOY_N)
    stuck, evaluations = hill_climb(toy_fitness, start, TOY_N)
    assert toy_fitness(stuck) == 10.0 and evaluations == 1 + len(neighbors(start, TOY_N))
    rng = np.random.default_rng(3)
    _, best = simulated_annealing(toy_fitness, start, TOY_N, rng, budget=200, t0=10.0, alpha=0.97)
    assert best == 20.0
    assert toy_runs() == (10.0, 18)


def test_annealing_on_case30_is_seeded_and_reaches_the_optimum():
    case = load_case30()
    run = anneal_case(case, seed=2)
    assert run.evaluator.n_evaluations == 1152
    assert -run.best == pytest.approx(14.391, abs=1e-3) and run.evaluations_to_reach(-14.391 - 1e-3) == 174
