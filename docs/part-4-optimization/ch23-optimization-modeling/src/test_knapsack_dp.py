import itertools

import numpy as np
from knapsack_dp import knapsack_dp, timing, verify_certificate


def brute_force(values, weights, capacity):
    best = 0
    for picks in itertools.product((0, 1), repeat=len(values)):
        if sum(p * w for p, w in zip(picks, weights, strict=True)) <= capacity:
            best = max(best, sum(p * v for p, v in zip(picks, values, strict=True)))
    return best


def test_exercise_3_instance():
    value, chosen, cells = knapsack_dp([5, 4, 3], [2, 3, 1], 5)
    assert value == 9 and chosen == [0, 1] and cells == 18
    assert verify_certificate([5, 4, 3], [2, 3, 1], 5, chosen, 9)
    assert not verify_certificate([5, 4, 3], [2, 3, 1], 5, [0, 1, 2], 9)  # over capacity


def test_matches_brute_force_on_random_instances():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(1, 9))
        values = [int(v) for v in rng.integers(1, 30, n)]
        weights = [int(w) for w in rng.integers(1, 15, n)]
        capacity = int(rng.integers(1, 40))
        value, chosen, _ = knapsack_dp(values, weights, capacity)
        assert value == brute_force(values, weights, capacity)
        assert verify_certificate(values, weights, capacity, chosen, value)


def test_cells_grow_linearly_with_capacity():
    assert knapsack_dp([1], [1], 999)[2] == 1000
    rows = timing(n=20, capacities=(1_000, 10_000))
    assert rows[1][0] == 10 * rows[0][0]
