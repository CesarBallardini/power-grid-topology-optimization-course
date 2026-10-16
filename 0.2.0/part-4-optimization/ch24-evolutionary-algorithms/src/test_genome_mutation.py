import numpy as np
from genome_mutation import mutate, normalize


def test_normalize_removes_duplicates_and_pads():
    np.testing.assert_array_equal(normalize(np.array([7, 7, 41]), 41), [7, 41, 41])
    np.testing.assert_array_equal(normalize(np.array([41, 3, 9, 3]), 41), [3, 9, 41, 41])


def test_mutation_keeps_the_genome_normalized():
    rng = np.random.default_rng(0)
    n, g = 41, normalize(np.array([7, 12, 41]), 41)
    for _ in range(200):
        g = mutate(g, n, rng)
        real = g[g < n]
        assert len(g) == 3 and np.all(np.diff(real) > 0) and np.all(g[len(real) :] == n)
