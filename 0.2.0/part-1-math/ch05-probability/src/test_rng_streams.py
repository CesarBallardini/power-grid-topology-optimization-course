import numpy as np
import rng_streams as rs


def test_same_seed_replays_and_different_seed_diverges():
    assert np.array_equal(rs.numpy_draws(42), rs.numpy_draws(42))
    assert not np.array_equal(rs.numpy_draws(42), rs.numpy_draws(43))
    assert rs.numpy_draws(42).tolist() == [4, 3, 3, 2, 1, 5, 2, 0]


def test_spawned_children_are_reproducible_and_distinct():
    first, second = rs.spawned_draws(42), rs.spawned_draws(42)
    assert len(first) == 4
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)
    streams = [tuple(c) for c in first]
    assert len(set(streams)) == 4
    assert all(tuple(rs.numpy_draws(42)) != s for s in streams)  # children differ from the parent's own stream


def test_jax_split_is_deterministic_and_clipped():
    n, k, key = rs.jax_mutation_count(0)
    n2, k2, key2 = rs.jax_mutation_count(0)
    assert (n, k) == (n2, k2) == (0, 1)  # seed 0 happens to draw N = 0, which the lower clip turns into 1
    assert np.array_equal(np.asarray(key), np.asarray(key2))
    for seed in range(1, 20):
        n, k, _ = rs.jax_mutation_count(seed)
        assert 1 <= k <= 3 and k == min(max(n, 1), 3)
