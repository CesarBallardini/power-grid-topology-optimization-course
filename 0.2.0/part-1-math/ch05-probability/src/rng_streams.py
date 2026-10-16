"""Seeds, reproducibility and independent streams, in NumPy and in JAX (Chapter 5, Lab 4).

A NumPy ``Generator`` seeded twice with the same integer replays the same draws; ``rng.spawn(k)`` derives k
independent child generators from one parent. JAX has no hidden state at all: a key is a value, and
``jax.random.split`` turns it into fresh keys, which is how ``mutate.py`` gets its ``substation_key`` and
``disconnection_key`` before drawing anything.
"""

import jax
import jax.numpy as jnp
import numpy as np

LAMBDA = 2.0
N_DRAWS = 8


def numpy_draws(seed: int, n_draws: int = N_DRAWS, lam: float = LAMBDA) -> np.ndarray:
    """Poisson draws from a fresh ``default_rng(seed)``."""
    return np.random.default_rng(seed).poisson(lam, size=n_draws)


def spawned_draws(seed: int, n_children: int = 4, n_draws: int = N_DRAWS, lam: float = LAMBDA) -> list[np.ndarray]:
    """Poisson draws from ``n_children`` independent child generators spawned from one seeded parent."""
    children = np.random.default_rng(seed).spawn(n_children)
    return [child.poisson(lam, size=n_draws) for child in children]


def jax_mutation_count(seed: int, lam: float = LAMBDA, max_num_splits: int = 3) -> tuple[int, int, jax.Array]:
    """The key split and the clipped Poisson draw of ``mutate.py``, from a seed.

    ``random_key, substation_key, disconnection_key = jax.random.split(random_key, 3)``; the Poisson count comes
    from ``substation_key`` and is clipped to [1, max_num_splits]; the returned ``random_key`` is the one to carry
    forward, never the two that were consumed.
    """
    random_key, substation_key, disconnection_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    n = jax.random.poisson(substation_key, lam=lam, shape=())
    k = jnp.clip(n, 1, max_num_splits)
    return int(n), int(k), random_key


def main() -> None:
    """Show reproducibility with equal seeds, divergence with different ones, spawned streams and the JAX split."""
    a, b, c = numpy_draws(42), numpy_draws(42), numpy_draws(43)
    print('seed 42 :', a)
    print('seed 42 :', b, ' identical:', bool(np.array_equal(a, b)))
    print('seed 43 :', c, ' identical to seed 42:', bool(np.array_equal(a, c)))
    for i, child in enumerate(spawned_draws(42)):
        print(f'child {i} :', child)
    n, k, _ = jax_mutation_count(0)
    print(f'JAX: PRNGKey(0) -> split(3) -> poisson(substation_key) = {n}, clipped to [1, 3] = {k}')
    n2, k2, _ = jax_mutation_count(0)
    print('same seed again:', n2, k2)


if __name__ == '__main__':
    main()
