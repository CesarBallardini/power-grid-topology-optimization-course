"""A fixed-length genome of substation actions with a sentinel for "no action", and one mutation (Chapter 24)."""

import numpy as np


def normalize(g: np.ndarray, n: int) -> np.ndarray:
    """Remove duplicates, sort, pad with the sentinel ``n`` (one past the last action index)."""
    u = np.unique(g[g < n])
    return np.concatenate([u, np.full(len(g) - len(u), n)])


def mutate(g: np.ndarray, n: int, rng: np.random.Generator, p_remove: float = 0.2) -> np.ndarray:
    """Overwrite one slot with the sentinel (probability ``p_remove``) or with a random action, then normalize."""
    g = g.copy()
    slot = rng.integers(len(g))
    g[slot] = n if rng.random() < p_remove else rng.integers(n)
    return normalize(g, n)


def main() -> None:
    """Normalize a genome with a duplicate and mutate it once."""
    rng = np.random.default_rng(seed=1)
    n = 41
    g = normalize(np.array([7, 7, 41]), n)  # -> [7, 41, 41]
    print(g, mutate(g, n, rng))


if __name__ == '__main__':
    main()
