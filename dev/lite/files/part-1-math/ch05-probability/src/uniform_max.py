"""The maximum of n independent Uniform(0, 1) values: E[max] = n / (n + 1), checked by simulation (Chapter 5, Ex. 4).

The maximum of n uniforms has distribution function F(x) = xⁿ and density n xⁿ⁻¹ on [0, 1], so its mean is
n / (n + 1): the worst of many draws is almost always close to the top.
"""

import numpy as np
import pandas as pd

SIZES = (1, 10, 100)
N_REPS = 100_000


def exact_mean_of_max(n: int) -> float:
    """E[max of n uniforms] = n / (n + 1)."""
    return n / (n + 1)


def simulated_mean_of_max(n: int, n_reps: int = N_REPS, seed: int = 0) -> float:
    """Average of the maximum of n uniforms over ``n_reps`` repetitions."""
    rng = np.random.default_rng(seed)
    return float(rng.random((n_reps, n)).max(axis=1).mean())


def comparison(sizes: tuple[int, ...] = SIZES, n_reps: int = N_REPS, seed: int = 0) -> pd.DataFrame:
    """Exact and simulated E[max] for each n, with the standard error of the simulation."""
    rows = []
    for n in sizes:
        rng = np.random.default_rng(seed)
        maxima = rng.random((n_reps, n)).max(axis=1)
        rows.append(
            {
                'n': n,
                'exact': exact_mean_of_max(n),
                'simulated': float(maxima.mean()),
                'std_error': float(maxima.std(ddof=1) / np.sqrt(n_reps)),
                'P(max > 0.9)': float((maxima > 0.9).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Print the comparison table of Exercise 4."""
    print(comparison().to_string(index=False, float_format=lambda x: f'{x:.5f}'))


if __name__ == '__main__':
    main()
