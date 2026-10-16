"""The clipped Poisson count of mutated substations, exactly and by simulation (Chapter 5, Lab 1–3, Exercises 1–2).

ToOp draws N ~ Poisson(λ) and uses K = min(max(N, 1), m) (``jnp.clip(n, 1, max_num_splits)`` in
``optimizer/dc/genetic_functions/mutation/mutate.py``). The lower clip moves the mass of N = 0 onto K = 1, the
upper clip piles everything from N ≥ m onto K = m.
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

LAMBDA = 2.0
N_DRAWS = 10_000


def poisson_pmf(lam: float, k: int) -> float:
    """P(N = k) for N ~ Poisson(λ)."""
    return math.exp(-lam) * lam**k / math.factorial(k)


def clipped_pmf(lam: float, m: int) -> pd.Series:
    """Exact P(K = k) for k = 1, …, m, where K = min(max(N, 1), m) and N ~ Poisson(λ) (Exercise 1).

    P(K = 1) = e^(−λ)(1 + λ) absorbs N = 0; P(K = k) = P(N = k) for 1 < k < m; P(K = m) = P(N ≥ m). With m = 1 the
    single value has probability 1.
    """
    if m < 1:
        raise ValueError('m must be at least 1')
    if m == 1:
        return pd.Series({1: 1.0}, name='P(K=k)')
    probs = {1: poisson_pmf(lam, 0) + poisson_pmf(lam, 1)}
    for k in range(2, m):
        probs[k] = poisson_pmf(lam, k)
    probs[m] = 1 - sum(poisson_pmf(lam, j) for j in range(m))
    return pd.Series(probs, name='P(K=k)')


def clipped_mean(lam: float, m: int) -> float:
    """E[K] from the exact pmf."""
    pmf = clipped_pmf(lam, m)
    return float((pmf.index.to_numpy() * pmf.to_numpy()).sum())


def mean_of_max_with_one(lam: float) -> float:
    """E[max(N, 1)] = λ + e^(−λ): the lower clip alone adds exactly the probability of N = 0 (Exercise 2)."""
    return lam + math.exp(-lam)


def simulate(lam: float = LAMBDA, m: int = 3, n_draws: int = N_DRAWS, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draws of N ~ Poisson(λ) and of the clipped K = min(max(N, 1), m) (lab step 1)."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam, size=n_draws)
    return n, np.clip(n, 1, m)


def frequencies(values: np.ndarray) -> pd.Series:
    """Relative frequencies of the integer values, sorted by value (``pd.Series.value_counts``)."""
    return pd.Series(values).value_counts(normalize=True).sort_index()


def sample_mean_vs_m(
    lam: float = LAMBDA, ms: range = range(1, 7), n_draws: int = N_DRAWS, seed: int = 0
) -> pd.DataFrame:
    """Sample mean and exact mean of K for each upper clip m, with the same Poisson draws (lab step 3)."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam, size=n_draws)
    rows = [{'m': m, 'sample_mean': float(np.clip(n, 1, m).mean()), 'exact_mean': clipped_mean(lam, m)} for m in ms]
    return pd.DataFrame(rows)


def plot_frequencies(n: np.ndarray, k: np.ndarray, lam: float, m: int) -> matplotlib.figure.Figure:
    """Bar charts of the frequencies of N and K with the exact probabilities of K overlaid (lab steps 1–2)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fn, fk = frequencies(n), frequencies(k)
    axes[0].bar(fn.index, fn.to_numpy(), label='simulated N')
    axes[0].plot(fn.index, st.poisson.pmf(fn.index, lam), 'ko', label='Poisson pmf')
    axes[0].set_title(rf'$N \sim$ Poisson({lam:g})')
    axes[1].bar(fk.index, fk.to_numpy(), label='simulated K')
    exact = clipped_pmf(lam, m)
    axes[1].plot(exact.index, exact.to_numpy(), 'ko', label='exact P(K = k)')
    axes[1].set_title(rf'$K = \min(\max(N, 1), {m})$')
    for ax in axes:
        ax.set_xlabel('value')
        ax.legend()
    axes[0].set_ylabel('relative frequency')
    return fig


def plot_mean_vs_m(table: pd.DataFrame, lam: float) -> matplotlib.figure.Figure:
    """Sample mean of K against m with the horizontal line λ + e^(−λ) (lab step 3)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table['m'], table['sample_mean'], 'o-', label='sample mean of K')
    ax.plot(table['m'], table['exact_mean'], 'k.', label='exact E[K]')
    ax.axhline(mean_of_max_with_one(lam), color='tab:red', linestyle='--', label=r'$\lambda + e^{-\lambda}$')
    ax.axhline(lam, color='gray', linestyle=':', label=r'$\lambda$')
    ax.set_xlabel('upper clip m')
    ax.set_ylabel('E[K]')
    ax.legend()
    return fig


def main() -> None:
    """Print every number of lab steps 1–3 and Exercises 1–2 quoted in the solutions."""
    print('exact P(K = k), lambda = 2, m = 3:')
    print(clipped_pmf(LAMBDA, 3).round(4).to_string())
    print(f'E[K] = {clipped_mean(LAMBDA, 3):.4f}   E[max(N, 1)] = {mean_of_max_with_one(LAMBDA):.4f}')
    print(f'P(N = 0) = e^-2 = {poisson_pmf(LAMBDA, 0):.4f}')
    n, k = simulate()
    print('simulated frequencies of N:')
    print(frequencies(n).round(4).to_string())
    print('simulated frequencies of K:')
    print(frequencies(k).round(4).to_string())
    print(f'sample mean of N = {n.mean():.4f}, of K = {k.mean():.4f}')
    print(sample_mean_vs_m().round(4).to_string(index=False))


if __name__ == '__main__':
    main()
