"""Thirty runs of two stochastic optimizer variants whose mean costs differ by 3 (Chapter 5, Exercise 5).

Each run returns a best cost C = μ + 10 Z with Z ~ N(0, 1): variant A has μ = 100, variant B has μ = 103. The
module summarizes both samples, draws their empirical CDFs and asks whether one run per variant, or thirty, can
tell them apart.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

MEAN_A, MEAN_B, SCALE, N_RUNS = 100.0, 103.0, 10.0, 30


def simulate_runs(seed: int = 0, n_runs: int = N_RUNS) -> pd.DataFrame:
    """Costs of ``n_runs`` runs of each variant, as columns ``A`` and ``B``."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {'A': MEAN_A + SCALE * rng.standard_normal(n_runs), 'B': MEAN_B + SCALE * rng.standard_normal(n_runs)}
    )


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, median and the 10 % and 90 % quantiles of each variant."""
    return pd.DataFrame(
        {
            'mean': runs.mean(),
            'median': runs.median(),
            'q10': runs.quantile(0.1),
            'q90': runs.quantile(0.9),
            'min': runs.min(),
            'max': runs.max(),
        }
    )


def probability_single_run_orders_correctly() -> float:
    """P(C_A < C_B) for one run of each: the difference is N(−3, 10√2), so Φ(3 / (10√2))."""
    return float(st.norm.cdf((MEAN_B - MEAN_A) / (SCALE * np.sqrt(2))))


def single_run_agreement(seed: int = 0, n_pairs: int = 100_000) -> float:
    """Simulated fraction of single-run pairs in which variant A looks better than B."""
    rng = np.random.default_rng(seed)
    a = MEAN_A + SCALE * rng.standard_normal(n_pairs)
    b = MEAN_B + SCALE * rng.standard_normal(n_pairs)
    return float((a < b).mean())


def welch_test(runs: pd.DataFrame) -> tuple[float, float]:
    """Welch's t statistic and p-value for equal means of the two variants."""
    result = st.ttest_ind(runs['A'], runs['B'], equal_var=False)
    return float(result.statistic), float(result.pvalue)


def runs_needed(power: float = 0.8, alpha: float = 0.05) -> float:
    """Runs per variant for a two-sided test of the 3-point difference to reach the given power.

    The usual normal approximation: n = 2 (z_{1−α/2} + z_power)² (σ / δ)² with σ = 10 and δ = 3.
    """
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return float(2 * z**2 * (SCALE / (MEAN_B - MEAN_A)) ** 2)


def plot_ecdfs(runs: pd.DataFrame) -> matplotlib.figure.Figure:
    """Empirical CDFs of both variants on one axis."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, colour in (('A', 'tab:blue'), ('B', 'tab:orange')):
        x = np.sort(runs[name].to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=colour, label=f'variant {name}')
    ax.set_xlabel('best cost of a run')
    ax.set_ylabel('fraction of runs at or below')
    ax.legend()
    return fig


def main() -> None:
    """Print the summary table, the single-run ordering probability and the Welch test for seed 0."""
    runs = simulate_runs()
    print(summary(runs).round(2).to_string())
    print(
        f'single run: P(A looks better) exact = {probability_single_run_orders_correctly():.4f}, '
        f'simulated = {single_run_agreement():.4f}'
    )
    t, p = welch_test(runs)
    print(f'Welch t-test on 30 runs each: t = {t:.2f}, p = {p:.3f}')
    print(f'standard error of the difference of two means of 30 runs: {SCALE * np.sqrt(2 / N_RUNS):.2f}')
    print(f'runs per variant for 80 % power at the 5 % level: {runs_needed():.0f}')
    for seed in range(1, 6):
        other = simulate_runs(seed)
        t, p = welch_test(other)
        print(f'  seed {seed}: mean A = {other["A"].mean():.2f}, mean B = {other["B"].mean():.2f}, p = {p:.3f}')


if __name__ == '__main__':
    main()
