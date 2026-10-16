"""Friedman test with Holm-corrected post-hoc comparisons, and the power of the rank-sum test (Chapter 26).

Exercise 2: three configurations on eight instances (the median of five seeds per instance is the paired
observation), Friedman over the three columns, then Wilcoxon signed-rank tests on the three pairs with the Holm
correction. Exercise 5: Monte Carlo estimate of the probability that the Mann-Whitney U test rejects at 5 % for
true median differences of 0, 1, 2 and 4 MW with a standard deviation of 3 MW and 5, 10 or 30 runs per
configuration.
"""

import numpy as np
import pandas as pd
from scipy import stats

CONFIGS = ('A', 'B', 'C')


def synthetic_instances(seed: int = 0, n_instances: int = 8, n_seeds: int = 5, advantage: float = 1.0) -> pd.DataFrame:
    """Final overload energies per instance (rows) and configuration (columns), median over ``n_seeds`` runs.

    Instances differ in difficulty (a random offset per row); configuration C is better by ``advantage`` MW.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.uniform(20.0, 60.0, size=n_instances)
    shift = {'A': 0.0, 'B': 0.0, 'C': -advantage}
    table = {
        name: np.median(rng.normal(difficulty[:, None] + shift[name], 3.0, size=(n_instances, n_seeds)), axis=1)
        for name in CONFIGS
    }
    return pd.DataFrame(table, index=[f'instance {i + 1}' for i in range(n_instances)])


def friedman(table: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """Friedman statistic, p-value and the mean rank of each configuration (rank 1 = best = lowest value)."""
    res = stats.friedmanchisquare(*(table[c].to_numpy() for c in table.columns))
    ranks = table.rank(axis=1).mean()
    return float(res.statistic), float(res.pvalue), ranks


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm step-down correction: sorted p-values compared with alpha / (m - i); adjusted p = max of running values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = [], 0.0
    for i, (_, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted.append(running)
    return pd.DataFrame(
        {
            'p': [p for _, p in items],
            'threshold': [alpha / (m - i) for i in range(m)],
            'p_holm': adjusted,
            'reject': [p_adj <= alpha for p_adj in adjusted],
        },
        index=[name for name, _ in items],
    )


def pairwise_signed_rank(table: pd.DataFrame) -> dict[str, float]:
    """Wilcoxon signed-rank p-value of every pair of columns (paired by instance)."""
    out = {}
    cols = list(table.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            out[f'{a} vs {b}'] = float(stats.wilcoxon(table[a], table[b]).pvalue)
    return out


def power_rank_sum(
    delta: float, n: int, sd: float = 3.0, n_rep: int = 2000, seed: int = 0, alpha: float = 0.05
) -> float:
    """Fraction of ``n_rep`` simulated experiments with p < alpha when the true medians differ by ``delta``."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd, size=(n_rep, n))
    b = rng.normal(delta, sd, size=(n_rep, n))
    p = stats.mannwhitneyu(a, b, alternative='two-sided', axis=1).pvalue
    return float((p < alpha).mean())


def power_table(deltas=(0.0, 1.0, 2.0, 4.0), sizes=(5, 10, 30), **kwargs) -> pd.DataFrame:
    """Power (rows: true difference in MW, columns: runs per configuration)."""
    return pd.DataFrame(
        {n: [power_rank_sum(d, n, seed=i * 100 + n, **kwargs) for i, d in enumerate(deltas)] for n in sizes},
        index=[f'{d:.0f} MW' for d in deltas],
    )


def main() -> None:
    """Exercise 2 on synthetic data with configuration C 1 MW better, and the power table of Exercise 5."""
    table = synthetic_instances()
    print(table.round(2).to_string())
    statistic, p, ranks = friedman(table)
    print(f'Friedman chi2 = {statistic:.2f}, p = {p:.4f}; mean ranks {ranks.round(2).to_dict()}')
    pairs = pairwise_signed_rank(table)
    print(holm(pairs).round(4).to_string())
    weak = synthetic_instances(advantage=0.3)
    statistic, p, _ = friedman(weak)
    rejections = int(holm(pairwise_signed_rank(weak))['reject'].sum())
    print(f'with a 0.3 MW advantage: Friedman p = {p:.4f}; Holm rejections: {rejections}')
    print(power_table().round(3).to_string())
    print(f'2 MW with 40 runs per configuration: power {power_rank_sum(2.0, 40, seed=7):.3f}')


if __name__ == '__main__':
    main()
