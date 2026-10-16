"""Compare two optimizer configurations over repeated runs: Mann-Whitney U, A12 and ECDF plots (Chapter 26)."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Mann-Whitney U statistic, its two-sided p-value and the Vargha-Delaney A12 effect size.

    A12 is the probability that a random run of ``a`` scores lower (better) than a random run of ``b``.
    """
    res = stats.mannwhitneyu(a, b, alternative='two-sided')
    a12 = (np.sum(a[:, None] < b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (a.size * b.size)
    return float(res.statistic), float(res.pvalue), float(a12)


def plot_ecdf(samples: dict[str, np.ndarray], ax=None) -> None:
    """Empirical distribution of the final metric for each configuration."""
    ax = ax or plt.gca()
    for label, x in samples.items():
        stats.ecdf(x).cdf.plot(ax, label=label)
    ax.set_xlabel('final overload energy [MW]')
    ax.set_ylabel('fraction of runs')
    ax.legend()


def main() -> None:
    """Ten runs per configuration, synthetic data."""
    rng = np.random.default_rng(0)
    a = rng.normal(42.0, 3.0, size=10)  # final overload energy [MW] of 10 runs, configuration A
    b = rng.normal(39.0, 3.0, size=10)  # configuration B (lower is better)
    u, p, a12 = compare(a, b)
    print(f'U={u:.1f}  p={p:.4f}  A12={a12:.2f}')  # U=92.0  p=0.0017  A12=0.08
    plot_ecdf({'A': a, 'B': b})
    plt.show()


if __name__ == '__main__':
    main()
