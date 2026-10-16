"""Two GA configurations, ten seeds, one honest conclusion (Chapter 26, Lab steps 1 to 5).

The instance is the IEEE 30 line-switching problem of Chapter 24 and the optimizer its genetic algorithm (that
chapter's ``src`` folder is put on the import path). The two configurations differ in one component, crossover
(probability 0.7 against 0.0), the budget is a fixed number of fitness evaluations, and the performance measure is
the best N-1 overload energy at the end of the budget. Every run logs its best-so-far value after each evaluation
with the elapsed time, one row per (configuration, seed, evaluation).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_runs import compare

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ch24-evolutionary-algorithms' / 'src'))

from line_switching import load_case30  # noqa: E402
from switching_ga import BUDGET, genetic_algorithm  # noqa: E402

CONFIGURATIONS = {'crossover': {'p_crossover': 0.7}, 'no crossover': {'p_crossover': 0.0}}
SEEDS = tuple(range(10))
ALPHA = 0.05


def protocol() -> dict:
    """The pre-registered protocol: written down before any run, printed with the results."""
    return {
        'instance': 'IEEE 30 (MATPOWER case30), loads and generation x1.4, switch off at most 3 lines',
        'configurations': CONFIGURATIONS,
        'budget': f'{BUDGET} fitness evaluations per run (islanding candidates count)',
        'measure': 'best N-1 overload energy [MW] at the end of the budget (lower is better)',
        'seeds': list(SEEDS),
        'test': 'two-sided Mann-Whitney U on the 10 final values per configuration',
        'effect size': 'Vargha-Delaney A12 and a bootstrap 95 % confidence interval of the difference in medians',
        'significance level': ALPHA,
    }


def run_all(seeds=SEEDS, budget: int = BUDGET) -> pd.DataFrame:
    """One row per (configuration, seed, evaluation): best overload energy so far and elapsed seconds."""
    case = load_case30()
    frames = []
    for name, kwargs in CONFIGURATIONS.items():
        for seed in seeds:
            run = genetic_algorithm(case, seed, budget=budget, **kwargs)
            h = np.array(run.evaluator.history)
            frames.append(
                pd.DataFrame(
                    {
                        'configuration': name,
                        'seed': seed,
                        'evaluation': h[:, 0].astype(int),
                        'best': -h[:, 1],
                        'seconds': h[:, 2],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def final_values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Best value at the end of the budget, one array of seeds per configuration."""
    last = df.sort_values('evaluation').groupby(['configuration', 'seed'])['best'].last()
    return {name: last[name].to_numpy() for name in CONFIGURATIONS}


def anytime(df: pd.DataFrame, x: str = 'evaluation', n_points: int = 60) -> pd.DataFrame:
    """Median and 25-75 % quantiles of the best-so-far value against ``x`` (``evaluation`` or ``seconds``).

    Before a run's first finite value (every candidate islanded so far) it has no value; grid points where no run
    has a value yet are NaN.
    """
    grid = np.linspace(df[x].min(), df[x].max(), n_points)
    rows = []
    for name in CONFIGURATIONS:
        curves = []
        for _, run in df[df['configuration'] == name].groupby('seed'):
            finite = run[np.isfinite(run['best'])]
            curves.append(np.interp(grid, finite[x], finite['best'], left=np.nan))
        stacked = np.array(curves)
        q = np.full((3, n_points), np.nan)
        for j in range(n_points):
            column = stacked[:, j][np.isfinite(stacked[:, j])]
            if len(column):
                q[:, j] = np.percentile(column, [25, 50, 75])
        rows.append(pd.DataFrame({'configuration': name, x: grid, 'q25': q[0], 'median': q[1], 'q75': q[2]}))
    return pd.concat(rows, ignore_index=True)


def plot_anytime(df: pd.DataFrame, x: str = 'evaluation', ax=None) -> None:
    """Median line and quantile band per configuration."""
    ax = ax or plt.gca()
    for name, curve in anytime(df, x).groupby('configuration'):
        ax.plot(curve[x], curve['median'], label=name)
        ax.fill_between(curve[x], curve['q25'], curve['q75'], alpha=0.25)
    ax.set_xlabel(x)
    ax.set_ylabel('best N-1 overload energy [MW]')
    ax.set_yscale('log')
    ax.legend()


def bootstrap_median_difference(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile 95 % confidence interval of median(a) - median(b) from paired resampling of each sample."""
    rng = np.random.default_rng(seed)
    ra = rng.choice(a, size=(n_boot, len(a)), replace=True)
    rb = rng.choice(b, size=(n_boot, len(b)), replace=True)
    diff = np.median(ra, axis=1) - np.median(rb, axis=1)
    low, high = np.percentile(diff, [2.5, 97.5])
    return float(low), float(high)


def analysis(a: np.ndarray, b: np.ndarray) -> dict:
    """U, p, A12 (probability that a run of ``a`` beats one of ``b``) and the bootstrap interval of the medians."""
    u, p, a12 = compare(a, b)
    low, high = bootstrap_median_difference(a, b)
    return {
        'U': u,
        'p': p,
        'A12': a12,
        'median a': float(np.median(a)),
        'median b': float(np.median(b)),
        'CI': (low, high),
    }


def wall_clock_budget(df: pd.DataFrame, seconds: float, slow_seeds=(), slowdown: float = 3.0) -> dict[str, np.ndarray]:
    """Best value reached within ``seconds`` of wall-clock time per run.

    The runs of ``slow_seeds`` are taken to have shared the machine with a heavy process: their recorded times are
    multiplied by ``slowdown`` before the budget is applied (a simulation of the trap, reproducible; running a real
    CPU burner during half of the runs gives the same effect and a different number each time).
    """
    out = {}
    for name in CONFIGURATIONS:
        values = []
        for seed, run in df[df['configuration'] == name].groupby('seed'):
            factor = slowdown if seed in slow_seeds else 1.0
            within = run[run['seconds'] * factor <= seconds]
            values.append(within['best'].iloc[-1] if len(within) else np.nan)
        out[name] = np.array(values)
    return out


def evaluations_within(df: pd.DataFrame, seconds: float, slow_seeds=(), slowdown: float = 3.0) -> pd.Series:
    """Evaluations completed within the wall-clock budget, per (configuration, seed)."""
    rows = {}
    for (name, seed), run in df.groupby(['configuration', 'seed']):
        factor = slowdown if seed in slow_seeds else 1.0
        rows[(name, seed)] = int((run['seconds'] * factor <= seconds).sum())
    return pd.Series(rows)


def main() -> None:
    """Protocol, runs, final values, test, effect size, interval, and the wall-clock trap."""
    for key, value in protocol().items():
        print(f'{key}: {value}')
    df = run_all()
    finals = final_values(df)
    for name, values in finals.items():
        print(f'{name:13s} final E [MW]: {np.round(values, 3).tolist()}  median {np.median(values):.3f}')
    a, b = finals['crossover'], finals['no crossover']
    result = analysis(a, b)
    print(
        f'U = {result["U"]:.1f}, p = {result["p"]:.4f}, A12 = {result["A12"]:.2f}, '
        f'95 % CI of median difference ({result["CI"][0]:.3f}, {result["CI"][1]:.3f}) MW'
    )
    curve = anytime(df)
    for name, c in curve.groupby('configuration'):
        at = c.iloc[np.searchsorted(c['evaluation'], 300)]
        print(
            f'{name:13s} at {at["evaluation"]:.0f} evaluations: '
            f'median {at["median"]:.2f}, q25 {at["q25"]:.2f}, q75 {at["q75"]:.2f}'
        )
    seconds = df.groupby(['configuration', 'seed'])['seconds'].max()
    print(f'run time per {BUDGET} evaluations: {seconds.min():.2f} to {seconds.max():.2f} s (machine dependent)')
    budget = float(seconds.median() / 2)
    slow = tuple(SEEDS[::2])
    fair = wall_clock_budget(df, budget)
    trapped = wall_clock_budget(df, budget, slow_seeds=slow)
    for label, values in (
        ('wall-clock budget, quiet machine', fair),
        ('wall-clock budget, half the runs slowed x3', trapped),
    ):
        r = analysis(values['crossover'], values['no crossover'])
        print(
            f'{label} ({budget:.2f} s): medians {r["median a"]:.3f} vs {r["median b"]:.3f}, '
            f'p = {r["p"]:.4f}, A12 = {r["A12"]:.2f}'
        )
    n_eval = evaluations_within(df, budget, slow_seeds=slow)
    slowed = n_eval[[s in slow for _, s in n_eval.index]]
    others = n_eval[[s not in slow for _, s in n_eval.index]]
    print(f'evaluations within the budget: slowed runs {slowed.median():.0f}, others {others.median():.0f}')


if __name__ == '__main__':
    main()
