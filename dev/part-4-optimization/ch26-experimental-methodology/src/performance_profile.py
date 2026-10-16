"""Dolan-More performance profiles for three solvers over twenty instances (Chapter 26, Exercise 3).

For solver s and instance p, the performance ratio is r_ps = t_ps / min_s t_ps, with r_ps = +inf when the solver
failed to reach the target; the profile rho_s(tau) is the fraction of instances with r_ps <= tau. rho_s(1) is the
fraction of instances on which s was the fastest, and rho_s(tau) for large tau tends to the fraction of instances
the solver solved at all.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOLVERS = ('GA', 'MAP-Elites', 'annealing')


def synthetic_times(seed: int = 0, n_instances: int = 20) -> pd.DataFrame:
    """Time to reach the target (s) per instance (rows) and solver (columns); NaN marks a failure.

    The GA is fast but fails on a quarter of the instances, MAP-Elites is slower and never fails, and annealing is
    the fastest on easy instances but slow on hard ones.
    """
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=1.0, sigma=0.6, size=n_instances)
    ga = base * rng.uniform(0.8, 1.2, size=n_instances)
    ga[rng.random(n_instances) < 0.25] = np.nan
    me = base * rng.uniform(1.5, 2.5, size=n_instances)
    sa = np.where(base < np.median(base), base * 0.6, base * 4.0) * rng.uniform(0.9, 1.1, size=n_instances)
    return pd.DataFrame({'GA': ga, 'MAP-Elites': me, 'annealing': sa}, index=[f'p{i + 1}' for i in range(n_instances)])


def ratios(times: pd.DataFrame) -> pd.DataFrame:
    """Performance ratios r_ps = t_ps / best time on the instance; failures become +inf."""
    best = times.min(axis=1)
    return times.div(best, axis=0).fillna(np.inf)


def profile(times: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    """rho_s(tau) for every solver (columns) and every tau in ``taus`` (index)."""
    r = ratios(times)
    solved = np.isfinite(r)  # a failure has an infinite ratio and never counts, not even for tau = inf
    return pd.DataFrame({s: [(solved[s] & (r[s] <= tau)).mean() for tau in taus] for s in times.columns}, index=taus)


def plot_profile(times: pd.DataFrame, tau_max: float | None = None, ax=None) -> None:
    """Step plot of rho_s(tau) on a log tau axis."""
    ax = ax or plt.gca()
    r = ratios(times)
    finite = r.to_numpy()[np.isfinite(r.to_numpy())]
    tau_max = tau_max or float(finite.max()) * 1.2
    taus = np.geomspace(1.0, tau_max, 200)
    rho = profile(times, taus)
    for s in times.columns:
        ax.step(taus, rho[s], where='post', label=s)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$\rho_s(\tau)$')
    ax.set_ylim(0, 1.02)
    ax.legend()


def main() -> None:
    """The synthetic table, rho_s(1), rho_s at a few tau and the limit for large tau."""
    times = synthetic_times()
    print(times.round(2).to_string())
    r = ratios(times)
    taus = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf])
    rho = profile(times, taus)
    print(rho.round(2).to_string())
    print('failures per solver:', times.isna().sum().to_dict())
    print('largest finite ratio per solver:', r.replace(np.inf, np.nan).max().round(2).to_dict())


if __name__ == '__main__':
    main()
