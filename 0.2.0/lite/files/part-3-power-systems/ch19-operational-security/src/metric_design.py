"""Exercises 1, 2, 3, 4 and 6 of Chapter 19: what each security metric rewards, and how the AC stage decides.

Pen-and-paper exercises get their numeric check here; ``main()`` prints every number the solutions quote. The
functions build on ``metrics.py``, the NumPy version of ToOp's DC metrics.
"""

import metrics
import numpy as np
import pandas as pd


def energy_and_count(n_1: np.ndarray, f_max: np.ndarray = metrics.f_max) -> tuple[float, int]:
    """Exercise 1: overload energy and critical branch count of an N-1 matrix."""
    return metrics.overload_energy_n1(n_1, f_max), metrics.critical_branch_count_n1(n_1, f_max)


def changed(n_1: np.ndarray, contingency: int, branch: int, value: float) -> np.ndarray:
    """Exercise 1: a copy of ``n_1`` with one entry (timestep 0) replaced."""
    out = n_1.copy()
    out[0, contingency, branch] = value
    return out


def penalty_curves(loading_pct: np.ndarray, f_max: float, alpha: float = 1.5) -> dict[str, np.ndarray]:
    """Exercise 2: per-branch penalty of three metrics as a function of the loading in percent.

    Overload energy is (|F| - f_max)+ in MW; the power law is (max(|F| / f_max, 1)^alpha - 1) f_max in MW;
    cumulative overload is the relative overload (|F| - f_max)+ / f_max, blind to the rating.
    """
    rel = loading_pct / 100.0
    over = np.clip(rel - 1.0, 0, None)
    return {
        'overload_energy': over * f_max,
        'power_law': (np.clip(rel, 1, None) ** alpha - 1) * f_max,
        'cumulative_overload': over,
    }


def dead_zone_example(f_max=(100.0, 100.0), unsplit=(95.0, 60.0), split=(97.0, 58.0)) -> dict[str, float]:
    """Exercise 3: two branches, the unsplit flows set the limited limits, a redistribution gets penalised."""
    f_max, unsplit, split = (np.asarray(x, float) for x in (f_max, unsplit, split))
    limited = metrics.double_limits(unsplit[None, None, :], f_max)
    return {
        'limited_limits': limited,
        'unsplit_overload_energy_limited_n_0': metrics.overload_energy_n1(unsplit[None, None, :], limited),
        'split_overload_energy_limited_n_0': metrics.overload_energy_n1(split[None, None, :], limited),
        'split_overload_energy_n_0': metrics.overload_energy_n1(split[None, None, :], f_max),
    }


def acceptance(
    split: dict[str, float],
    unsplit: dict[str, float],
    convergence: float = 1.0,
    overload: float = 0.95,
    critical: float = 1.1,
) -> str | None:
    """Exercise 4: ToOp's ``evaluate_acceptance`` with its default thresholds and voltage checks off.

    ``split`` and ``unsplit`` hold ``non_converging_loadflows``, ``overload_energy_n_1``, ``critical_branch_count_n_1``
    and, for the split, ``disconnected_branches``. Returns the rejection criterion, or ``None`` when accepted.
    """
    non_converged = split.get('non_converging_loadflows', 0) - split.get('disconnected_branches', 0)
    if non_converged > unsplit.get('non_converging_loadflows', 0) * convergence:
        return 'convergence'
    if split.get('overload_energy_n_1', 99999) > unsplit.get('overload_energy_n_1', 0) * overload:
        return 'overload-energy'
    if split.get('critical_branch_count_n_1', 0) > unsplit.get('critical_branch_count_n_1', 999) * critical:
        return 'critical-branch-count'
    return None


def acceptance_with_slack(split: dict[str, float], unsplit: dict[str, float], slack: int = 1) -> str | None:
    """Exercise 4: the same rule with an absolute slack of ``slack`` critical branches instead of 10 %."""
    non_converged = split.get('non_converging_loadflows', 0) - split.get('disconnected_branches', 0)
    if non_converged > unsplit.get('non_converging_loadflows', 0):
        return 'convergence'
    if split.get('overload_energy_n_1', 99999) > unsplit.get('overload_energy_n_1', 0) * 0.95:
        return 'overload-energy'
    if split.get('critical_branch_count_n_1', 0) > unsplit.get('critical_branch_count_n_1', 999) + slack:
        return 'critical-branch-count'
    return None


def fitness_table(weights: dict[str, float], values: dict[str, float]) -> pd.DataFrame:
    """Exercise 6: one row per metric with value, weight and contribution; the DC fitness is minus the total."""
    table = pd.DataFrame(
        {'metric': list(weights), 'value': [values[m] for m in weights], 'weight': list(weights.values())}
    )
    table['contribution'] = table.value * table.weight
    return table


def main() -> None:
    """Print the numeric checks of Exercises 1, 2, 3, 4 and 6."""
    base = energy_and_count(metrics.n_1)
    print('Exercise 1: worked example (energy, count) =', base, 'of', metrics.n_1.shape[2], 'branches')
    more = changed(metrics.n_1, 1, 0, 140.0)  # branch 0, contingency 2: 120 -> 140 MW
    print('  120 -> 140 MW on branch 0 in contingency 2: (energy, count) =', energy_and_count(more))
    same = changed(metrics.n_1, 0, 0, 115.0)  # branch 0, contingency 1: 95 -> 115 MW, below its worst case
    print('  95 -> 115 MW on branch 0 in contingency 1: (energy, count) =', energy_and_count(same))
    smaller = metrics.n_1 * np.array([1.0, 1.0, 0.9])  # branch 2 at 36 and 49.5 MW: not critical any more
    print('  start with branch 2 below its limit: (energy, count) =', energy_and_count(smaller))
    print('    then 49.5 -> 50.001 MW: (energy, count) =', energy_and_count(changed(smaller, 1, 2, 50.001)))

    loading = np.array([100.0, 110.0, 150.0])
    for f_max in (100.0, 1000.0):
        curves = penalty_curves(loading, f_max)
        print(
            f'Exercise 2: f_max = {f_max:.0f} MW, loading {loading} %:', {k: np.round(v, 3) for k, v in curves.items()}
        )

    print('Exercise 3:', dead_zone_example())

    unsplit = {'non_converging_loadflows': 2, 'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 0}
    split = {
        'non_converging_loadflows': 2,
        'disconnected_branches': 1,
        'overload_energy_n_1': 0.0,
        'critical_branch_count_n_1': 1,
    }
    print(
        'Exercise 4: default thresholds ->',
        acceptance(split, unsplit),
        '; with one branch of slack ->',
        acceptance_with_slack(split, unsplit),
    )

    table = fitness_table(
        {'overload_energy_n_1': 1.0, 'critical_branch_count_n_1': 10.0},
        {'overload_energy_n_1': 213.0, 'critical_branch_count_n_1': 50},
    )
    print('Exercise 6:')
    print(table.to_string(index=False))
    print('  DC fitness = -', table.contribution.sum(), '; AC fitness = overload_energy_n_1 in MW, lower is better')


if __name__ == '__main__':
    main()
