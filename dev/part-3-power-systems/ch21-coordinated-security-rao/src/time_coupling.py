"""One topology for the whole day, or one per hour with a switching cost: Chapter 21, Exercise 6.

The hourly overload energies are made up for the exercise: a daily load shape, four candidate topologies that each
relieve a different part of the day, drawn once from a seeded generator so that every run sees the same table.
"""

import numpy as np

HOURS, N_TOPO = 24, 4


def overload_table(seed: int = 21) -> np.ndarray:
    """Overload energy (MW) per hour and candidate topology; topology 0 is the unsplit grid."""
    rng = np.random.default_rng(seed)
    hours = np.arange(HOURS)
    load = 0.55 + 0.45 * np.sin(np.pi * (hours - 5) / 14).clip(0)  # low at night, peak in the afternoon
    base = 60 * load - 15 + rng.normal(0, 3, HOURS)
    relief = np.column_stack(
        [
            np.zeros(HOURS),
            18 + 6 * np.cos(np.pi * (hours - 9) / 12),  # best in the morning
            18 + 6 * np.cos(np.pi * (hours - 17) / 12),  # best in the evening
            14 + np.zeros(HOURS),  # mediocre everywhere
        ]
    )
    return np.maximum(base[:, None] - relief, 0.0).round(1)


def best_single(table: np.ndarray) -> tuple[int, float]:
    """The topology held all day with the least total overload energy: ToOp scores one topology on all timesteps."""
    totals = table.sum(axis=0)
    return int(totals.argmin()), float(totals.min())


def best_sequence(table: np.ndarray, switching_cost: float) -> tuple[list[int], float, int]:
    """Dynamic programming over the hours: topology per hour, total cost (overload plus switches) and switch count."""
    n_hours, n_topo = table.shape
    cost = table[0].copy()
    back = np.zeros((n_hours, n_topo), dtype=int)
    for h in range(1, n_hours):
        stay = cost
        move = cost.min() + switching_cost
        best_prev = int(cost.argmin())
        new = np.empty(n_topo)
        for t in range(n_topo):
            if stay[t] <= move:
                new[t], back[h, t] = stay[t], t
            else:
                new[t], back[h, t] = move, best_prev
        cost = new + table[h]
    path = [int(cost.argmin())]
    for h in range(n_hours - 1, 0, -1):
        path.append(int(back[h, path[-1]]))
    path.reverse()
    switches = int(np.sum(np.diff(path) != 0))
    return path, float(cost.min()), switches


def break_even(table: np.ndarray, step: float = 0.5) -> float:
    """The smallest switching cost (on a grid of ``step``) at which the hourly sequence stops switching."""
    c = 0.0
    while best_sequence(table, c)[2] > 0:
        c += step
    return c


def main() -> None:
    """Totals of the single topology, of the free hourly sequence and the switching cost that makes them coincide."""
    table = overload_table()
    print('hourly overload energy (MW), columns = topologies 0-3:')
    print(table)
    topo, total = best_single(table)
    print(f'best single topology: {topo}, total {total:.1f} MW')
    for c in (0.0, 5.0, 10.0, 20.0):
        path, cost, switches = best_sequence(table, c)
        overload = float(table[np.arange(HOURS), path].sum())
        print(f'switching cost {c:4.1f}: {path}, overload {overload:.1f} MW, {switches} switches, cost {cost:.1f}')
    c_star = break_even(table)
    print(f'the sequence stops switching at a switching cost of {c_star:.1f} per change')


if __name__ == '__main__':
    main()
