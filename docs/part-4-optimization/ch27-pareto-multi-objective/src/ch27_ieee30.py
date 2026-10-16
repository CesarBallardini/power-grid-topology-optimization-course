"""IEEE 30 line switching as a bi-objective problem (Chapter 27, Lab 6).

Objectives, both minimized: the N-1 overload energy of the switched network and the number of switched-off lines.
The exact front for at most ``K_MAX`` switched lines comes from enumeration; a small NSGA-II and a small MAP-Elites
built on ``pareto.py`` are compared with it by hypervolume and IGD+. Run it to print every number quoted in the
solutions::

    python ch27_ieee30.py

The network is pandapower's ``case30`` (MATPOWER data, 41 lines with ratings). Loads and generation are scaled by
``SCALE`` so that the intact network already has N-1 overloads, as the Chapter 24 lab asks. Chapters 28 and 29 import
this module for the same network data.
"""

import itertools
import time

import numpy as np
import pandapower.networks as pn
import pareto
from scipy.sparse.csgraph import connected_components

SCALE = 1.5  # load and generation multiplier that makes the intact grid N-1 insecure
K_MAX = 3  # at most this many switched-off lines in the enumeration
EVALUATIONS = 1200  # budget of the heuristics, about 10 % of the enumeration up to K_MAX
PENALTY = 1e6  # overload energy assigned to an islanded network inside the heuristics (the exact code uses inf)


# --8<-- [start:network]
def load_case(scale: float = SCALE) -> dict:
    """Branch list, ratings and net injections of IEEE 30 with loads and generation scaled by ``scale``.

    Returns a dict with ``lines`` (from, to) zero-based, ``x`` (p.u. reactance), ``rating`` (MW), ``p`` (net injection
    per bus in MW, the slack bus 0 balancing the rest) and ``n_bus``.
    """
    net = pn.case30()
    z_base = net.bus.vn_kv.iloc[0] ** 2 / net.sn_mva  # ohm
    x = (net.line.x_ohm_per_km * net.line.length_km / z_base).to_numpy()
    rating = (np.sqrt(3) * net.bus.vn_kv.iloc[0] * net.line.max_i_ka).to_numpy()
    p = np.zeros(len(net.bus))
    np.add.at(p, net.gen.bus.to_numpy(), scale * net.gen.p_mw.to_numpy())
    np.add.at(p, net.load.bus.to_numpy(), -scale * net.load.p_mw.to_numpy())
    p[0] = -p[1:].sum()
    return {
        'lines': net.line[['from_bus', 'to_bus']].to_numpy(),
        'x': x,
        'rating': rating,
        'p': p,
        'n_bus': len(net.bus),
    }


def ptdf(case: dict, in_service: np.ndarray) -> np.ndarray | None:
    """PTDF of the lines in service (slack bus 0), or ``None`` when they do not connect every bus."""
    lines, n_bus = case['lines'][in_service], case['n_bus']
    a = np.zeros((len(lines), n_bus))
    a[np.arange(len(lines)), lines[:, 0]] = 1
    a[np.arange(len(lines)), lines[:, 1]] = -1
    adjacency = np.zeros((n_bus, n_bus))
    adjacency[lines[:, 0], lines[:, 1]] = 1
    if connected_components(adjacency, directed=False)[0] > 1:
        return None
    b = 1 / case['x'][in_service]
    bbus = a.T @ (b[:, None] * a)
    h = np.zeros((len(lines), n_bus))
    h[:, 1:] = (b[:, None] * a[:, 1:]) @ np.linalg.inv(bbus[1:, 1:])
    return h


def overload_energy(case: dict, switched_off: tuple[int, ...] = ()) -> float:
    """N-1 overload energy in MW of the network without the lines in ``switched_off``; ``inf`` when it is islanded.

    Sum over the non-bridge single outages and over every monitored line of the flow above the rating. The outage of
    a bridge (a line whose loss islands the grid, LODF denominator zero) is left out, as ToOp does.
    """
    in_service = np.ones(len(case['lines']), dtype=bool)
    in_service[list(switched_off)] = False
    h = ptdf(case, in_service)
    if h is None:
        return np.inf
    lines, rating = case['lines'][in_service], case['rating'][in_service]
    f = h @ case['p']
    hk = h[:, lines[:, 0]] - h[:, lines[:, 1]]  # column k: sensitivity to a transfer along line k
    denominator = 1 - np.diag(hk)
    not_bridge = np.abs(denominator) > 1e-9
    lodf = hk[:, not_bridge] / denominator[not_bridge]
    post = f[:, None] + lodf * f[not_bridge]  # (line, outage)
    post[np.flatnonzero(not_bridge), np.arange(not_bridge.sum())] = 0  # the outaged line carries nothing
    return float(np.maximum(0, np.abs(post) - rating[:, None]).sum())


# --8<-- [end:network]
# --8<-- [start:front]
def enumerate_front(case: dict, k_max: int = K_MAX) -> tuple[np.ndarray, list[tuple[int, ...]], int]:
    """Exact bi-objective front (overload energy, number of switched lines) for at most ``k_max`` switched lines.

    Returns the non-dominated points, the set of lines behind each, and the number of evaluations.
    """
    n_line = len(case['lines'])
    best: list[tuple[float, tuple[int, ...]]] = []
    count = 0
    for k in range(k_max + 1):
        value, arg = np.inf, ()
        for subset in itertools.combinations(range(n_line), k):
            count += 1
            e = overload_energy(case, subset)
            if e < value:
                value, arg = e, subset
        best.append((value, arg))
    points = np.array([[e, k] for k, (e, _) in enumerate(best)], dtype=float)
    keep = pareto.non_dominated_mask(points)
    return points[keep], [best[k][1] for k in np.flatnonzero(keep)], count


def _evaluate(case: dict, genome: np.ndarray, n_line: int) -> np.ndarray:
    """Objectives of a genome: sorted line indices padded with ``n_line`` (the sentinel of Chapter 24)."""
    switched = tuple(int(g) for g in genome if g < n_line)
    return np.array([min(overload_energy(case, switched), PENALTY), len(switched)])


def _random_genome(rng: np.random.Generator, n_line: int, k_max: int) -> np.ndarray:
    k = rng.integers(k_max + 1)
    g = np.full(k_max, n_line)
    g[:k] = rng.choice(n_line, k, replace=False)
    return np.sort(g)


def _mutate(rng: np.random.Generator, genome: np.ndarray, n_line: int) -> np.ndarray:
    g = genome.copy()
    slot = rng.integers(len(g))
    g[slot] = n_line if rng.random() < 0.3 else rng.integers(n_line)
    unique = np.unique(g[g < n_line])
    return np.concatenate([unique, np.full(len(g) - len(unique), n_line)])


def nsga2(case: dict, seed: int, evaluations: int = EVALUATIONS, pop: int = 24, k_max: int = K_MAX) -> np.ndarray:
    """A minimal NSGA-II (rank, then crowding distance) with the mutation of Chapter 24; returns the final front."""
    rng = np.random.default_rng(seed)
    n_line = len(case['lines'])
    genomes = np.array([_random_genome(rng, n_line, k_max) for _ in range(pop)])
    objectives = np.array([_evaluate(case, g, n_line) for g in genomes])
    used = pop
    while used < evaluations:
        children = np.array([_mutate(rng, genomes[rng.integers(pop)], n_line) for _ in range(pop)])
        child_obj = np.array([_evaluate(case, g, n_line) for g in children])
        used += pop
        genomes, objectives = np.vstack([genomes, children]), np.vstack([objectives, child_obj])
        ranks = pareto.fast_non_dominated_sort(objectives)
        crowd = pareto.crowding_distance(objectives, ranks)
        order = np.lexsort((-crowd, ranks))[:pop]
        genomes, objectives = genomes[order], objectives[order]
    finite = objectives[:, 0] < PENALTY
    front = objectives[finite][pareto.non_dominated_mask(objectives[finite])]
    return np.unique(front, axis=0)


def map_elites(case: dict, seed: int, evaluations: int = EVALUATIONS, k_max: int = K_MAX) -> np.ndarray:
    """MAP-Elites with one cell per number of switched lines, elite = lowest overload energy; returns its front."""
    rng = np.random.default_rng(seed)
    n_line = len(case['lines'])
    cells: dict[int, tuple[float, np.ndarray]] = {}
    for used in range(evaluations):
        parent = cells[rng.choice(list(cells))][1] if cells and used >= 50 else _random_genome(rng, n_line, k_max)
        genome = _mutate(rng, parent, n_line) if cells and used >= 50 else parent
        e, k = _evaluate(case, genome, n_line)
        if e < PENALTY and (int(k) not in cells or e < cells[int(k)][0]):
            cells[int(k)] = (e, genome)
    points = np.array([[e, k] for k, (e, _) in sorted(cells.items())])
    return points[pareto.non_dominated_mask(points)]


def compare(case: dict, exact: np.ndarray, seeds: range = range(5)) -> dict[str, dict[str, list[float]]]:
    """Hypervolume and IGD+ of NSGA-II and MAP-Elites fronts against the exact front, normalized, per seed."""
    ideal, nadir = exact.min(axis=0), exact.max(axis=0)
    reference = np.array([1.1, 1.1])

    def normalize(points: np.ndarray) -> np.ndarray:
        return (points - ideal) / np.where(nadir > ideal, nadir - ideal, 1)

    out = {'nsga2': {'hv': [], 'igd_plus': []}, 'map_elites': {'hv': [], 'igd_plus': []}}
    for seed in seeds:
        for name, run in (('nsga2', nsga2), ('map_elites', map_elites)):
            front = normalize(run(case, seed))
            out[name]['hv'].append(pareto.hypervolume_2d(front, reference))
            out[name]['igd_plus'].append(pareto.igd_plus(front, normalize(exact)))
    return out


# --8<-- [end:front]
def main() -> None:
    """Exact front, time per evaluation, and the heuristics against it."""
    case = load_case()
    print(f'scale {SCALE}: intact N-1 overload energy {overload_energy(case):.1f} MW')
    t0 = time.perf_counter()
    exact, sets, count = enumerate_front(case)
    dt = time.perf_counter() - t0
    print(f'enumeration of |D| <= {K_MAX}: {count} evaluations in {dt:.1f} s ({1e3 * dt / count:.2f} ms each)')
    for (e, k), lines in zip(exact, sets, strict=True):
        print(f'  {int(k)} switched line(s) {list(lines)}: overload energy {e:.1f} MW')
    print(f'exact front: {len(exact)} points; dominated k values are left out')
    ideal, nadir = exact.min(axis=0), exact.max(axis=0)
    print(f'ideal {ideal}, nadir {nadir}, normalized hypervolume of the exact front with reference (1.1, 1.1):')
    scaled = (exact - ideal) / np.where(nadir > ideal, nadir - ideal, 1)
    print(f'  {pareto.hypervolume_2d(scaled, np.array([1.1, 1.1])):.4f}')
    results = compare(case, exact)
    for name, values in results.items():
        hv, igd = np.array(values['hv']), np.array(values['igd_plus'])
        print(f'{name:10s} HV  median {np.median(hv):.4f}  min {hv.min():.4f} max {hv.max():.4f}')
        print(f'{name:10s} IGD+ median {np.median(igd):.4f}  min {igd.min():.4f} max {igd.max():.4f}')
    print('knee points of the exact front (normalized):', pareto.knee_points(scaled))
    print('pseudo-weights of the exact front:', np.round(pareto.pseudo_weights(exact), 3))
    reference = (np.array([0.0, 1.0]) - ideal) / (nadir - ideal)  # "no overload with one switched line", normalized
    print(
        'achievement function, reference point (0 MW, 1 line):',
        exact[pareto.achievement_scalarizing(scaled, reference)],
    )


if __name__ == '__main__':
    main()
