"""IEEE 30 line switching: the N-1 overload energy of a switched topology and its brute-force optimum (Chapter 24).

The grid is MATPOWER's ``case30`` as shipped with pandapower (41 branches with MVA ratings). Loads and generation
are scaled by a common factor so that the unswitched grid has N-1 overloads; the slack bus absorbs the imbalance.
The fitness of a set D of switched-off lines is the negated N-1 overload energy of the reduced grid, computed from
its PTDF and LODF matrices, or -inf when removing D islands the grid.
"""

import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandapower.networks as pn
from pandapower.converter.pypower import to_ppc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SCALE = 1.4  # common factor on loads and generation; the unswitched grid then has 51 N-1 overloads


@dataclass(frozen=True)
class Case:
    """A DC grid: branch ends (0-based), reactances (p.u.), MW limits, net injections (MW) and branch names."""

    frm: np.ndarray
    to: np.ndarray
    x: np.ndarray
    limit: np.ndarray
    p: np.ndarray
    names: tuple[str, ...]

    @property
    def n_bus(self) -> int:
        """Number of buses."""
        return len(self.p)

    @property
    def n_branch(self) -> int:
        """Number of branches."""
        return len(self.frm)


def load_case30(scale: float = SCALE) -> Case:
    """MATPOWER case30 through pandapower, with every load and generator scaled by ``scale``."""
    ppc = to_ppc(pn.case30(), init='flat')
    branch, bus, gen = ppc['branch'], ppc['bus'], ppc['gen']
    frm, to = branch[:, 0].astype(int), branch[:, 1].astype(int)
    p = np.zeros(len(bus))
    np.add.at(p, gen[:, 0].astype(int), gen[:, 1])
    p -= bus[:, 2]
    names = tuple(f'L{f + 1}-{t + 1}' for f, t in zip(frm, to, strict=True))
    return Case(frm, to, branch[:, 3].copy(), branch[:, 5].copy(), scale * p, names)


def kept_branches(case: Case, switched) -> np.ndarray:
    """Indices of the branches that stay in service when the lines in ``switched`` are opened."""
    mask = np.ones(case.n_branch, dtype=bool)
    mask[list(switched)] = False
    return np.flatnonzero(mask)


def is_connected(case: Case, switched) -> bool:
    """True when the grid without the lines in ``switched`` is a single connected component."""
    keep = kept_branches(case, switched)
    adjacency = coo_matrix((np.ones(len(keep)), (case.frm[keep], case.to[keep])), shape=(case.n_bus, case.n_bus))
    return connected_components(adjacency, directed=False)[0] == 1


def ptdf_matrix(case: Case, keep: np.ndarray, slack: int = 0) -> np.ndarray:
    """PTDF (kept branches x buses) of the reduced grid, bus ``slack`` as reference; the grid must be connected."""
    m = len(keep)
    a = np.zeros((m, case.n_bus))
    a[np.arange(m), case.frm[keep]] = 1.0
    a[np.arange(m), case.to[keep]] = -1.0
    bf = a / case.x[keep][:, None]
    bbus = a.T @ bf
    other = np.arange(case.n_bus) != slack
    h = np.zeros_like(a)
    h[:, other] = np.linalg.solve(bbus[np.ix_(other, other)], bf[:, other].T).T
    return h


def n1_flows(case: Case, switched=()) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Base flows, N-1 flow matrix (branch, outage) and the bridge mask of the reduced grid; None if islanded.

    Column k of the N-1 matrix holds the flows after the outage of kept branch k, from the LODF update
    f + LODF[:, k] f_k. Outages of bridges would island the grid; their columns are left equal to the base flows
    and flagged in the returned mask so that the caller can skip them.
    """
    if not is_connected(case, switched):
        return None
    keep = kept_branches(case, switched)
    h = ptdf_matrix(case, keep)
    f0 = h @ case.p
    m = len(keep)
    a = np.zeros((m, case.n_bus))
    a[np.arange(m), case.frm[keep]] = 1.0
    a[np.arange(m), case.to[keep]] = -1.0
    c = h @ a.T  # c[l, k]: flow on l per MW injected at the ends of k
    denominator = 1.0 - np.diag(c)
    bridge = np.abs(denominator) < 1e-9
    lodf = c / np.where(bridge, 1.0, denominator)[None, :]
    np.fill_diagonal(lodf, -1.0)
    lodf[:, bridge] = 0.0
    return f0, f0[:, None] + lodf * f0[None, :], bridge


def overload_energy(case: Case, switched=()) -> float:
    """N-1 overload energy E(D) in MW: sum over non-bridge outages and monitored branches of the MW above the limit.

    Returns +inf when the switched grid is disconnected.
    """
    result = n1_flows(case, switched)
    if result is None:
        return np.inf
    _, f1, bridge = result
    keep = kept_branches(case, switched)
    over = np.maximum(0.0, np.abs(f1) - case.limit[keep][:, None])
    over[:, bridge] = 0.0
    return float(over.sum())


def fitness(case: Case, switched=()) -> float:
    """Negated overload energy, -inf for an islanding set: the quantity the heuristics maximize."""
    return -overload_energy(case, switched)


def worst_loading(case: Case, switched=()) -> float:
    """Largest N-1 loading in percent of the limit over the non-bridge outages (inf if islanded)."""
    result = n1_flows(case, switched)
    if result is None:
        return np.inf
    _, f1, bridge = result
    keep = kept_branches(case, switched)
    loading = 100.0 * np.abs(f1) / case.limit[keep][:, None]
    loading[:, bridge] = 0.0
    return float(loading.max())


def dc_power_flow_check(case: Case, switched, outage: int) -> float:
    """Largest difference (MW) between the LODF post-contingency flows and a fresh DC power flow without ``outage``.

    ``outage`` is a branch index of the original grid, not in ``switched``.
    """
    _, f1, _ = n1_flows(case, switched)
    keep = kept_branches(case, switched)
    column = int(np.flatnonzero(keep == outage)[0])
    full = kept_branches(case, (*switched, outage))
    direct = ptdf_matrix(case, full) @ case.p
    lodf_flows = np.delete(f1[:, column], column)
    return float(np.abs(lodf_flows - direct).max())


@dataclass
class BruteForce:
    """Result of an exhaustive enumeration of all sets with at most ``k_max`` switched lines."""

    k_max: int
    values: dict[tuple[int, ...], float]
    seconds: float

    @property
    def n_evaluations(self) -> int:
        """Number of fitness evaluations, one per set."""
        return len(self.values)

    @property
    def best(self) -> tuple[tuple[int, ...], float]:
        """The optimal set and its fitness."""
        return max(self.values.items(), key=lambda item: item[1])

    @property
    def n_islanding(self) -> int:
        """Number of enumerated sets that disconnect the grid."""
        return sum(1 for v in self.values.values() if v == -np.inf)


def brute_force(case: Case, k_max: int) -> BruteForce:
    """Fitness of every set of at most ``k_max`` lines, enumerated lazily with ``itertools.combinations``."""
    start = time.perf_counter()
    sets = itertools.chain.from_iterable(itertools.combinations(range(case.n_branch), k) for k in range(k_max + 1))
    values = {d: fitness(case, d) for d in sets}
    return BruteForce(k_max, values, time.perf_counter() - start)


def count_sets(n: int, k_max: int) -> int:
    """Number of subsets of at most ``k_max`` out of ``n`` lines."""
    from math import comb

    return sum(comb(n, k) for k in range(k_max + 1))


def interaction(case: Case, a: int, b: int) -> tuple[float, float, float, float]:
    """E(empty), E({a}), E({b}) and E({a, b}); the benefits are additive when the last equals the sum of gains."""
    return (
        overload_energy(case),
        overload_energy(case, (a,)),
        overload_energy(case, (b,)),
        overload_energy(case, (a, b)),
    )


def per_line_energy(case: Case, switched=()) -> dict[str, float]:
    """Contribution of each overloaded branch to E(D), summed over the outages, by branch name (largest first)."""
    _, f1, bridge = n1_flows(case, switched)
    keep = kept_branches(case, switched)
    over = np.maximum(0.0, np.abs(f1) - case.limit[keep][:, None])
    over[:, bridge] = 0.0
    totals = over.sum(axis=1)
    order = np.argsort(-totals)
    return {case.names[keep[i]]: float(totals[i]) for i in order if totals[i] > 0}


def greedy(case: Case, k_max: int) -> tuple[list[int], list[float]]:
    """Add the single line that lowers E most at each step (no backtracking); the path of sets and energies."""
    chosen: list[int] = []
    energies = [overload_energy(case)]
    for _ in range(k_max):
        candidates = [
            (overload_energy(case, (*chosen, line)), line) for line in range(case.n_branch) if line not in chosen
        ]
        best_energy, best_line = min(candidates)
        if best_energy >= energies[-1]:
            break
        chosen.append(best_line)
        energies.append(best_energy)
    return chosen, energies


def describe(case: Case, switched) -> str:
    """Branch names of a set, for printing."""
    return '{' + ', '.join(case.names[i] for i in sorted(switched)) + '}' if len(switched) else '{}'


def main() -> None:
    """Lab steps 1 and 2 and Exercise 3: evaluator, brute force up to three lines, non-additive pair, greedy path."""
    for scale in (1.0, 1.2, 1.4):
        print(f'scale {scale}: E(empty) = {overload_energy(load_case30(scale)):.3f} MW')
    case = load_case30()
    _, f1, bridge = n1_flows(case)
    over = (np.abs(f1) - case.limit[:, None] > 0) & ~bridge[None, :]
    print(
        f'unswitched grid: {int(over.sum())} N-1 overloads, worst loading {worst_loading(case):.1f} %, '
        f'bridges {[case.names[i] for i in np.flatnonzero(bridge)]}'
    )
    for switched, outage in (((), 0), ((32, 40), 5), ((34,), 10)):
        print(
            f'LODF vs DC power flow, D={describe(case, switched)} outage {case.names[outage]}: '
            f'{dc_power_flow_check(case, switched, outage):.2e} MW'
        )
    a, b = 32, 40  # L24-25 and L6-28, the best pair
    for k_max in (2, 3):
        bf = brute_force(case, k_max)
        best, value = bf.best
        print(
            f'|D| <= {k_max}: {bf.n_evaluations} evaluations in {bf.seconds:.1f} s, {bf.n_islanding} islanding sets, '
            f'optimum E = {-value:.3f} MW at {describe(case, best)}'
        )
    per_eval = bf.seconds / bf.n_evaluations
    for k_max in (4, 5):
        n_sets = count_sets(case.n_branch, k_max)
        print(f'|D| <= {k_max}: {n_sets} sets, predicted {per_eval * n_sets:.0f} s')
    good = [d for d, v in bf.values.items() if -v < 30]
    with_pair = sum(1 for d in good if a in d and b in d)
    print(f'{len(good)} sets below 30 MW, {with_pair} of them contain both lines of the best pair')
    e0, ea, eb, eab = interaction(case, a, b)
    print(
        f'E(empty)={e0:.3f} E({case.names[a]})={ea:.3f} E({case.names[b]})={eb:.3f} E(both)={eab:.3f}; '
        f'additive prediction {ea + eb - e0:.3f}'
    )
    for switched in ((), (a,), (b,), (a, b)):
        top = list(per_line_energy(case, switched).items())[:3]
        print(f'  D={describe(case, switched)}: ' + ', '.join(f'{name} {e:.2f}' for name, e in top))
    path, energies = greedy(case, 3)
    print('greedy path', [case.names[i] for i in path], [round(e, 3) for e in energies])


if __name__ == '__main__':
    main()
