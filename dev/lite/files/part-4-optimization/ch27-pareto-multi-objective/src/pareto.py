"""Pareto dominance, scalarizations, non-dominated sorting and quality indicators in NumPy (Chapter 27).

Every objective is minimized. Points are rows of a 2-D array. Run it to print the numbers of Exercises 1-4::

    python pareto.py
"""

import numpy as np
from scipy.optimize import linprog

POINTS = np.array([[1, 9], [3, 4], [4, 3.5], [6, 2], [9, 1]], dtype=float)  # Exercise 1


# --8<-- [start:dominance]
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """``a`` weakly better in every objective and strictly better in at least one."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_mask(points: np.ndarray) -> np.ndarray:
    """True for the points that no other point dominates (duplicates of a non-dominated point are kept)."""
    p = points[:, None, :]  # candidate
    q = points[None, :, :]  # possible dominator
    dominated_by = np.all(q <= p, axis=2) & np.any(q < p, axis=2)
    return ~dominated_by.any(axis=1)


def fast_non_dominated_sort(points: np.ndarray) -> np.ndarray:
    """Rank of every point (0 = first front) with the domination counts of Deb et al. 2002."""
    n = len(points)
    p, q = points[:, None, :], points[None, :, :]
    dom = np.all(p <= q, axis=2) & np.any(p < q, axis=2)  # dom[i, j]: i dominates j
    count = dom.sum(axis=0)  # how many dominate each point
    ranks = np.full(n, -1)
    front = np.flatnonzero(count == 0)
    rank = 0
    while front.size:
        ranks[front] = rank
        count = count - dom[front].sum(axis=0)
        count[ranks >= 0] = -1  # already assigned
        front = np.flatnonzero(count == 0)
        rank += 1
    return ranks


def crowding_distance(points: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Crowding distance within each front; boundary points get ``inf`` (Deb et al. 2002)."""
    distance = np.zeros(len(points))
    for rank in np.unique(ranks):
        idx = np.flatnonzero(ranks == rank)
        if len(idx) <= 2:
            distance[idx] = np.inf
            continue
        for m in range(points.shape[1]):
            order = idx[np.argsort(points[idx, m], kind='stable')]
            span = points[order[-1], m] - points[order[0], m]
            distance[order[[0, -1]]] = np.inf
            if span > 0:
                distance[order[1:-1]] += (points[order[2:], m] - points[order[:-2], m]) / span
    return distance


# --8<-- [end:dominance]
# --8<-- [start:scalarization]
def weighted_sum_argmin(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Indices of every minimizer of the weighted sum (ties kept, within 1e-12)."""
    values = points @ weights
    return np.flatnonzero(values <= values.min() + 1e-12)


def weighted_sum_sweep(points: np.ndarray, n_weights: int = 1001) -> set[int]:
    """Indices that win the two-objective weighted sum for some weight (w, 1 - w) on a grid of ``n_weights`` values."""
    winners: set[int] = set()
    for w in np.linspace(0, 1, n_weights):
        winners.update(int(i) for i in weighted_sum_argmin(points, np.array([w, 1 - w])))
    return winners


def supported_mask(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """True for the non-dominated points that minimize some weighted sum with non-negative weights.

    For each point ``p`` an LP looks for weights on the simplex with ``w . (q - p) >= t`` for every other point
    ``q`` and the largest ``t``; ``p`` is supported when that ``t`` is not negative. Points inside a non-convex
    part of the front (unsupported) have ``t < 0`` for every weight.
    """
    n, m = points.shape
    supported = np.zeros(n, dtype=bool)
    nd = non_dominated_mask(points)
    for i in np.flatnonzero(nd):
        others = np.delete(points, i, axis=0) - points[i]
        # variables: w (m), t; maximize t  <=>  minimize -t
        c = np.r_[np.zeros(m), -1.0]
        a_ub = np.c_[-others, np.ones(len(others))]  # t - w.(q-p) <= 0
        a_eq = np.r_[np.ones(m), 0.0][None, :]
        bounds = [(0, None)] * m + [(None, None)]
        res = linprog(c, A_ub=a_ub, b_ub=np.zeros(len(others)), A_eq=a_eq, b_eq=[1.0], bounds=bounds, method='highs')
        supported[i] = res.success and -res.fun >= -tol
    return supported


def epsilon_constraint(points: np.ndarray, objective: int, bounds: np.ndarray) -> int | None:
    """Index of the point minimizing ``objective`` among those whose other objectives respect ``bounds``.

    Ties on the minimized objective are broken by the sum of the others, which removes the weakly efficient
    points that a plain epsilon-constraint can return (the slack term of AUGMECON plays the same role).
    """
    others = np.delete(np.arange(points.shape[1]), objective)
    feasible = np.flatnonzero(np.all(points[:, others] <= bounds[others] + 1e-12, axis=1))
    if feasible.size == 0:
        return None
    key = np.lexsort((points[feasible][:, others].sum(axis=1), points[feasible, objective]))
    return int(feasible[key[0]])


def achievement_scalarizing(points: np.ndarray, reference: np.ndarray, weights=None, rho: float = 1e-3) -> int:
    """Index minimizing the achievement scalarizing function max_i w_i (f_i - r_i) + rho sum_i w_i (f_i - r_i)."""
    w = np.ones(points.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    d = w * (points - reference)
    return int(np.argmin(d.max(axis=1) + rho * d.sum(axis=1)))


def pseudo_weights(points: np.ndarray) -> np.ndarray:
    """Pseudo-weight vector of each point: (f_max - f_i) / (f_max - f_min), normalized to sum one per point."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    w = (hi - points) / span
    return w / w.sum(axis=1, keepdims=True)


def knee_points(points: np.ndarray) -> np.ndarray:
    """Indices of the two-objective front points farthest below the line through its two extreme points.

    A knee is where a small gain in one objective costs a large loss in the other. On a convex front the farthest
    point from the chord between the extremes is the usual knee estimate; every point within 1e-9 of that distance
    is returned.
    """
    order = np.argsort(points[:, 0])
    a, b = points[order[0]], points[order[-1]]
    direction = b - a
    normal = np.array([direction[1], -direction[0]])
    normal = normal / (np.linalg.norm(normal) or 1.0)
    distance = (points - a) @ normal
    distance = distance * np.sign(distance[np.abs(distance).argmax()] or 1.0)
    below = np.flatnonzero(distance >= distance.max() - 1e-9)
    return below if distance.max() > 1e-9 else np.array([], dtype=int)


# --8<-- [end:scalarization]
# --8<-- [start:indicators]
def hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    """Area dominated by the two-objective ``points`` and bounded by ``reference``: sort by f1, add rectangles."""
    p = points[np.all(points < reference, axis=1)]
    if len(p) == 0:
        return 0.0
    p = p[non_dominated_mask(p)]
    p = p[np.argsort(p[:, 0])]
    widths = np.diff(np.r_[p[:, 0], reference[0]])
    heights = reference[1] - p[:, 1]
    return float((widths * heights).sum())


def igd_plus(front: np.ndarray, reference_front: np.ndarray) -> float:
    """IGD+ of Ishibuchi et al.: mean over reference points of the smallest dominance-aware distance to the front."""
    diff = np.maximum(front[None, :, :] - reference_front[:, None, :], 0)  # (reference, front, objective)
    return float(np.linalg.norm(diff, axis=2).min(axis=1).mean())


def epsilon_indicator(front: np.ndarray, reference_front: np.ndarray) -> float:
    """Additive epsilon indicator: the smallest shift that makes ``front`` weakly dominate every reference point."""
    diff = front[None, :, :] - reference_front[:, None, :]
    return float(diff.max(axis=2).min(axis=1).max())


# --8<-- [end:indicators]
def main() -> None:
    """Exercises 1-4 of Chapter 27."""
    nd = non_dominated_mask(POINTS)
    supported = supported_mask(POINTS)
    print('Exercise 1: non-dominated', POINTS[nd].tolist())
    print('  supported', POINTS[supported].tolist(), ' unsupported', POINTS[nd & ~supported].tolist())
    print('  weighted-sum winners over 1001 weights:', sorted(weighted_sum_sweep(POINTS)))
    print('  eps-constraint f2 <= 3.6 minimizing f1:', POINTS[epsilon_constraint(POINTS, 0, np.array([0, 3.6]))])
    print('  Tchebycheff with reference (0, 0):', POINTS[achievement_scalarizing(POINTS, np.zeros(2), rho=0)])

    weak = np.array([[1, 2], [1, 3], [2, 1]], dtype=float)
    print('Exercise 2: argmin of (1, 0) weights on', weak.tolist(), '->', weighted_sum_argmin(weak, np.array([1, 0])))

    rng = np.random.default_rng(seed=27)
    cloud = rng.random((1000, 3))
    ranks = fast_non_dominated_sort(cloud)
    print(
        f'Exercise 3: 1000 random points in 3 objectives: {ranks.max() + 1} fronts, {np.sum(ranks == 0)} in the first'
    )
    crowd = crowding_distance(cloud, ranks)
    print(f'  crowding distance in front 0: {np.isinf(crowd[ranks == 0]).sum()} boundary points (inf)')

    front = POINTS
    nadir = front.max(axis=0)
    print('Exercise 4: hypervolume of the Exercise 1 points for reference points nadir, 1.1 nadir, 2 nadir:')
    for factor in (1.0, 1.1, 2.0):
        print(f'  factor {factor}: HV = {hypervolume_2d(front, factor * nadir):.4f}')
    hits = rng.random((200_000, 2)) * 2 * nadir
    monte_carlo = np.any(np.all(hits[:, None, :] >= front[None, :, :], axis=2), axis=1).mean() * np.prod(2 * nadir)
    print(f'  Monte Carlo check of the 2 nadir value with 200,000 samples: {monte_carlo:.2f}')


if __name__ == '__main__':
    main()
