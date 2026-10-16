"""Chapter 27, Exercise 3 with pymoo: the NumPy sorting and crowding distance of ``pareto.py`` against pymoo's.

Runs in the labs container (``make test-labs``, see ``docker/labs``), where pymoo is installed. pymoo's
``NonDominatedSorting().do(F)`` returns the fronts as lists of indices, and ``calc_crowding_distance(F)`` takes one
front and divides the sum of the normalized gaps by the number of objectives, which ``pareto.crowding_distance``
does not; the comparison scales by that factor. Run it to print the numbers quoted in the solutions::

    python pymoo_sorting_labs.py
"""

import numpy as np
import pareto
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

SEED = 27
N_POINTS = 1000
N_OBJ = 3


# --8<-- [start:compare]
def pymoo_ranks(points: np.ndarray) -> np.ndarray:
    """Front index of every point from pymoo's fast non-dominated sorting (0 = first front)."""
    ranks = np.full(len(points), -1)
    for rank, front in enumerate(NonDominatedSorting(method='fast_non_dominated_sort').do(points)):
        ranks[front] = rank
    return ranks


def pymoo_crowding(points: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Pymoo's crowding distance computed front by front, rescaled by the number of objectives."""
    distance = np.zeros(len(points))
    for rank in np.unique(ranks):
        idx = np.flatnonzero(ranks == rank)
        distance[idx] = calc_crowding_distance(points[idx]) * points.shape[1]
    return distance


def compare(points: np.ndarray) -> dict:
    """Ranks and crowding distances of both implementations, and whether they agree."""
    ours = pareto.fast_non_dominated_sort(points)
    theirs = pymoo_ranks(points)
    crowd_ours = pareto.crowding_distance(points, ours)
    crowd_theirs = pymoo_crowding(points, theirs)
    big = np.isin(ours, np.flatnonzero(np.bincount(ours) > 2))  # fronts of 1-2 points: inf by convention in pareto.py
    return {
        'fronts': int(ours.max() + 1),
        'first_front': int(np.sum(ours == 0)),
        'ranks_equal': bool(np.array_equal(ours, theirs)),
        'crowding_equal': bool(np.allclose(crowd_ours[big], crowd_theirs[big], rtol=1e-12, atol=1e-12)),
        'boundary_first_front': int(np.isinf(crowd_ours[ours == 0]).sum()),
    }


# --8<-- [end:compare]
def main() -> None:
    """Exercise 3 on 1,000 uniform random points in three objectives (seed 27)."""
    import pymoo

    points = np.random.default_rng(SEED).random((N_POINTS, N_OBJ))
    result = compare(points)
    print(f'pymoo {pymoo.__version__}, {N_POINTS} random points in {N_OBJ} objectives (seed {SEED}):')
    print(f'  {result["fronts"]} fronts, {result["first_front"]} points in the first')
    print(f'  ranks identical to pareto.fast_non_dominated_sort: {result["ranks_equal"]}')
    print(f'  crowding distances identical (pymoo x {N_OBJ}): {result["crowding_equal"]}')
    print(f'  boundary points (inf) in the first front: {result["boundary_first_front"]}')


if __name__ == '__main__':
    main()
