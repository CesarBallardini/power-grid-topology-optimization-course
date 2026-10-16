"""Chapter 27, Exercise 4 with moocore: the rectangle-sum hypervolume of ``pareto.py`` against moocore's.

Runs in the labs container (``make test-labs``, see ``docker/labs``), where moocore is installed. Both minimize
every objective. Run it to print the numbers quoted in the solutions and save nothing::

    python moocore_hypervolume_labs.py
"""

import moocore
import numpy as np
import pareto

FACTORS = (1.0, 1.1, 2.0)  # reference point = factor x nadir, as the exercise asks


# --8<-- [start:compare]
def reference_sweep(points: np.ndarray, factors=FACTORS) -> list[tuple[float, float, float]]:
    """``(factor, rectangles, moocore)`` hypervolumes for the reference point ``factor`` times the nadir."""
    nadir = points[pareto.non_dominated_mask(points)].max(axis=0)
    return [
        (float(f), pareto.hypervolume_2d(points, f * nadir), float(moocore.hypervolume(points, ref=f * nadir)))
        for f in factors
    ]


def random_fronts(n_fronts: int = 200, seed: int = 27) -> float:
    """Largest absolute difference between both implementations on random two-objective point sets."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_fronts):
        points = rng.random((int(rng.integers(1, 40)), 2))
        reference = 1 + rng.random(2)
        worst = max(worst, abs(pareto.hypervolume_2d(points, reference) - moocore.hypervolume(points, ref=reference)))
    return worst


# --8<-- [end:compare]
def main() -> None:
    """Exercise 4 on the Exercise 1 points, and a check on random fronts."""
    print(f'moocore {moocore.__version__}, Exercise 1 points {pareto.POINTS.tolist()}:')
    for factor, ours, theirs in reference_sweep(pareto.POINTS):
        print(f'  reference {factor} x nadir: rectangles {ours:.4f}, moocore {theirs:.4f}')
    curve = reference_sweep(pareto.POINTS, np.linspace(1, 2, 11))
    print('  HV against the factor 1.0, 1.1, ..., 2.0:', [round(hv, 2) for _, hv, _ in curve])
    print(f'200 random point sets, largest |rectangles - moocore|: {random_fronts():.2e}')


if __name__ == '__main__':
    main()
