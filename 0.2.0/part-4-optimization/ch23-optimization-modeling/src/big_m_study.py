"""Strong against big-M formulation on random facility location instances (Chapter 23, Lab step 3).

Sites and customers are random points in the unit square; the assignment cost is the distance times 10 and the
fixed costs are drawn between 3 and 8, so that a handful of facilities opens. Every instance is solved four times
with HiGHS: LP relaxation and MILP of both formulations, with the solve time and the branch-and-bound node count.
"""

import time

import numpy as np
import pandas as pd
import pulp
from facility_location import build

SIZES = ((10, 30), (20, 60), (30, 90), (40, 120))  # (sites, customers)


def instance(n_sites: int, n_customers: int, seed: int = 23) -> tuple[dict, dict]:
    """Fixed costs ``f`` and assignment costs ``c`` of one random instance."""
    rng = np.random.default_rng(seed + n_sites)
    sites, customers = rng.random((n_sites, 2)), rng.random((n_customers, 2))
    f = {i: float(v) for i, v in enumerate(rng.uniform(3.0, 8.0, n_sites))}
    dist = np.linalg.norm(sites[:, None, :] - customers[None, :, :], axis=2)
    c = {(i, j): float(10.0 * dist[i, j]) for i in range(n_sites) for j in range(n_customers)}
    return f, c


def timed_solve(strong: bool, relax: bool, f: dict, c: dict) -> tuple[float, float, int]:
    """Objective, wall-clock seconds and HiGHS' branch-and-bound node count (-1 for an LP)."""
    m, _ = build(strong, relax, f, c)
    start = time.perf_counter()
    m.solve(pulp.HiGHS(msg=False))
    seconds = time.perf_counter() - start
    return pulp.value(m.objective), seconds, int(m.solverModel.getInfo().mip_node_count)


def study(sizes: tuple[tuple[int, int], ...] = SIZES) -> pd.DataFrame:
    """One row per instance and formulation: relaxation value, MILP optimum, gap, seconds and nodes."""
    rows = []
    for n_sites, n_customers in sizes:
        f, c = instance(n_sites, n_customers)
        for strong in (True, False):
            lp, lp_seconds, _ = timed_solve(strong, True, f, c)
            milp, milp_seconds, nodes = timed_solve(strong, False, f, c)
            rows.append(
                {
                    'sites': n_sites,
                    'customers': n_customers,
                    'formulation': 'strong' if strong else 'big-M',
                    'lp_relaxation': round(lp, 2),
                    'milp': round(milp, 2),
                    'gap_%': round(100 * (milp - lp) / milp, 1),
                    'lp_seconds': round(lp_seconds, 3),
                    'milp_seconds': round(milp_seconds, 3),
                    'nodes': nodes,
                }
            )
    return pd.DataFrame(rows).set_index(['sites', 'customers', 'formulation'])


def main() -> None:
    """The table of the Lab; the plot of time against size is left to the notebook."""
    pd.set_option('display.width', 200)
    print(study().to_string())


if __name__ == '__main__':
    main()
