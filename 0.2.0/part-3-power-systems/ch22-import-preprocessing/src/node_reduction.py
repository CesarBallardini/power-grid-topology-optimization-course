"""Node reduction of a DC PTDF: keep the significant buses, collapse the rest into one column (Chapter 22, Exercise 2).

A ten-bus grid with unit reactances on a ring plus two chords. ToOp's ``reduce_ptdf_and_nodal_injections`` keeps
the columns of the significant buses and replaces all other columns by their flow contribution at the reference
injections, a single column with a unit injection; the branch flows are exact as long as those injections do not
change.
"""

import numpy as np

BRANCHES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0), (0, 5), (2, 7)]
N_BUS = 10
P_REF = np.array([120.0, -30.0, 40.0, -50.0, -20.0, 60.0, -35.0, -45.0, 30.0, -70.0])  # MW, sums to zero
SIGNIFICANT = [0, 2, 5, 7]  # the slack and the three stations ToOp may switch


def ptdf_matrix(branches: list[tuple[int, int]], n_bus: int, slack: int = 0) -> np.ndarray:
    """Nodal PTDF (branches x buses) of a grid of unit reactances."""
    a = np.zeros((len(branches), n_bus))
    for k, (f, t) in enumerate(branches):
        a[k, f], a[k, t] = 1.0, -1.0
    keep = np.arange(n_bus) != slack
    h = np.zeros_like(a)
    h[:, keep] = a[:, keep] @ np.linalg.inv((a.T @ a)[np.ix_(keep, keep)])
    return h


def reduce_ptdf(h: np.ndarray, p: np.ndarray, significant: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Reduced PTDF (branches x (significant + 1)) and injections; the last column carries the collapsed buses."""
    insignificant = [i for i in range(h.shape[1]) if i not in significant]
    reduced_column = h[:, insignificant] @ p[insignificant]
    h_red = np.column_stack([h[:, significant], reduced_column])
    p_red = np.append(p[significant], 1.0)  # unit injection on the REDUCED_NODE column
    return h_red, p_red


def main() -> None:
    """Exact at the reference injections, wrong after a change at a collapsed bus, exact after one at a kept bus."""
    h = ptdf_matrix(BRANCHES, N_BUS)
    h_red, p_red = reduce_ptdf(h, P_REF, SIGNIFICANT)
    full, reduced = h @ P_REF, h_red @ p_red
    print(
        f'PTDF {h.shape} -> reduced {h_red.shape}; max error at the reference: {np.abs(full - reduced).max():.1e} MW'
    )
    p_new = P_REF.copy()
    p_new[3] -= 20.0  # a collapsed bus: 20 MW more load at bus 3 ...
    p_new[0] += 20.0  # ... covered by the slack, which is significant
    reduced_new = h_red @ np.append(p_new[SIGNIFICANT], 1.0)
    print(f'after +20 MW load at collapsed bus 3: max error {np.abs(h @ p_new - reduced_new).max():.1f} MW')
    p_new = P_REF.copy()
    p_new[2] += 20.0  # a significant bus
    p_new[0] -= 20.0
    reduced_new = h_red @ np.append(p_new[SIGNIFICANT], 1.0)
    print(f'after +20 MW at significant bus 2: max error {np.abs(h @ p_new - reduced_new).max():.1e} MW')


if __name__ == '__main__':
    main()
