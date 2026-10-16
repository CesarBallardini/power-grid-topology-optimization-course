"""Flows of a phase-shifting transformer in the DC model: the two-path worked example and a PSDF column.

Two parallel paths between buses 1 and 2: line a (reactance x_a) with an ideal PST of angle α, and line b (x_b).
F_a = (θ12 + α) / x_a, F_b = θ12 / x_b and F_a + F_b = P give F_a = x_b P / (x_a + x_b) + α / (x_a + x_b): a base
share plus a loop flow proportional to α. The same PST is a pair of injections −α/x_a and +α/x_a at its two
buses, plus α/x_a added back to its own branch, which is how a PSDF column is built on any network.
"""

import numpy as np
from dc_nodal import nodal_solve

X_A, X_B, P = 0.1, 0.3, 1.0  # pu, the worked example

# the Chapter 2 network, conductances read as susceptances, with the Chapter 8 lab injections
FR = np.array([0, 0, 1, 1, 2, 3, 4])
TO = np.array([1, 2, 2, 3, 4, 4, 5])
B = np.array([10.0, 5.0, 8.0, 4.0, 6.0, 3.0, 2.0])
INJ = np.array([0.0, 0.0, 0.5, 0.0, 0.0, -1.0])


def two_path_flows(alpha_deg: np.ndarray | float, p: float = P, x_a: float = X_A, x_b: float = X_B) -> tuple:
    """F_a and F_b of the worked example for a PST angle in degrees, from the closed formula."""
    alpha = np.radians(alpha_deg)
    f_a = x_b * p / (x_a + x_b) + alpha / (x_a + x_b)
    return f_a, p - f_a


def slope_per_degree(x_a: float = X_A, x_b: float = X_B) -> float:
    """Change of F_a per degree of PST angle, (π/180) / (x_a + x_b) pu/degree."""
    return (np.pi / 180) / (x_a + x_b)


def alpha_for_zero_flow_b(p: float = P, x_a: float = X_A) -> float:
    """The angle (degrees) at which line b carries nothing: F_a = P, so θ12 = 0 and α = x_a P."""
    return float(np.degrees(x_a * p))


# --8<-- [start:injections]
def two_path_by_injections(alpha_deg: float, p: float = P, x_a: float = X_A, x_b: float = X_B) -> tuple:
    """F_a and F_b from ``nodal_solve`` with the PST replaced by injections −α/x_a at bus 1 and +α/x_a at bus 2.

    The plain two-line network is solved with the modified injections; the PST's own branch then gets α/x_a
    added back, because the angle across its reactance is θ12 + α, not θ12.
    """
    shift = np.radians(alpha_deg) / x_a
    inj = np.array([p - shift, -p + shift])
    _, flows = nodal_solve([0, 0], [1, 1], [1 / x_a, 1 / x_b], inj, 2)
    return flows[0] + shift, flows[1]


def psdf_column(
    fr: np.ndarray, to: np.ndarray, b: np.ndarray, pst_branch: int, inj: np.ndarray, n: int, step_deg: float = 1.0
) -> np.ndarray:
    """Change of every branch flow per degree of a PST on ``pst_branch``, by finite differences.

    The DC model is linear, so the difference quotient is exact whatever the step and the injections; the
    column is what ToOp's PSDF stores for that transformer (before its base_mva × π/180 scaling).
    """
    shift = np.radians(step_deg) * b[pst_branch]
    inj_pst = np.array(inj, dtype=float)
    inj_pst[fr[pst_branch]] -= shift
    inj_pst[to[pst_branch]] += shift
    _, base = nodal_solve(fr, to, b, inj, n)
    _, shifted = nodal_solve(fr, to, b, inj_pst, n)
    shifted[pst_branch] += shift
    return (shifted - base) / step_deg


# --8<-- [end:injections]


def main() -> None:
    """Print the numbers of the worked example and of lab steps 3 to 5."""
    f_a, f_b = two_path_flows(0.0)
    print(
        f'alpha = 0: F_a = {f_a:.4f}, F_b = {f_b:.4f} pu; slope {slope_per_degree():.4f} pu/deg = '
        f'{100 * slope_per_degree():.2f} MW/deg on 100 MVA'
    )
    for alpha in (-10.0, -5.0, 0.0, 5.0, 10.0):
        f_a, f_b = two_path_flows(alpha)
        g_a, g_b = two_path_by_injections(alpha)
        print(f'alpha = {alpha:+5.1f} deg: F_a = {f_a:.4f}, F_b = {f_b:+.4f} pu  (injections: {g_a:.4f}, {g_b:+.4f})')
    print(f'line b carries nothing at alpha = {alpha_for_zero_flow_b():.2f} deg and reverses beyond it')
    col = psdf_column(FR, TO, B, 3, INJ, 6)
    _, base = nodal_solve(FR, TO, B, INJ, 6)
    print('lab 5, PST on branch (1, 3) of the Chapter 2 network; base flows and change per degree:')
    for k, (f, t) in enumerate(zip(FR, TO, strict=True)):
        print(f'  branch ({f}, {t}): base {base[k]:+.4f}, PSDF {col[k]:+.5f} per degree')
    print(
        f'  check: the column sums to zero at every node except through the PST branch; '
        f'inj change at node 1 = {col[FR == 1].sum() - col[TO == 1].sum():.1e}'
    )


if __name__ == '__main__':
    main()
