"""The Wheatstone bridge by nodal analysis and a numeric check of Tellegen's theorem (Chapter 8, Exercises 1 and 3).

Bridge nodes: 0 is ground (bottom), 1 the top, 2 the left and 3 the right corner. R1 joins 1–2, R2 joins 1–3,
R3 joins 2–0, R4 joins 3–0 and R5 is the middle branch 2–3; a current source feeds node 1. The middle current
vanishes when R1 R4 = R2 R3.
"""

import numpy as np
from nodal import FR, INJ, TO, G, N, incidence, nodal_solve

BRIDGE_FR = [1, 1, 2, 3, 2]
BRIDGE_TO = [2, 3, 0, 0, 3]


# --8<-- [start:bridge]
def bridge_currents(r: list[float], i_src: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Node voltages and branch currents (R1, R2, R3, R4, R5 order) of the bridge fed by ``i_src`` at node 1."""
    g = 1 / np.asarray(r, float)
    return nodal_solve(BRIDGE_FR, BRIDGE_TO, g, [-i_src, i_src, 0.0, 0.0], 4)


def is_balanced(r: list[float]) -> bool:
    """R1 R4 = R2 R3, the condition for zero current in the middle branch."""
    r1, r2, r3, r4, _ = r
    return bool(np.isclose(r1 * r4, r2 * r3))


# --8<-- [end:bridge]


# --8<-- [start:tellegen]
def tellegen_check(fr, to, g, inj, n: int, g_other) -> tuple[float, float]:
    """uᵀ j against vᵀ i, with u = A v from the network ``g`` and j the branch currents of a network ``g_other``.

    Both networks share the graph and the injections, so j satisfies Aᵀ j = i although it belongs to another
    circuit; Tellegen's theorem says the two numbers agree anyway.
    """
    v, _ = nodal_solve(fr, to, g, inj, n)
    _, j = nodal_solve(fr, to, g_other, inj, n)
    u = incidence(fr, to, n) @ v
    return float(u @ j), float(v @ np.asarray(inj, float))


# --8<-- [end:tellegen]


def main() -> None:
    """Print the numbers quoted in the solutions of Exercises 1 and 3."""
    for r in ([100.0, 200.0, 50.0, 100.0, 1000.0], [100.0, 200.0, 50.0, 120.0, 1000.0]):
        v, i = bridge_currents(r)
        print(f'bridge R = {r}: balanced = {is_balanced(r)}, v2 - v3 = {v[2] - v[3]:.5f} V, middle {i[4]:.6f} A')
    g_other = [1.0] * len(G)
    lhs, rhs = tellegen_check(FR, TO, G, INJ, N, g_other)
    print(f'Tellegen: u^T j = {lhs:.6f} W, v^T i = {rhs:.6f} W (j from a unit-conductance network on the same graph)')
    v, i = nodal_solve(FR, TO, G, INJ, N)
    dissipated = np.sum(i**2 / np.asarray(G))
    print(f'power balance (same network): v^T i = {v @ np.asarray(INJ):.6f} W, sum g (Av)^2 = {dissipated:.6f} W')


if __name__ == '__main__':
    main()
