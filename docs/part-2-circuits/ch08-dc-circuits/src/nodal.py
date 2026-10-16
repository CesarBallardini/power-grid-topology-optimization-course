"""Nodal analysis in matrix form: G = Aᵀ diag(g) A, superposition, Thévenin resistance and a split node (Chapter 8).

``nodal_solve`` is the lab's solver; the other functions apply it to the worked example and to the six-node
resistor network of the Chapter 2 lab (conductances in siemens), restated here as ``FR``, ``TO``, ``G``.
"""

import numpy as np

# the worked example: node 0 is ground, 1 S from node 1 to ground, 2 S between 1 and 2, 4 S from node 2 to ground
EX_FR, EX_TO, EX_G, EX_INJ = [1, 1, 2], [0, 2, 0], [1.0, 2.0, 4.0], [0.0, 3.0, 0.0]

# the Chapter 2 lab network: six nodes, seven branches
FR = [0, 0, 1, 1, 2, 3, 4]
TO = [1, 2, 2, 3, 4, 4, 5]
G = [10.0, 5.0, 8.0, 4.0, 6.0, 3.0, 2.0]
N = 6
INJ = [0.5, 0.0, 0.5, 0.0, 0.0, -1.0]  # 0.5 A into node 2, 1 A out of node 5, ground (node 0) supplies the other 0.5 A


# --8<-- [start:nodal_solve]
def incidence(fr, to, n: int) -> np.ndarray:
    """Branch-node incidence matrix A (branches × nodes): +1 at the from node, −1 at the to node."""
    a = np.zeros((len(fr), n))
    a[np.arange(len(fr)), fr] = 1.0
    a[np.arange(len(fr)), to] = -1.0
    return a


def nodal_solve(fr, to, g, inj, n: int, ground: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Node voltages and branch currents of a resistive network with current injections.

    Builds A and G = Aᵀ diag(g) A, deletes the ground row and column, solves G_r v_r = i_r and returns the
    full voltage vector (0 at ``ground``) with the branch currents g ⊙ (A v), positive from ``fr`` to ``to``.
    The injection at the ground node is ignored: ground supplies whatever balances the others.
    """
    a = incidence(fr, to, n)
    gmat = a.T @ np.diag(g) @ a
    keep = [k for k in range(n) if k != ground]
    v = np.zeros(n)
    v[keep] = np.linalg.solve(gmat[np.ix_(keep, keep)], np.asarray(inj, float)[keep])
    return v, np.asarray(g) * (a @ v)


# --8<-- [end:nodal_solve]


def kcl_residual(fr, to, i_branch, inj, n: int) -> np.ndarray:
    """Net current leaving each node through its branches minus its injection: Aᵀ i − inj (zero except at ground)."""
    return incidence(fr, to, n).T @ i_branch - np.asarray(inj, float)


def power_balance(v, i_branch, inj, g) -> tuple[float, float]:
    """(vᵀ inj, Σ g_k (A v)_k²): the power delivered by the sources and the power dissipated in the branches."""
    return float(v @ np.asarray(inj)), float(np.sum(i_branch**2 / np.asarray(g)))


# --8<-- [start:thevenin]
def thevenin_resistance(fr, to, g, n: int, a: int, b: int) -> float:
    """Thévenin (effective) resistance between nodes a and b: inject 1 A at a, withdraw it at b, read v_a − v_b."""
    inj = np.zeros(n)
    inj[a], inj[b] = 1.0, -1.0
    v, _ = nodal_solve(fr, to, g, inj, n, ground=b)
    return float(v[a] - v[b])


def effective_resistance(fr, to, g, n: int, a: int, b: int) -> float:
    """The Chapter 3 formula (e_a − e_b)ᵀ L⁺ (e_a − e_b) with the pseudoinverse of the full Laplacian."""
    amat = incidence(fr, to, n)
    lap = amat.T @ np.diag(g) @ amat
    e = np.zeros(n)
    e[a], e[b] = 1.0, -1.0
    return float(e @ np.linalg.pinv(lap) @ e)


# --8<-- [end:thevenin]


# --8<-- [start:split]
def split_node_2(coupler_g: float = 1e6) -> dict[str, float]:
    """Split node 2 of the lab network into busbar A (branches (0, 2), (1, 2), the 0.5 A) and busbar B (branch (2, 4)).

    Busbar B is a new node 6; a coupler branch (2, 6) of conductance ``coupler_g`` joins them. Returns the coupler
    current (A to B), the KCL imbalance at A computed from the unsplit solution (flows into A over its branches
    plus the injection at A) and the voltage across the coupler.
    """
    fr, to, g = [*FR, 2], [*TO, 6], [*G, coupler_g]
    fr[FR.index(2)] = 6  # branch (2, 4) now leaves busbar B
    inj = [*INJ, 0.0]
    v, i = nodal_solve(fr, to, g, inj, N + 1)
    _, i0 = nodal_solve(FR, TO, G, INJ, N)
    into_a = sum(i0[k] for k in range(len(FR)) if TO[k] == 2)  # branches (0, 2) and (1, 2) end at A
    return {'coupler': float(i[-1]), 'imbalance_a': float(into_a + INJ[2]), 'coupler_volts': float(v[2] - v[6])}


# --8<-- [end:split]


def superposition_check() -> tuple[np.ndarray, np.ndarray]:
    """Voltages of the lab network with both sources, and the sum of the voltages with each source alone."""
    v_both, _ = nodal_solve(FR, TO, G, INJ, N)
    total = np.zeros(N)
    for k in (2, 5):
        inj = np.zeros(N)
        inj[k] = INJ[k]
        total += nodal_solve(FR, TO, G, inj, N)[0]
    return v_both, total


def main() -> None:
    """Print the numbers quoted in the lab solution and in Exercises 4 and 5."""
    v, i = nodal_solve(EX_FR, EX_TO, EX_G, EX_INJ, 3)
    print('worked example: v =', np.round(v, 4), ' branch currents =', np.round(i, 4))
    v, i = nodal_solve(FR, TO, G, INJ, N)
    print('lab network: v =', np.round(v, 5))
    print('             i =', np.round(i, 5))
    print(
        '             KCL residual =',
        np.round(kcl_residual(FR, TO, i, INJ, N), 12),
        '(the ground entry is not enforced)',
    )
    sources, branches = power_balance(v, i, INJ, G)
    print(f'             power: sources {sources:.6f} W, branches {branches:.6f} W')
    v_both, v_sum = superposition_check()
    print(f'superposition: max |v_both - v_sum| = {np.max(np.abs(v_both - v_sum)):.1e}')
    rth, reff = thevenin_resistance(FR, TO, G, N, 5, 0), effective_resistance(FR, TO, G, N, 5, 0)
    print(f'Thevenin resistance 5-0: {rth:.5f} ohm; effective resistance (pseudoinverse): {reff:.5f} ohm')
    for cg in (1e6, 1e3, 10.0):
        s = split_node_2(cg)
        print(
            f'coupler {cg:g} S: coupler current {s["coupler"]:.5f} A, '
            f'unsplit imbalance at A {s["imbalance_a"]:.5f} A, coupler voltage {s["coupler_volts"]:.2e} V'
        )
    for ground in (0, 3):
        vg, ig = nodal_solve(FR, TO, G, INJ, N, ground=ground)
        print(f'ground {ground}: v = {np.round(vg, 5)}  max |i - i_ground0| = {np.max(np.abs(ig - i)):.1e}')


if __name__ == '__main__':
    main()
