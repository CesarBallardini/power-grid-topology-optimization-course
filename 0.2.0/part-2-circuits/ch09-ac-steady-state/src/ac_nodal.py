"""Complex nodal analysis of a three-node AC circuit and its complex power balance (Chapter 9).

Node 0 is ground. A voltage source holds node 1 at V1, a current source injects I2 into node 2, an RL branch
joins nodes 1 and 2 and an RC branch joins node 2 to ground. Y V = I with complex admittances: the row of the
node with a known voltage is dropped and its column moves to the right-hand side. The power injected by the two
sources, Σ V_k I_k*, equals the power absorbed by the branches, Σ |I_b|² Z_b.
"""

import cmath
import math

import numpy as np

FR = np.array([1, 2])  # branch from-nodes: RL from 1 to 2, RC from 2 to ground
TO = np.array([2, 0])
Z_BRANCH = np.array([2.0 + 5.0j, 10.0 - 4.0j])  # ohm, at the supply frequency
V_SOURCE = 230.0 + 0.0j  # V, node 1
I_SOURCE = 5.0 * cmath.exp(-1j * math.radians(30.0))  # A injected into node 2


def admittance_matrix(fr: np.ndarray, to: np.ndarray, z: np.ndarray, n: int) -> np.ndarray:
    """Nodal admittance matrix Y = Aᵀ diag(1/z) A over ``n`` nodes, ground included."""
    a = np.zeros((len(fr), n), dtype=complex)
    a[np.arange(len(fr)), fr] = 1.0
    a[np.arange(len(fr)), to] = -1.0
    return a.T @ np.diag(1 / z) @ a


def solve(fr: np.ndarray, to: np.ndarray, z: np.ndarray, v1: complex, i2: complex) -> tuple[np.ndarray, np.ndarray]:
    """Node voltages (ground, V1, V2), the source currents (I1 from the voltage source, I2) and branch currents."""
    y = admittance_matrix(fr, to, z, 3)
    v = np.array([0.0, v1, 0.0], dtype=complex)
    v[2] = (i2 - y[2, 1] * v1) / y[2, 2]  # KCL at node 2 with V1 known
    inj = y @ v  # net injections at every node; inj[1] is the current of the voltage source
    return v, inj, (v[fr] - v[to]) / z


def power_balance(v: np.ndarray, inj: np.ndarray, i_branch: np.ndarray, z: np.ndarray) -> tuple[complex, complex]:
    """Complex power injected by the sources and absorbed by the branches; the two are equal (Tellegen)."""
    injected = complex(np.sum(v[1:] * np.conj(inj[1:])))
    absorbed = complex(np.sum(np.abs(i_branch) ** 2 * z))
    return injected, absorbed


def main() -> None:
    """Print the voltages, source and branch currents and the two sides of the power balance of Exercise 5."""
    v, inj, i_branch = solve(FR, TO, Z_BRANCH, V_SOURCE, I_SOURCE)
    print(f'V2 = {abs(v[2]):.2f} V at {math.degrees(cmath.phase(v[2])):.1f} deg')
    print(f'I1 (voltage source) = {abs(inj[1]):.2f} A at {math.degrees(cmath.phase(inj[1])):.1f} deg')
    for name, i in zip(('RL', 'RC'), i_branch, strict=True):
        print(f'I_{name} = {abs(i):.2f} A at {math.degrees(cmath.phase(i)):.1f} deg')
    injected, absorbed = power_balance(v, inj, i_branch, Z_BRANCH)
    print(f'injected  S = {injected.real:.1f} {injected.imag:+.1f}j VA')
    print(f'absorbed  S = {absorbed.real:.1f} {absorbed.imag:+.1f}j VA')


if __name__ == '__main__':
    main()
