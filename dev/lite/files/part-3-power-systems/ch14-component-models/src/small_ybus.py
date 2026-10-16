"""Ybus of a three-bus ring by inspection and the branch matrix of a tapped transformer (Chapter 14, Exercises 1–2).

By inspection, the diagonal entry of a bus is the sum of the admittances of everything connected to it (series
admittances of its lines plus half the charging of each line plus shunts) and the off-diagonal entry between two
buses is minus the series admittance of the line joining them. For a transformer with ratio tau at the from end,
Ybr = [[ys/tau², −ys/tau], [−ys/tau, ys]], whose pi equivalent has series admittance ys/tau, a from-side shunt of
ys (1/tau² − 1/tau) and a to-side shunt of ys (1 − 1/tau).
"""

import numpy as np

RING_LINES = [(0, 1, 0.1j), (1, 2, 0.2j), (0, 2, 0.25j)]  # (from, to, series impedance in p.u.)
RING_BC = 0.02  # total charging susceptance of each line, p.u.


def ring3_ybus(lines: list[tuple[int, int, complex]] = RING_LINES, bc: float = RING_BC) -> np.ndarray:
    """Ybus of the three-bus ring, filled in by inspection line by line."""
    ybus = np.zeros((3, 3), dtype=complex)
    for f, t, z in lines:
        ys = 1 / z
        ybus[f, f] += ys + 1j * bc / 2
        ybus[t, t] += ys + 1j * bc / 2
        ybus[f, t] -= ys
        ybus[t, f] -= ys
    return ybus


def row_sums(ybus: np.ndarray) -> np.ndarray:
    """The sum of each row: with no shunt elements it is the total charging admittance connected to the bus."""
    return ybus.sum(axis=1)


def transformer_ybr(tau: float, x: float, shift_deg: float = 0.0) -> np.ndarray:
    """Branch admittance matrix of a transformer with ratio tau·exp(j·shift) at the from end and reactance x."""
    ys = 1 / (1j * x)
    tap = tau * np.exp(1j * np.deg2rad(shift_deg))
    return np.array([[ys / (tap * np.conj(tap)), -ys / np.conj(tap)], [-ys / tap, ys]])


def pi_equivalent(ybr: np.ndarray) -> tuple[complex, complex, complex]:
    """Series admittance, from-side shunt and to-side shunt of the pi circuit with the same branch matrix.

    Valid when the two off-diagonal entries are equal (no phase shift): a pi circuit cannot represent a phase
    shifter, whose Ybr is not symmetric.
    """
    series = -ybr[0, 1]
    return series, ybr[0, 0] - series, ybr[1, 1] - series


def main() -> None:
    """Print the ring's Ybus with its row sums and the transformer's Ybr with its pi equivalent."""
    np.set_printoptions(precision=4, suppress=True)
    ybus = ring3_ybus()
    print('three-bus ring Ybus =')
    print(ybus)
    print('row sums =', row_sums(ybus), '  (expected j bc = 0.02j at every bus)')
    ybr = transformer_ybr(1.05, 0.1)
    print('transformer tau = 1.05, x = 0.1: Ybr =')
    print(ybr)
    series, shunt_from, shunt_to = pi_equivalent(ybr)
    print(f'pi equivalent: series {series:.4f}, shunt at from {shunt_from:.4f}, shunt at to {shunt_to:.4f}')


if __name__ == '__main__':
    main()
