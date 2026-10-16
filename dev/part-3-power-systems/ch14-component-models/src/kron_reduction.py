"""Kron reduction of the IEEE 14 admittance matrix onto a set of kept buses (Chapter 14, Exercise 4).

Partition the buses into kept (k) and eliminated (e). With zero current injection at the eliminated buses the
nodal equations I = Y V reduce to I_k = (Y_kk − Y_ke Y_ee⁻¹ Y_ek) V_k: the Schur complement is the Ward-type
equivalent, an admittance matrix on the kept buses alone that gives the same kept-bus voltages for any such
injection pattern. Eliminated buses leave fictitious branches between every pair of kept buses they connected, so the
equivalent fills in where the original was sparse.
"""

import numpy as np
from ybus_numpy import build_ybus, case14_ppc

KEEP = np.array([0, 1, 2, 3, 4])  # buses 1–5 (1-based): the 132 kV part of IEEE 14 and the transformer buses


def kron_reduce(ybus: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Schur complement Y_kk − Y_ke Y_ee⁻¹ Y_ek of the admittance matrix onto the kept buses."""
    eliminate = np.setdiff1d(np.arange(ybus.shape[0]), keep)
    y_kk = ybus[np.ix_(keep, keep)]
    y_ke = ybus[np.ix_(keep, eliminate)]
    y_ek = ybus[np.ix_(eliminate, keep)]
    y_ee = ybus[np.ix_(eliminate, eliminate)]
    return y_kk - y_ke @ np.linalg.solve(y_ee, y_ek)


def kept_voltages_full(ybus: np.ndarray, keep: np.ndarray, injection_kept: np.ndarray) -> np.ndarray:
    """Kept-bus voltages from the full network, with the given currents at the kept buses and zero elsewhere."""
    current = np.zeros(ybus.shape[0], dtype=complex)
    current[keep] = injection_kept
    return np.linalg.solve(ybus, current)[keep]


def kept_voltages_reduced(ybus: np.ndarray, keep: np.ndarray, injection_kept: np.ndarray) -> np.ndarray:
    """Kept-bus voltages from the Kron-reduced matrix alone."""
    return np.linalg.solve(kron_reduce(ybus, keep), injection_kept)


def main() -> None:
    """Print the reduced matrix of IEEE 14 onto buses 1–5 and the voltage check."""
    np.set_printoptions(precision=3, suppress=True, linewidth=119)
    ppc = case14_ppc()
    ybus, _, _ = build_ybus(ppc['bus'], ppc['branch'], ppc['baseMVA'])
    reduced = kron_reduce(ybus, KEEP)
    print(f'kept buses (1-based): {KEEP + 1}, eliminated: {np.setdiff1d(np.arange(14), KEEP) + 1}')
    print(
        'nonzeros: full block Y_kk', np.count_nonzero(ybus[np.ix_(KEEP, KEEP)]), ', reduced', np.count_nonzero(reduced)
    )
    print('reduced matrix (imaginary part):')
    print(reduced.imag)
    print('new coupling between buses 4 and 5 (zero in Y_kk):', f'{ybus[3, 4]:.3f} -> {reduced[3, 4]:.3f}')
    rng = np.random.default_rng(14)
    injection = rng.normal(size=len(KEEP)) + 1j * rng.normal(size=len(KEEP))
    v_full = kept_voltages_full(ybus, KEEP, injection)
    v_reduced = kept_voltages_reduced(ybus, KEEP, injection)
    print(f'max |V_full - V_reduced| at the kept buses = {np.abs(v_full - v_reduced).max():.2e}')
    with_injection_elsewhere = np.zeros(14, dtype=complex)
    with_injection_elsewhere[KEEP] = injection
    with_injection_elsewhere[8] = 1.0  # one ampere at bus 9: the equivalent no longer applies
    v_bad = np.linalg.solve(ybus, with_injection_elsewhere)[KEEP]
    print(f'with a current at an eliminated bus the mismatch is {np.abs(v_bad - v_reduced).max():.3f}')


if __name__ == '__main__':
    main()
