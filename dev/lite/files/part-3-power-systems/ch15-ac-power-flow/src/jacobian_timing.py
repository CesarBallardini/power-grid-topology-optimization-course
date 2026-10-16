"""Polar Jacobian of IEEE 14 and IEEE 118 against finite differences; dense versus sparse (Chapter 15, Exercise 3).

The bus admittance matrix comes from pandapower's internal ``Ybus`` after ``runpp``; the state is the converged
one, read from the same internal structure.
"""

import timeit

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import scipy.sparse as sps
from ac_jacobian import injections, jacobian_blocks


def case_state(name: str) -> tuple[np.ndarray, np.ndarray, sps.csr_matrix]:
    """Converged angles (rad), magnitudes (pu) and the sparse Ybus (pu) of a pandapower network case."""
    net = getattr(pn, name)()
    pp.runpp(net)
    v = np.asarray(net._ppc['internal']['V'])
    return np.angle(v), np.abs(v), sps.csr_matrix(net._ppc['internal']['Ybus'])


def jacobian_blocks_sparse(theta: np.ndarray, vm: np.ndarray, y: sps.spmatrix) -> tuple[sps.csr_matrix, ...]:
    """The same four blocks as ``ac_jacobian.jacobian_blocks``, with sparse diagonals and a sparse Ybus."""
    v = vm * np.exp(1j * theta)
    i = y @ v
    diag_v, diag_i, e = sps.diags(v), sps.diags(i), sps.diags(v / vm)
    ds_dth = 1j * diag_v @ (diag_i - y @ diag_v).conj()
    ds_dvm = diag_v @ (y @ e).conj() + diag_i.conj() @ e
    return ds_dth.real.tocsr(), ds_dvm.real.tocsr(), ds_dth.imag.tocsr(), ds_dvm.imag.tocsr()


def finite_difference_error(theta: np.ndarray, vm: np.ndarray, y: np.ndarray, h: float = 1e-6) -> float:
    """Largest difference between the analytic blocks and central differences of ``injections``, column by column."""
    j11, j12, j21, j22 = jacobian_blocks(theta, vm, y)
    n = len(theta)
    worst = 0.0
    for k in range(n):
        d = np.zeros(n)
        d[k] = h
        ds = (injections(theta + d, vm, y) - injections(theta - d, vm, y)) / (2 * h)
        worst = max(worst, np.abs(j11[:, k] - ds.real).max(), np.abs(j21[:, k] - ds.imag).max())
        ds = (injections(theta, vm + d, y) - injections(theta, vm - d, y)) / (2 * h)
        worst = max(worst, np.abs(j12[:, k] - ds.real).max(), np.abs(j22[:, k] - ds.imag).max())
    return worst


def time_both(theta: np.ndarray, vm: np.ndarray, y: sps.csr_matrix, number: int = 50) -> tuple[float, float]:
    """Milliseconds per evaluation of the dense and of the sparse Jacobian blocks (best of three runs)."""
    y_dense = y.toarray()
    dense = min(timeit.repeat(lambda: jacobian_blocks(theta, vm, y_dense), number=number, repeat=3)) / number
    sparse = min(timeit.repeat(lambda: jacobian_blocks_sparse(theta, vm, y), number=number, repeat=3)) / number
    return 1e3 * dense, 1e3 * sparse


def main() -> None:
    """Finite-difference error and timings for IEEE 14 and IEEE 118."""
    for name in ('case14', 'case118'):
        theta, vm, y = case_state(name)
        err = finite_difference_error(theta, vm, y.toarray())
        dense, sparse = time_both(theta, vm, y)
        print(f'{name}: {len(theta)} buses, {y.nnz} nonzeros in Ybus, max |analytic - central difference| = {err:.1e}')
        print(f'  dense blocks {dense:.3f} ms, sparse blocks {sparse:.3f} ms per evaluation')


if __name__ == '__main__':
    main()
