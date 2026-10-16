"""Ybus of IEEE 14 from the connection matrices, checked against pandapower's makeYbus (Chapter 14, Lab).

Every branch is a pi section with series admittance ys = 1/(r + jx), total charging bc and an ideal transformer of
complex ratio tap = tau·exp(j·shift) at the from end. Its four entries are Yff = (ys + j·bc/2)/|tap|²,
Yft = −ys/conj(tap), Ytf = −ys/tap and Ytt = ys + j·bc/2. With Cf and Ct marking the from and to bus of each
branch, Yf = diag(Yff) Cf + diag(Yft) Ct, Yt = diag(Ytf) Cf + diag(Ytt) Ct and Ybus = Cfᵀ Yf + Ctᵀ Yt + diag(Ysh).
pandapower keeps a MATPOWER-style ``ppc`` after a power flow; it stores TAP = 1 for lines where MATPOWER files
write 0, so the tap column can be used as it is.
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, SHIFT, T_BUS, TAP
from pandapower.pypower.idx_bus import BS, GS
from pandapower.pypower.makeYbus import makeYbus


def case14_ppc() -> dict:
    """The internal MATPOWER-style case of pandapower's IEEE 14, populated by one power flow."""
    net = pn.case14()
    pp.runpp(net)
    return net._ppc  # noqa: SLF001  (pandapower exposes the ppc only as a private attribute)


def connection_matrices(branch: np.ndarray, n_bus: int) -> tuple[np.ndarray, np.ndarray]:
    """Cf and Ct, shape (branches, buses): a one in the column of the from bus and of the to bus of each branch."""
    n_branch = branch.shape[0]
    cf, ct = np.zeros((n_branch, n_bus)), np.zeros((n_branch, n_bus))
    cf[np.arange(n_branch), branch[:, F_BUS].real.astype(int)] = 1
    ct[np.arange(n_branch), branch[:, T_BUS].real.astype(int)] = 1
    return cf, ct


def branch_admittances(branch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four entries Yff, Yft, Ytf, Ytt of every branch, out-of-service branches set to zero."""
    status = branch[:, BR_STATUS].real
    ys = status / (branch[:, BR_R] + 1j * branch[:, BR_X])
    bc = status * branch[:, BR_B].real
    tau = np.where(branch[:, TAP].real == 0, 1.0, branch[:, TAP].real)
    tap = tau * np.exp(1j * np.pi / 180 * branch[:, SHIFT].real)
    ytt = ys + 1j * bc / 2
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    return yff, yft, ytf, ytt


def build_ybus(bus: np.ndarray, branch: np.ndarray, base_mva: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ybus, Yf and Yt (dense) assembled as Ybus = Cfᵀ Yf + Ctᵀ Yt + diag(Ysh)."""
    n_bus = bus.shape[0]
    cf, ct = connection_matrices(branch, n_bus)
    yff, yft, ytf, ytt = branch_admittances(branch)
    yf = np.diag(yff) @ cf + np.diag(yft) @ ct
    yt = np.diag(ytf) @ cf + np.diag(ytt) @ ct
    ysh = (bus[:, GS] + 1j * bus[:, BS]) / base_mva
    return cf.T @ yf + ct.T @ yt + np.diag(ysh), yf, yt


def max_difference(ppc: dict) -> float:
    """Largest absolute entry of Ybus (this module) minus Ybus (pandapower's makeYbus)."""
    ybus, _, _ = build_ybus(ppc['bus'], ppc['branch'], ppc['baseMVA'])
    reference, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])
    return float(np.abs(ybus - reference.toarray()).max())


def transformer_branches(branch: np.ndarray) -> np.ndarray:
    """Indices of the branches with an off-nominal tap or a phase shift."""
    tap, shift = branch[:, TAP].real, branch[:, SHIFT].real
    return np.flatnonzero(((tap != 0) & (tap != 1)) | (shift != 0))


def asymmetry(matrix: np.ndarray) -> float:
    """Frobenius norm of Y − Yᵀ, zero for a symmetric matrix."""
    return float(np.linalg.norm(matrix - matrix.T))


def with_shift(branch: np.ndarray, index: int, degrees: float) -> np.ndarray:
    """A copy of the branch table with a phase shift (degrees) on one branch."""
    shifted = branch.copy()
    shifted[index, SHIFT] = degrees
    return shifted


def main() -> None:
    """Print the numbers of Lab steps 1–4 for IEEE 14."""
    ppc = case14_ppc()
    bus, branch, base_mva = ppc['bus'], ppc['branch'], ppc['baseMVA']
    ybus, _, _ = build_ybus(bus, branch, base_mva)
    print(f'{bus.shape[0]} buses, {branch.shape[0]} branches, baseMVA {base_mva}')
    print(f'max |Ybus - makeYbus| = {max_difference(ppc):.3e}')
    nnz = int(np.count_nonzero(ybus))
    print(f'nonzeros {nnz} of {ybus.size} ({100 * nnz / ybus.size:.0f} %)')
    print(f'  {bus.shape[0]} diagonal + 2 x {(nnz - bus.shape[0]) // 2} off-diagonal')
    trafos = transformer_branches(branch)
    for k in trafos:
        f, t = int(branch[k, F_BUS].real), int(branch[k, T_BUS].real)
        tap, x = branch[k, TAP].real, branch[k, BR_X].real
        print(f'transformer branch {k}: buses {f + 1}-{t + 1} (1-based), tap {tap:.4f}, x {x:.5f}')
    print(f'||Ybus - Ybus^T|| = {asymmetry(ybus):.3e} (symmetric despite the taps)')
    shifted, _, _ = build_ybus(bus, with_shift(branch, int(trafos[0]), 5.0), base_mva)
    f, t = int(branch[trafos[0], F_BUS].real), int(branch[trafos[0], T_BUS].real)
    print(f'with a 5 degree shift on branch {trafos[0]}: ||Ybus - Ybus^T|| = {asymmetry(shifted):.4f}')
    print(f'  Y[{f + 1},{t + 1}] = {shifted[f, t]:.4f}, Y[{t + 1},{f + 1}] = {shifted[t, f]:.4f}')
    print(f'  same modulus {abs(shifted[f, t]):.4f}')


if __name__ == '__main__':
    main()
