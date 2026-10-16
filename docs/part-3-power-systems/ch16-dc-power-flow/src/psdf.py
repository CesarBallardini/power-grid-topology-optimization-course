"""Phase shift distribution factors from DC power flows (Chapter 16, Exercise 5).

ToOp's convention: the flow on branch k is b_k (theta_f - theta_t + phi_k), so the angle phi_k (rad) acts as the
injections -b_k phi_k at f_k and +b_k phi_k at t_k plus a direct flow b_k phi_k on branch k itself, and
PSDF[l, k] = b_k (delta_lk - PTDF[l, f_k] + PTDF[l, t_k]).
"""

import numpy as np
from dc_power_flow import dc_power_flow


def ptdf_from_solves(frm: np.ndarray, to: np.ndarray, x: np.ndarray, n_bus: int, slack: int) -> np.ndarray:
    """PTDF (branches x buses), one DC power flow per column: a unit injected at a bus and withdrawn at the slack."""
    h = np.zeros((len(frm), n_bus))
    for i in range(n_bus):
        if i == slack:
            continue
        p = np.zeros(n_bus)
        p[i], p[slack] = 1.0, -1.0
        h[:, i] = dc_power_flow(frm, to, x, p, slack)
    return h


def psdf_matrix(h: np.ndarray, frm: np.ndarray, to: np.ndarray, x: np.ndarray) -> np.ndarray:
    """PSDF (branches x branches) in flow units per radian of shift, ToOp's sign convention."""
    b = 1 / x
    return b * (np.eye(len(frm)) - h[:, frm] + h[:, to])


def flows_with_shift(
    frm: np.ndarray, to: np.ndarray, x: np.ndarray, p: np.ndarray, slack: int, phi: np.ndarray
) -> np.ndarray:
    """DC flows with shift angles ``phi`` (rad, one per branch) modelled as equivalent injections, ToOp's sign."""
    b = 1 / x
    p_eq = p.copy()
    np.add.at(p_eq, frm, -b * phi)
    np.add.at(p_eq, to, b * phi)
    return dc_power_flow(frm, to, x, p_eq, slack) + b * phi


def main() -> None:
    """Four-bus ring in pu: PSDF column of branch 0 against a DC power flow with a 5-degree shift on that branch."""
    frm, to = np.array([0, 0, 1, 2]), np.array([1, 2, 3, 3])
    x = np.array([0.1, 0.2, 0.1, 0.2])
    p = np.array([0.8, 0.2, -0.5, -0.5])
    base_mva = 100.0
    h = ptdf_from_solves(frm, to, x, 4, slack=0)
    psdf = psdf_matrix(h, frm, to, x)
    print('PSDF (pu per radian), columns = shifted branch; 1 / (sum of x around the ring) = 1 / 0.6:')
    print(np.round(psdf, 4))
    phi = np.zeros(4)
    phi[0] = np.deg2rad(5.0)
    base = dc_power_flow(frm, to, x, p, 0)
    shifted = flows_with_shift(frm, to, x, p, 0, phi)
    print('base flows (pu)       :', np.round(base, 4))
    print('with 5 deg on branch 0:', np.round(shifted, 4))
    print('PSDF prediction       :', np.round(base + psdf[:, 0] * phi[0], 4))
    print(f'max |prediction - power flow| = {np.abs(base + psdf @ phi - shifted).max():.1e}')
    print(
        'MW per degree, as compute_psdf scales it (base_mva * pi / 180):',
        np.round(psdf[:, 0] * base_mva * np.pi / 180, 4),
    )


if __name__ == '__main__':
    main()
