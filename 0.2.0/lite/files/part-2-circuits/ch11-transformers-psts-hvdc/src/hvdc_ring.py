"""An HVDC link across a three-bus AC ring as a pair of injections (Chapter 11, Exercise 6).

Buses 1, 2, 3 (nodes 0, 1, 2) form a ring; a generator at bus 1 feeds loads at buses 2 and 3. The HVDC link
withdraws P_dc at bus 1 and injects it at bus 3. In the DC model every AC flow is affine in P_dc,
F(P_dc) = F(0) + P_dc d, where d is the flow pattern of a unit transfer from bus 1 to bus 3; the setpoint that
minimizes the largest AC flow is a one-variable linear program.
"""

import numpy as np
from dc_nodal import nodal_solve
from scipy.optimize import linprog

FR = np.array([0, 1, 0])
TO = np.array([1, 2, 2])
X = np.array([0.10, 0.15, 0.20])  # pu
NAMES = ['L1-2', 'L2-3', 'L1-3']
LOAD = np.array([0.0, 0.6, 0.4])  # pu, at buses 1, 2, 3; the generator at bus 1 balances


def ac_flows(p_dc: float, load: np.ndarray = LOAD) -> np.ndarray:
    """AC branch flows with the HVDC link carrying ``p_dc`` from bus 1 to bus 3 (bus 1 is the slack)."""
    inj = -load.copy()
    inj[0] -= p_dc  # withdrawn by the converter at bus 1 (the slack balances it anyway)
    inj[2] += p_dc  # injected by the converter at bus 3
    _, flows = nodal_solve(FR, TO, 1 / X, inj, 3)
    return flows


def sensitivity() -> np.ndarray:
    """Change of every AC flow per pu of HVDC transfer, d = F(1) − F(0)."""
    return ac_flows(1.0) - ac_flows(0.0)


def best_setpoint() -> tuple[float, float]:
    """The P_dc that minimizes max |F_k(P_dc)| over the AC lines, and that minimum."""
    f0, d = ac_flows(0.0), sensitivity()
    # variables (p_dc, t): minimize t subject to f0 + d p_dc <= t and -(f0 + d p_dc) <= t
    a_ub = np.vstack([np.column_stack([d, -np.ones(3)]), np.column_stack([-d, -np.ones(3)])])
    b_ub = np.concatenate([-f0, f0])
    res = linprog([0.0, 1.0], A_ub=a_ub, b_ub=b_ub, bounds=[(None, None), (0, None)])
    return float(res.x[0]), float(res.x[1])


def main() -> None:
    """Print the base flows, the sensitivities, the best setpoint and a check of affinity."""
    f0, d = ac_flows(0.0), sensitivity()
    for name, f, s in zip(NAMES, f0, d, strict=True):
        print(f'{name}: flow at P_dc = 0 {f:+.4f} pu, change per pu of P_dc {s:+.4f}')
    worst = int(np.argmax(np.abs(f0)))
    print(
        f'most loaded line at P_dc = 0: {NAMES[worst]} ({abs(f0[worst]):.4f} pu); '
        f'its flow is zero at P_dc = {-f0[worst] / d[worst]:.4f} pu'
    )
    p_best, f_max = best_setpoint()
    flows = np.round(ac_flows(p_best), 4)
    print(f'minimax setpoint P_dc = {p_best:.4f} pu, largest AC flow {f_max:.4f} pu; flows {flows}')
    p = np.linspace(-0.5, 1.5, 9)
    affine = np.array([f0 + pk * d for pk in p])
    exact = np.array([ac_flows(pk) for pk in p])
    print(f'affinity check over 9 setpoints: max |exact - affine| = {np.abs(exact - affine).max():.1e}')


if __name__ == '__main__':
    main()
