"""A 3-bus DC-OPF where opening a line lowers the cost, and the big-M of a line on a cycle (Chapter 28, Exercises 1-2).

Bus 1 has a cheap generator, bus 3 an expensive one and the only load. The three lines have the same reactance, so
two thirds of a transfer from bus 1 to bus 3 take the direct line 1-3, whose rating is low. Run it with::

    python braess3.py
"""

import numpy as np
import pulp

LINES = [(0, 1), (1, 2), (0, 2)]  # 1-2, 2-3, 1-3 (zero-based buses)
X = np.array([0.1, 0.1, 0.1])  # p.u.
RATING = np.array([200.0, 200.0, 100.0])  # MW
LOAD = np.array([0.0, 0.0, 180.0])  # MW
GENS = [(0, 300.0, 10.0), (2, 300.0, 50.0)]  # (bus, max MW, $/MWh)


# --8<-- [start:opf]
def dc_opf(open_lines: tuple[int, ...] = ()) -> dict:
    """DC-OPF of the 3-bus case with the lines in ``open_lines`` removed; returns cost, dispatch, flows and LMPs."""
    m = pulp.LpProblem('braess', pulp.LpMinimize)
    p = [m.add_variable(f'p_{g}', 0, p_max) for g, (_, p_max, _) in enumerate(GENS)]
    theta = [m.add_variable(f'theta_{n}', None, None) for n in range(3)]
    f = [m.add_variable(f'f_{k}', -r, r) for k, r in enumerate(RATING)]
    m += theta[0] == 0
    for k, (i, j) in enumerate(LINES):
        if k in open_lines:
            m += f[k] == 0
        else:
            m += f[k] == (theta[i] - theta[j]) / X[k] * 100  # 100 MVA base, flows in MW
    balance = []
    for n in range(3):
        c = pulp.lpSum(p[g] for g, (bus, _, _) in enumerate(GENS) if bus == n) - LOAD[n] == pulp.lpSum(
            f[k] for k, (i, _) in enumerate(LINES) if i == n
        ) - pulp.lpSum(f[k] for k, (_, j) in enumerate(LINES) if j == n)
        m += c, f'balance_{n}'
        balance.append(c)
    m += pulp.lpSum(cost * p[g] for g, (_, _, cost) in enumerate(GENS))
    m.solve(pulp.HiGHS(msg=False))
    return {
        'cost': pulp.value(m.objective),
        'p': np.array([v.value() for v in p]),
        'f': np.array([v.value() for v in f]),
        'theta': np.array([v.value() for v in theta]),
        'lmp': np.array([c.pi for c in balance]),
    }


# --8<-- [end:opf]
# --8<-- [start:bigm]
def big_m_cycle(line: int) -> float:
    """Exercise 1: b_l times the largest angle gap the rest of the cycle allows, sum of rating / b over its lines."""
    b = 100 / X  # MW per radian
    others = [k for k in range(len(LINES)) if k != line]
    return float(b[line] * sum(RATING[k] / b[k] for k in others))


def gap_when_open(line: int) -> float:
    """|b_l (theta_i - theta_j)| in MW at the DC-OPF optimum with ``line`` open: must stay below ``big_m_cycle``."""
    r = dc_opf((line,))
    i, j = LINES[line]
    return float(abs(100 / X[line] * (r['theta'][i] - r['theta'][j])))


# --8<-- [end:bigm]
def main() -> None:
    """Both topologies, their LMPs, and the big-M check."""
    for name, open_lines in (('all lines closed', ()), ('line 1-3 open', (2,))):
        r = dc_opf(open_lines)
        print(
            f'{name}: cost {r["cost"]:.0f} $/h, dispatch {np.round(r["p"], 1)} MW, '
            f'flows {np.round(r["f"], 1)} MW, LMP {np.round(r["lmp"], 1)} $/MWh'
        )
    print(
        f'big-M for line 1-3 on the cycle: {big_m_cycle(2):.0f} MW; angle gap with it open: {gap_when_open(2):.0f} MW'
    )


if __name__ == '__main__':
    main()
