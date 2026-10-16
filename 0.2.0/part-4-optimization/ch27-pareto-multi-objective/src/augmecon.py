"""AUGMECON2 on a bi-objective facility location problem with PuLP and HiGHS (Chapter 27, Lab 6).

The instance is the one of Chapter 23 (three facilities, four customers). Objective 1 is the opening cost, objective 2
the assignment cost; both are minimized. The exact Pareto set is found by the augmented epsilon-constraint method
with the bypass rule of Mavrotas & Florios 2013 and checked against enumeration of the opening decisions. Run it with::

    python augmecon.py
"""

import itertools

import pulp

FIXED = {'A': 10, 'B': 12, 'C': 8}
ASSIGN = {
    ('A', 1): 2,
    ('A', 2): 4,
    ('A', 3): 5,
    ('A', 4): 7,
    ('B', 1): 6,
    ('B', 2): 3,
    ('B', 3): 2,
    ('B', 4): 4,
    ('C', 1): 7,
    ('C', 2): 6,
    ('C', 3): 5,
    ('C', 4): 2,
}
CUSTOMERS = sorted({j for _, j in ASSIGN})
EPS = 1e-3  # weight of the slack in the augmented objective


# --8<-- [start:augmecon]
def model() -> tuple[pulp.LpProblem, pulp.LpAffineExpression, pulp.LpAffineExpression]:
    """The feasible set of the facility location problem and its two objective expressions."""
    m = pulp.LpProblem('ufl_bi', pulp.LpMinimize)
    y = m.add_variable_dicts('open', list(FIXED), 0, 1, cat='Binary')
    x = m.add_variable_dicts('assign', list(ASSIGN), 0, 1)
    for j in CUSTOMERS:
        m += pulp.lpSum(x[i, j] for i in FIXED) == 1
    for i, j in ASSIGN:
        m += x[i, j] <= y[i]
    return m, pulp.lpSum(FIXED[i] * y[i] for i in FIXED), pulp.lpSum(ASSIGN[k] * x[k] for k in ASSIGN)


def lexmin(first: int, second: int) -> tuple[float, float]:
    """Payoff-table row: minimize objective ``first``, then objective ``second`` with the first held at its optimum."""
    m, f1, f2 = model()
    objectives = [f1, f2]
    m.setObjective(objectives[first])
    m.solve(pulp.HiGHS(msg=False))
    best = pulp.value(objectives[first])
    m += objectives[first] <= best + 1e-9
    m.setObjective(objectives[second])
    m.solve(pulp.HiGHS(msg=False))
    return pulp.value(f1), pulp.value(f2)


def augmecon2(grid_points: int = 20) -> tuple[list[tuple[float, float]], int]:
    """Exact Pareto set (f1, f2) and the number of MILPs solved.

    Objective 2 is constrained, ``f2 + s = e``, on a grid from its best to its worst payoff value; the objective is
    ``f1 - EPS * s / r2`` so that only efficient solutions are returned. When the slack covers ``b`` grid steps the
    next ``b`` grid points would return the same solution and are skipped (the bypass of AUGMECON2).
    """
    payoff = [lexmin(0, 1), lexmin(1, 0)]
    f2_min, f2_max = payoff[1][1], payoff[0][1]
    r2 = f2_max - f2_min
    step = r2 / grid_points
    solutions: list[tuple[float, float]] = []
    k, solved = 0, 0
    while k <= grid_points:
        e = f2_max - k * step
        m, f1, f2 = model()
        s = m.add_variable('slack', 0)
        m += f2 + s == e
        m.setObjective(f1 - EPS * s / r2)
        m.solve(pulp.HiGHS(msg=False))
        solved += 1
        if pulp.LpStatus[m.status] != 'Optimal':
            break
        point = (round(pulp.value(f1), 6), round(pulp.value(f2), 6))
        if point not in solutions:
            solutions.append(point)
        k += 1 + int(s.value() // step) if step > 0 else grid_points + 1
    return sorted(solutions), solved


def enumerate_front() -> list[tuple[float, float]]:
    """Every non-empty set of open facilities with the cheapest assignment, then the non-dominated ones."""
    points = []
    for r in range(1, len(FIXED) + 1):
        for open_set in itertools.combinations(FIXED, r):
            f1 = sum(FIXED[i] for i in open_set)
            f2 = sum(min(ASSIGN[i, j] for i in open_set) for j in CUSTOMERS)
            points.append((float(f1), float(f2)))
    front = [p for p in points if not any((q[0] <= p[0] and q[1] <= p[1] and q != p) for q in points)]
    return sorted(set(front))


# --8<-- [end:augmecon]
def main() -> None:
    """Payoff table, AUGMECON2 result and the enumeration check."""
    print('payoff table (f1 opening, f2 assignment):', lexmin(0, 1), lexmin(1, 0))
    front, solved = augmecon2()
    print(f'AUGMECON2: {len(front)} efficient points {front} with {solved} MILPs (grid of 20 points)')
    print('enumeration of the 7 opening sets:', enumerate_front())


if __name__ == '__main__':
    main()
