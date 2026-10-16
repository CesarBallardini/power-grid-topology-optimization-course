"""Uncapacitated facility location: strong versus big-M formulation and their LP relaxations (Chapter 23)."""

import pulp

FIXED = {'A': 10, 'B': 12, 'C': 8}  # opening cost per facility
ASSIGN = {  # (facility, customer): assignment cost
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


def build(strong: bool, relax: bool, f: dict = FIXED, c: dict = ASSIGN) -> tuple[pulp.LpProblem, dict]:
    """The PuLP model and its opening variables; ``strong`` picks x_ij <= y_i over the big-M sum constraint."""
    facilities, customers = list(f), sorted({j for _, j in c})
    m = pulp.LpProblem('ufl', pulp.LpMinimize)
    y = m.add_variable_dicts('open', facilities, 0, 1, cat='Continuous' if relax else 'Binary')
    x = m.add_variable_dicts('assign', list(c), lowBound=0)
    m += pulp.lpSum(f[i] * y[i] for i in facilities) + pulp.lpSum(c[k] * x[k] for k in c)
    for j in customers:
        m += pulp.lpSum(x[i, j] for i in facilities) == 1
    if strong:
        for i, j in c:
            m += x[i, j] <= y[i]
    else:  # big-M with M = |J|
        for i in facilities:
            m += pulp.lpSum(x[i, j] for j in customers) <= len(customers) * y[i]
    return m, y


def solve(strong: bool, relax: bool, f: dict = FIXED, c: dict = ASSIGN) -> tuple[float, dict]:
    """Minimum cost and the opening decisions of the chosen formulation, solved with HiGHS."""
    m, y = build(strong, relax, f, c)
    m.solve(pulp.HiGHS(msg=False))
    return pulp.value(m.objective), {i: y[i].value() for i in f}


def main() -> None:
    """Integer optimum and LP relaxation of both formulations."""
    for strong in (True, False):
        print('strong' if strong else 'big-M', solve(strong, relax=False), solve(strong, relax=True))


if __name__ == '__main__':
    main()
