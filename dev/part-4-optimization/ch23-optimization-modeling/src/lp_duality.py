"""The two-variable LP of the Lab: vertices, dual values, ranging (Chapter 23, Lab steps 1 and 2) and Exercise 1.

Lab LP: max 3 x1 + 5 x2 subject to x1 <= 4, 2 x2 <= 12, 3 x1 + 2 x2 <= 18, x >= 0. Exercise 1: max 2 x1 - x2 subject to
x1 + x2 <= 4, x1 - x2 <= 1, x1 >= 0 and x2 free, with its dual min 4 y1 + y2 subject to y1 + y2 >= 2, y1 - y2 = -1,
y >= 0.
"""

import itertools

import numpy as np
import pulp

A = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])  # constraint rows of the Lab LP
B = np.array([4.0, 12.0, 18.0])
C = np.array([3.0, 5.0])


def vertices(a: np.ndarray = A, b: np.ndarray = B) -> list[tuple[np.ndarray, float]]:
    """Feasible vertices of {a x <= b, x >= 0} in the plane with their objective values, by pairwise intersection."""
    rows = np.vstack([a, -np.eye(2)])
    rhs = np.concatenate([b, np.zeros(2)])
    out = []
    for i, j in itertools.combinations(range(len(rhs)), 2):
        pair = rows[[i, j]]
        if abs(np.linalg.det(pair)) < 1e-12:
            continue
        x = np.linalg.solve(pair, rhs[[i, j]])
        if np.all(rows @ x <= rhs + 1e-9) and not any(np.allclose(x, v) for v, _ in out):
            out.append((x, float(C @ x)))
    return out


def solve_lab_lp(b: np.ndarray = B, c: np.ndarray = C) -> tuple[float, np.ndarray, np.ndarray]:
    """Optimum, solution and shadow prices of the Lab LP with HiGHS through PuLP.

    PuLP hands HiGHS a minimization of the negated objective, so for a maximization ``constraint.pi`` comes back
    with the opposite sign of the textbook shadow price; the sign is undone here.
    """
    m = pulp.LpProblem('lab', pulp.LpMaximize)
    x = [m.add_variable(f'x{i + 1}', lowBound=0) for i in range(2)]
    m += pulp.lpSum(float(c[i]) * x[i] for i in range(2))
    for k in range(len(b)):
        m += pulp.lpSum(float(A[k, i]) * x[i] for i in range(2)) <= float(b[k]), f'c{k + 1}'
    m.solve(pulp.HiGHS(msg=False))
    duals = -np.array([m.get_constraint_by_name(f'c{k + 1}').pi for k in range(len(b))])
    return pulp.value(m.objective), np.array([v.value() for v in x]), duals


def rhs_ranging(k: int = 2, values: tuple[float, ...] = (10.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0)) -> list[tuple]:
    """Optimum and shadow price of constraint ``k`` as its right-hand side moves."""
    out = []
    for value in values:
        b = B.copy()
        b[k] = value
        z, x, duals = solve_lab_lp(b)
        out.append((value, round(z, 3), np.round(x, 3), round(duals[k], 3)))
    return out


def objective_ranging(c1_values: tuple[float, ...] = (0.0, 2.0, 3.0, 7.0, 7.5, 8.0, 12.0)) -> list[tuple]:
    """Where the optimum sits as the first objective coefficient rotates the objective (c2 stays 5)."""
    return [(c1, np.round(solve_lab_lp(c=np.array([c1, 5.0]))[1], 3)) for c1 in c1_values]


def exercise_1() -> tuple[float, np.ndarray, float, np.ndarray]:
    """Primal and dual of Exercise 1 solved with HiGHS: optimum and solution of each."""
    p = pulp.LpProblem('primal', pulp.LpMaximize)
    x1, x2 = p.add_variable('x1', lowBound=0), p.add_variable('x2')  # x2 is free
    p += 2 * x1 - x2
    p += x1 + x2 <= 4
    p += x1 - x2 <= 1
    p.solve(pulp.HiGHS(msg=False))
    d = pulp.LpProblem('dual', pulp.LpMinimize)
    y1, y2 = d.add_variable('y1', lowBound=0), d.add_variable('y2', lowBound=0)
    d += 4 * y1 + y2
    d += y1 + y2 >= 2
    d += y1 - y2 == -1
    d.solve(pulp.HiGHS(msg=False))
    return (
        pulp.value(p.objective),
        np.array([x1.value(), x2.value()]),
        pulp.value(d.objective),
        np.array([y1.value(), y2.value()]),
    )


def main() -> None:
    """Lab steps 1 and 2 in numbers, then Exercise 1."""
    print('vertices (x1, x2) -> 3 x1 + 5 x2:')
    for x, z in sorted(vertices(), key=lambda t: t[1]):
        print(f'  {np.round(x, 3)} -> {z:.1f}')
    z, x, duals = solve_lab_lp()
    print(f'optimum {z:.1f} at {x}, shadow prices {duals}')
    slack = B - A @ x
    print(f'slacks {slack}; complementary slackness y_k * slack_k = {duals * slack}; b . y = {B @ duals:.1f}')
    print('right-hand side of constraint 3 ranging (b3, z, x, shadow price):')
    for row in rhs_ranging():
        print('  ', row)
    print('objective ranging (c1 with c2 = 5 -> optimal vertex):', objective_ranging())
    zp, xp, zd, yd = exercise_1()
    print(f'exercise 1: primal {zp:.2f} at x = {xp}, dual {zd:.2f} at y = {yd}')


if __name__ == '__main__':
    main()
