"""Either-or constraints with one binary and a big-M, and what a loose M does to the LP relaxation (Chapter 23, Ex. 2).

Either x1 + x2 <= 4 or 3 x1 + x2 <= 6 must hold, with 0 <= x <= 10. With y in {0, 1}:
x1 + x2 <= 4 + M1 y and 3 x1 + x2 <= 6 + M2 (1 - y). A valid M1 must cover every point of the *other* disjunct, so
the smallest one is max {x1 + x2 - 4 : 3 x1 + x2 <= 6, x in the box} = 2, and likewise M2 = max {3 x1 + x2 - 6 :
x1 + x2 <= 4, x in the box} = 6. Bounding the violation over the whole box instead gives 16 and 34.
"""

import numpy as np
import pulp

BOX = 10.0
M_BOX = (2 * BOX - 4.0, 4 * BOX - 6.0)  # (16, 34): the largest violation anywhere in the box


def smallest_valid_m() -> tuple[float, float]:
    """M1 and M2 as LPs: the largest violation of each constraint over the points that satisfy the other one."""
    out = []
    for own, other in (((1.0, 1.0, 4.0), (3.0, 1.0, 6.0)), ((3.0, 1.0, 6.0), (1.0, 1.0, 4.0))):
        m = pulp.LpProblem('violation', pulp.LpMaximize)
        x1, x2 = m.add_variable('x1', 0, BOX), m.add_variable('x2', 0, BOX)
        m += own[0] * x1 + own[1] * x2 - own[2]
        m += other[0] * x1 + other[1] * x2 <= other[2]
        m.solve(pulp.HiGHS(msg=False))
        out.append(float(pulp.value(m.objective)))
    return out[0], out[1]


def solve(
    m1: float, m2: float, relax: bool, objective: tuple[float, float] = (1.0, 1.0)
) -> tuple[float, np.ndarray, float]:
    """Maximize ``objective . x`` over the either-or set (``relax`` lets y take any value in [0, 1])."""
    m = pulp.LpProblem('either_or', pulp.LpMaximize)
    x1, x2 = m.add_variable('x1', 0, BOX), m.add_variable('x2', 0, BOX)
    y = m.add_variable('y', 0, 1, cat='Continuous' if relax else 'Binary')
    m += objective[0] * x1 + objective[1] * x2
    m += x1 + x2 <= 4 + m1 * y
    m += 3 * x1 + x2 <= 6 + m2 * (1 - y)
    m.solve(pulp.HiGHS(msg=False))
    return pulp.value(m.objective), np.array([x1.value(), x2.value()]), y.value()


def main() -> None:
    """The smallest valid big-Ms, the integer optimum and the relaxation bound as M grows."""
    m_tight = smallest_valid_m()
    print(f'smallest valid (M1, M2) = {m_tight}; the box bound would give {M_BOX}')
    for m1, m2 in (m_tight, M_BOX):
        z, x, y = solve(m1, m2, relax=False)
        print(f'(M1, M2) = ({m1:.0f}, {m2:.0f}): integer optimum of max x1 + x2 is {z:.1f} at x = {x}, y = {y:.0f}')
    for m1, m2 in (m_tight, M_BOX, (10 * M_BOX[0], 10 * M_BOX[1])):
        z, x, y = solve(m1, m2, relax=True)
        print(f'(M1, M2) = ({m1:.0f}, {m2:.0f}): LP relaxation bound {z:6.2f} at x = {np.round(x, 2)}, y = {y:.3f}')


if __name__ == '__main__':
    main()
