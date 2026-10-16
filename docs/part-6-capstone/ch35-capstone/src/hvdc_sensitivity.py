"""The DC sensitivity of every branch flow to an HVDC setpoint change, checked against a re-solve (Chapter 35, Ex. 5).

An HVDC link between buses i and j carrying P from i to j is, for the DC power flow of the AC grid, a withdrawal
of P at i and an injection of P at j. Raising the setpoint by dP therefore changes the nodal injections by
dp = -dP e_i + dP e_j and every branch flow by PTDF dp = dP (PTDF[:, j] - PTDF[:, i]); with the opposite
convention for the link's direction the sign flips. ``python hvdc_sensitivity.py`` prints the check.
"""

import numpy as np


def ptdf_matrix(fr, to, b, n_bus, slack=0):
    """PTDF (branch x bus) with a zero slack column."""
    a = np.zeros((len(fr), n_bus))
    a[np.arange(len(fr)), fr] = 1.0
    a[np.arange(len(fr)), to] = -1.0
    bf = a * np.asarray(b)[:, None]
    keep = np.arange(n_bus) != slack
    ptdf = np.zeros_like(a)
    ptdf[:, keep] = np.linalg.solve((a.T @ bf)[np.ix_(keep, keep)], bf[:, keep].T).T
    return ptdf


def six_bus():
    """A meshed six-bus grid: a ring with two chords, unit-ish susceptances, and a load pattern."""
    fr = np.array([0, 1, 2, 3, 4, 5, 0, 1])
    to = np.array([1, 2, 3, 4, 5, 0, 3, 4])
    b = np.array([2.0, 1.5, 1.0, 2.5, 1.0, 1.2, 0.8, 1.7])
    p = np.array([1.5, -0.4, -0.6, 0.0, -0.3, -0.2])
    return fr, to, b, p


def hvdc_column(ptdf, i, j):
    """Sensitivity of every branch flow to +1 MW of setpoint on a link that takes power out at i and in at j."""
    return ptdf[:, j] - ptdf[:, i]


def flows_with_setpoint(ptdf, p, i, j, setpoint):
    """Branch flows when the link moves ``setpoint`` MW from bus i to bus j (the AC grid sees two injections)."""
    dp = np.zeros_like(p)
    dp[i] -= setpoint
    dp[j] += setpoint
    return ptdf @ (p + dp)


def check(fr, to, b, p, i, j, delta=0.37):
    """Largest difference between the re-solved flow change and delta times the HVDC column."""
    ptdf = ptdf_matrix(fr, to, b, len(p))
    base = ptdf @ p
    direct = flows_with_setpoint(ptdf, p, i, j, delta) - base
    predicted = delta * hvdc_column(ptdf, i, j)
    return float(np.abs(direct - predicted).max())


def main():
    """Check the identity for every pair of buses, including pairs that contain the slack."""
    fr, to, b, p = six_bus()
    n = len(p)
    worst = max(check(fr, to, b, p, i, j) for i in range(n) for j in range(n) if i != j)
    print(f'six-bus grid, every ordered pair (i, j): largest |re-solve - dP (PTDF_j - PTDF_i)| = {worst:.2e}')
    ptdf = ptdf_matrix(fr, to, b, n)
    col = hvdc_column(ptdf, 2, 5)
    print('link 2 -> 5, flow change per MW of setpoint on each branch:', np.round(col, 4))
    print('sum of |column| (MW of branch flow moved per MW of setpoint):', round(float(np.abs(col).sum()), 4))
    print(
        'link 0 -> 3 with bus 0 the slack: column equals PTDF[:, 3]:',
        bool(np.allclose(hvdc_column(ptdf, 0, 3), ptdf[:, 3])),
    )


if __name__ == '__main__':
    main()
