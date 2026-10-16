"""DC network data and sensitivity matrices shared by the Chapter 33 programs: PTDF, LODF, pandapower extraction."""

from dataclasses import dataclass

import numpy as np


def ptdf_matrix(fr, to, b, n_bus, slack=0):
    """PTDF (branch x bus) of a DC network with branch susceptances ``b``; the slack column is zero."""
    a = np.zeros((len(fr), n_bus))
    a[np.arange(len(fr)), fr] = 1.0
    a[np.arange(len(fr)), to] = -1.0
    bf = a * np.asarray(b)[:, None]
    bbus = a.T @ bf
    keep = np.arange(n_bus) != slack
    ptdf = np.zeros_like(a)
    ptdf[:, keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], bf[:, keep].T).T
    return ptdf


def lodf_matrix(ptdf, fr, to, fix_diagonal=True):
    """LODF (monitored branch x outaged branch) with -1 on the diagonal, and the denominators 1 - c_b per outage.

    A bridge has denominator 0: its column is then inf or nan, and the caller decides what to do with it, the way
    ToOp's kernel returns a success flag instead of raising. With ``fix_diagonal=False`` the diagonal keeps the
    value of the bare formula, c_b / (1 - c_b), the mistake the worked example of the chapter plants.
    """
    col = ptdf[:, fr] - ptdf[:, to]
    den = 1.0 - np.diag(col)
    with np.errstate(divide='ignore', invalid='ignore'):
        lodf = col / den
    if fix_diagonal:
        np.fill_diagonal(lodf, -1.0)
    return lodf, den


def n1_mismatch(fr, to, b, p, fix_diagonal=True):
    """Largest difference between LODF-based N-1 flows and a re-solve without the branch, over non-bridge outages."""
    n_bus = len(p)
    ptdf = ptdf_matrix(fr, to, b, n_bus)
    f0 = ptdf @ p
    lodf, den = lodf_matrix(ptdf, fr, to, fix_diagonal)
    worst = 0.0
    for o in np.flatnonzero(np.abs(den) > 1e-11):
        keep = np.arange(len(fr)) != o
        f_ref = np.zeros(len(fr))
        f_ref[keep] = ptdf_matrix(fr[keep], to[keep], b[keep], n_bus) @ p
        worst = max(worst, float(np.abs(f0 + lodf[:, o] * f0[o] - f_ref).max()))
    return worst


@dataclass
class DCData:
    """The DC model pandapower built internally, plus its own DC power flow result as an oracle."""

    fr: np.ndarray
    to: np.ndarray
    b: np.ndarray
    shift: np.ndarray
    p: np.ndarray
    slack: int
    n_bus: int
    base_mva: float
    flows: np.ndarray

    def without(self, branch):
        """The same network with one branch removed (the oracle flows are not updated)."""
        keep = np.arange(len(self.fr)) != branch
        return DCData(
            self.fr[keep],
            self.to[keep],
            self.b[keep],
            self.shift[keep],
            self.p,
            self.slack,
            self.n_bus,
            self.base_mva,
            self.flows[keep],
        )


def pandapower_dc(net):
    """Run pandapower's DC power flow and read its internal (MATPOWER-style) model and flows.

    Susceptances are 1/(x * tap) in per unit, shifts are in radians, injections ``p`` are in MW and leave out the
    slack generator, whose output is whatever balances the grid, exactly as the DC power flow treats it.
    """
    import pandapower as pp

    pp.rundcpp(net)
    br, bus, gen = (np.real(net._ppc[k]) for k in ('branch', 'bus', 'gen'))
    fr, to = br[:, 0].astype(int), br[:, 1].astype(int)
    tap = np.where(br[:, 8] == 0.0, 1.0, br[:, 8])
    slack = int(np.flatnonzero(bus[:, 1] == 3)[0])
    p = -bus[:, 2].copy()
    for g_bus, pg in zip(gen[:, 0].astype(int), gen[:, 1], strict=True):
        if g_bus != slack:
            p[g_bus] += pg
    return DCData(
        fr, to, 1.0 / (br[:, 3] * tap), np.deg2rad(br[:, 9]), p, slack, len(bus), net._ppc['baseMVA'], br[:, 13].copy()
    )
