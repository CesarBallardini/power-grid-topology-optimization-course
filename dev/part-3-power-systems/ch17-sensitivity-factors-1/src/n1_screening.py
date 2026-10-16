"""PTDF, PSDF and LODF of IEEE 14 and 118 against brute force, bridges, slack choice and the PTDF update (Chapter 17).

Branch data come from the MATPOWER-style table pandapower builds for its own solvers (``net._ppc['branch']``), so the
reactances are in pu on ``sn_mva`` and include the tap ratio (b = 1/(x tau)); the orientation is pandapower's.
"""

import timeit

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import scipy.linalg as sla
from lodf import lodf_column, ptdf_matrix
from pandapower.pypower.makeLODF import makeLODF
from pandapower.pypower.makePTDF import makePTDF

F_BUS, T_BUS, BR_X, TAP = 0, 1, 3, 8


def case_data(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """from, to, effective reactance x tau (pu), balanced injections (MW) and slack bus of a pandapower case."""
    net = getattr(pn, name)()
    pp.rundcpp(net)
    br = net._ppc['branch'].real
    tap = np.where(br[:, TAP] == 0, 1.0, br[:, TAP])
    p = -net.res_bus.p_mw.to_numpy()
    return br[:, F_BUS].astype(int), br[:, T_BUS].astype(int), br[:, BR_X] * tap, p, int(net.ext_grid.bus.iloc[0])


def pandapower_lodf(name: str) -> np.ndarray:
    """LODF matrix of a case from pandapower's PYPOWER port (``makePTDF`` then ``makeLODF``), same orientation.

    pandapower divides by zero on a bridge column and then subtracts infinities on its diagonal; ``errstate``
    silences those two floating-point warnings, and the column comes back as NaN or infinite.
    """
    net = getattr(pn, name)()
    pp.rundcpp(net)
    ppc = net._ppc
    h = makePTDF(ppc['baseMVA'], ppc['bus'], ppc['branch'], slack=int(net.ext_grid.bus.iloc[0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.asarray(makeLODF(ppc['branch'], h))


def incidence(frm: np.ndarray, to: np.ndarray, n_bus: int) -> np.ndarray:
    """Branch-bus incidence matrix A = Cf - Ct."""
    a = np.zeros((len(frm), n_bus))
    a[np.arange(len(frm)), frm], a[np.arange(len(frm)), to] = 1.0, -1.0
    return a


def psdf_matrix(h: np.ndarray, frm: np.ndarray, to: np.ndarray, x: np.ndarray) -> np.ndarray:
    """PSDF[l, k] = b_k (delta_lk - H[l, f_k] + H[l, t_k]) in flow units per radian (ToOp's sign convention)."""
    return (np.eye(len(frm)) - h[:, frm] + h[:, to]) / x


def transfer_factors(h: np.ndarray, frm: np.ndarray, to: np.ndarray) -> np.ndarray:
    """The matrix of all h_lk = H[l, f_k] - H[l, t_k]: flow on l for one MW sent from f_k to t_k."""
    return h[:, frm] - h[:, to]


def lodf_matrix(h: np.ndarray, frm: np.ndarray, to: np.ndarray, tol: float = 1e-11) -> np.ndarray:
    """Full LODF matrix (branch, outage) with -1 on the diagonal; the columns of bridges are NaN."""
    hh = transfer_factors(h, frm, to)
    den = 1.0 - np.diag(hh)
    bridge = np.abs(den) < tol
    with np.errstate(divide='ignore', invalid='ignore'):
        lodf = hh / np.where(bridge, np.nan, den)
    lodf[np.diag_indices(len(frm))] = -1.0
    lodf[:, bridge] = np.nan
    return lodf


def n1_matrix(lodf: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """N-1 flows (outage, branch) in one broadcasted expression: F0 + LODF[:, k] F0[k] for every k."""
    return f0[None, :] + lodf.T * f0[:, None]


def dc_flows(frm: np.ndarray, to: np.ndarray, x: np.ndarray, p: np.ndarray, slack: int) -> np.ndarray:
    """One DC power flow (MW): a single solve of the reduced B theta = P, then F = Bf theta.

    ``scipy.linalg.solve`` with ``assume_a='pos'`` (the reduced Laplacian is positive definite) is used because
    ``numpy.linalg.solve`` with numpy's multithreaded OpenBLAS took 100 times longer on a 117 x 117 system here.
    """
    a = incidence(frm, to, len(p))
    bf = a / x[:, None]
    keep = np.arange(len(p)) != slack
    theta = np.zeros(len(p))
    theta[keep] = sla.solve((a.T @ bf)[np.ix_(keep, keep)], p[keep], assume_a='pos')
    return bf @ theta


def brute_force_n1(frm: np.ndarray, to: np.ndarray, x: np.ndarray, p: np.ndarray, slack: int) -> np.ndarray:
    """N-1 flows by re-solving one DC power flow without each branch; rows of bridges are NaN."""
    n_bus, m = len(p), len(frm)
    out = np.full((m, m), np.nan)
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n_bus))
    graph.add_edges_from(zip(frm, to, strict=True))
    for k in range(m):
        graph.remove_edge(frm[k], to[k])
        connected = nx.is_connected(graph)
        graph.add_edge(frm[k], to[k])
        if not connected:
            continue
        keep = np.arange(m) != k
        out[k, keep] = dc_flows(frm[keep], to[keep], x[keep], p, slack)
        out[k, k] = 0.0
    return out


def bridges(frm: np.ndarray, to: np.ndarray, n_bus: int) -> list[int]:
    """Indices of the branches whose removal disconnects the grid (parallel branches are never bridges)."""
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n_bus))
    graph.add_edges_from(zip(frm, to, strict=True))
    pairs = {frozenset(e) for e in nx.bridges(nx.Graph(graph)) if graph.number_of_edges(*e) == 1}
    return [k for k in range(len(frm)) if frozenset((frm[k], to[k])) in pairs]


def distributed_slack_ptdf(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    """H_w = H (I - w 1ᵀ): the PTDF for a slack distributed with weights ``w`` (MATPOWER manual eq. 4.37)."""
    return h - np.outer(h @ w, np.ones(h.shape[1]))


def generator_outage(
    h: np.ndarray, p: np.ndarray, bus: int, lost_mw: float, w: np.ndarray | None = None
) -> np.ndarray:
    """Flows after losing ``lost_mw`` at ``bus``; the loss is picked up at the slack, or with weights ``w``."""
    hw = h if w is None else distributed_slack_ptdf(h, w)
    return hw @ p - lost_mw * hw[:, bus]


def ptdf_after_outage(h: np.ndarray, frm: np.ndarray, to: np.ndarray, k: int) -> np.ndarray:
    """H' = H + LODF[:, k] H[k, :], with row k set to zero: the PTDF of the grid without branch k."""
    out = h + np.outer(lodf_column(h, frm, to, k), h[k])
    out[k] = 0.0
    return out


def double_outage(h: np.ndarray, frm: np.ndarray, to: np.ndarray, f0: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """Flows after the outage of k1 and then k2, by two LODF steps (the second from the PTDF without k1)."""
    f1 = f0 + lodf_column(h, frm, to, k1) * f0[k1]
    h1 = ptdf_after_outage(h, frm, to, k1)
    return f1 + lodf_column(h1, frm, to, k2) * f1[k2]


def report_case(name: str) -> None:  # noqa: C901
    """Print the Lab numbers (and Exercise 3 or 6) for one pandapower case."""
    frm, to, x, p, slack = case_data(name)
    n_bus, m = len(p), len(frm)
    h = ptdf_matrix(frm, to, x, n_bus, slack)
    f0 = h @ p
    lodf = lodf_matrix(h, frm, to)
    psdf = psdf_matrix(h, frm, to, x)
    n1 = n1_matrix(lodf, f0)
    brute = brute_force_n1(frm, to, x, p, slack)
    ok = ~np.isnan(brute).any(axis=1)
    br = bridges(frm, to, n_bus)
    print(
        f'== {name}: {n_bus} buses, {m} branches, slack {slack}, PTDF {h.shape}, PSDF {psdf.shape}, LODF {lodf.shape}'
    )
    print(f'  largest base flow {np.abs(f0).max():.1f} MW; PTDF slack column all zero: {np.allclose(h[:, slack], 0)}')
    d_brute = np.abs(n1[ok] - brute[ok]).max()
    print(f'  N-1 vs brute force on the {ok.sum()} non-islanding outages: max |diff| {d_brute:.1e} MW')
    hkk = np.diag(transfer_factors(h, frm, to))
    not_bridge = ~np.isin(np.arange(m), br)
    print(
        f'  bridges (networkx): {[f"{k}: {frm[k]}-{to[k]}" for k in br]}; their 1 - h_kk: {np.round(1 - hkk[br], 15)}'
    )
    nan_cols = [int(k) for k in np.flatnonzero(np.isnan(lodf).all(axis=0))]
    print(f'  LODF columns that are NaN: {nan_cols}; smallest 1 - h_kk elsewhere: {(1 - hkk)[not_bridge].min():.4f}')
    pp_lodf = pandapower_lodf(name)
    finite = np.isfinite(pp_lodf).all(axis=0)
    flagged = [int(k) for k in np.flatnonzero(~finite)]
    missed = [k for k in br if finite[k]]
    d_pp = np.abs(pp_lodf[:, not_bridge] - lodf[:, not_bridge]).max()
    print(f'  pandapower makeLODF: non-finite columns {flagged}; max |diff| on the non-bridge columns {d_pp:.1e}')
    if missed:
        huge = np.abs(pp_lodf[:, missed]).max()
        print(
            f'    bridges it returns as finite numbers: {missed}, |LODF| up to {huge:.1e} (round-off over round-off)'
        )
    w = np.full(n_bus, 1 / n_bus)
    hw = distributed_slack_ptdf(h, w)
    d_lodf = np.nanmax(np.abs(lodf_matrix(hw, frm, to) - lodf))
    print(f'  distributed slack: max |F0 change| {np.abs(hw @ p - f0).max():.1e} MW, ', end='')
    print(f'max |LODF change| {d_lodf:.1e}, max |PTDF change| {np.abs(hw - h).max():.3f}')
    gen_bus = int(np.argmax(np.where(np.arange(n_bus) != slack, p, -np.inf)))  # largest non-slack injection
    single, shared = generator_outage(h, p, gen_bus, 40.0), generator_outage(h, p, gen_bus, 40.0, w)
    d_gen = np.abs(single - shared).max()
    print(f'  losing 40 MW at bus {gen_bus}: post-outage flows differ by up to {d_gen:.1f} MW between the two slacks')
    k = next(i for i in range(m) if i not in br)
    keep = np.arange(m) != k
    h_ref = np.zeros_like(h)
    h_ref[keep] = ptdf_matrix(frm[keep], to[keep], x[keep], n_bus, slack)
    d_update = np.abs(ptdf_after_outage(h, frm, to, k) - h_ref).max()
    print(f'  PTDF update after the outage of branch {k}: max |updated - recomputed| {d_update:.1e}')
    if name == 'case118':
        t_bc = min(timeit.repeat(lambda: n1_matrix(lodf_matrix(h, frm, to), f0), number=1, repeat=5))
        t_bf = min(timeit.repeat(lambda: brute_force_n1(frm, to, x, p, slack), number=1, repeat=2))
        print(f'  Exercise 3: broadcast N-1 of {m} outages {1e3 * t_bc:.2f} ms ', end='')
        print(f'vs {m} DC power flows {1e3 * t_bf:.0f} ms; ratio {t_bf / t_bc:.0f}x')
    if name == 'case14':
        k1, k2 = 2, 9
        both = np.flatnonzero((np.arange(m) != k1) & (np.arange(m) != k2))
        f_ref = np.zeros(m)
        f_ref[both] = ptdf_matrix(frm[both], to[both], x[both], n_bus, slack) @ p
        d_double = np.abs(double_outage(h, frm, to, f0, k1, k2) - f_ref).max()
        print(f'  Exercise 6: double outage {k1}, {k2} by two LODF steps vs brute force: max |diff| {d_double:.1e} MW')


def main() -> None:
    """IEEE 14, then IEEE 118."""
    for name in ('case14', 'case118'):
        report_case(name)


if __name__ == '__main__':
    main()
