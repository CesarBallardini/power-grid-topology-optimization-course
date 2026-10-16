"""Newton-Raphson power flow of a five-bus grid, its failure modes and its initializations (Chapter 15, Lab 1-4).

The grid is a MATPOWER-style case dict (``CASE5``), so the same data feed the Newton-Raphson solver of this module,
``pandapower`` (through ``pandapower.converter.from_ppc``) and, saved as a case file, MATPOWER's ``runpf``.
Column indices follow the MATPOWER manual: bus table (bus number, type, Pd, Qd, Gs, Bs, ..., Vm), branch table
(from, to, r, x, b, ..., ratio, angle, status), generator table (bus, Pg, Qg, Qmax, Qmin, Vg, ...).
"""

import numpy as np
import pandapower as pp
from ac_jacobian import injections, jacobian_blocks

PQ, PV, SLACK = 1, 2, 3
# fmt: off
CASE5 = {
    'version': '2',
    'baseMVA': 100.0,
    # bus_i type Pd   Qd   Gs Bs area Vm    Va baseKV zone Vmax Vmin
    'bus': np.array([
        [1, 3,  0.0,  0.0, 0, 0, 1, 1.06, 0, 230, 1, 1.1, 0.9],
        [2, 2, 20.0, 10.0, 0, 0, 1, 1.045, 0, 230, 1, 1.1, 0.9],
        [3, 1, 45.0, 15.0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
        [4, 1, 40.0,  5.0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
        [5, 1, 60.0, 10.0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
    ], dtype=float),
    # bus Pg   Qg Qmax Qmin  Vg    mBase status Pmax Pmin (+ 11 cost/ramp columns MATPOWER expects)
    'gen': np.array([
        [1,  0.0, 0, 300, -300, 1.06,  100, 1, 300, 0] + [0] * 11,
        [2, 40.0, 0, 300, -300, 1.045, 100, 1, 300, 0] + [0] * 11,
    ], dtype=float),
    # f t  r     x     b     rateA rateB rateC ratio angle status angmin angmax
    'branch': np.array([
        [1, 2, 0.02, 0.06, 0.06, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 3, 0.08, 0.24, 0.05, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 3, 0.06, 0.18, 0.04, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 4, 0.06, 0.18, 0.04, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 5, 0.04, 0.12, 0.03, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 4, 0.01, 0.03, 0.02, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.08, 0.24, 0.05, 250, 250, 250, 0, 0, 1, -360, 360],
    ], dtype=float),
}
# fmt: on


def make_ybus(ppc: dict) -> np.ndarray:
    """Dense bus admittance matrix (pu) of a case dict: series branches, line charging, taps and bus shunts."""
    n = len(ppc['bus'])
    y = np.zeros((n, n), dtype=complex)
    for f, t, r, x, b, *_rest, ratio, angle, status in ppc['branch'][:, :11]:
        if not status:
            continue
        i, k = int(f) - 1, int(t) - 1
        ys = 1 / (r + 1j * x)
        tap = (ratio if ratio else 1.0) * np.exp(1j * np.deg2rad(angle))
        y[i, i] += (ys + 1j * b / 2) / abs(tap) ** 2
        y[k, k] += ys + 1j * b / 2
        y[i, k] -= ys / np.conj(tap)
        y[k, i] -= ys / tap
    y[np.diag_indices(n)] += (ppc['bus'][:, 4] + 1j * ppc['bus'][:, 5]) / ppc['baseMVA']
    return y


def specified(ppc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bus types, specified injections P and Q (pu, generation minus load) and the voltage set points."""
    bus, gen = ppc['bus'], ppc['gen']
    types = bus[:, 1].astype(int)
    p = -bus[:, 2] / ppc['baseMVA']
    q = -bus[:, 3] / ppc['baseMVA']
    vset = np.ones(len(bus))
    for g in gen[gen[:, 7] > 0]:
        i = int(g[0]) - 1
        p[i] += g[1] / ppc['baseMVA']
        q[i] += g[2] / ppc['baseMVA']
        vset[i] = g[5]
    return types, p, q, vset


def newton_pf(
    ybus: np.ndarray,
    types: np.ndarray,
    p_spec: np.ndarray,
    q_spec: np.ndarray,
    theta0: np.ndarray,
    vm0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[float], bool]:
    """Polar Newton-Raphson: angles, magnitudes, the mismatch infinity norm per iteration and a converged flag.

    Unknowns are the angles of the non-slack buses and the magnitudes of the PQ buses; the Jacobian blocks come
    from ``ac_jacobian.jacobian_blocks``. The history has one entry per evaluated mismatch, the first one at the
    starting point. A singular Jacobian (an island without a slack) raises ``numpy.linalg.LinAlgError``.
    """
    theta, vm = theta0.astype(float).copy(), vm0.astype(float).copy()
    non_slack = np.flatnonzero(types != SLACK)
    pq = np.flatnonzero(types == PQ)
    history = []
    for _ in range(max_iter):
        s = injections(theta, vm, ybus)
        mismatch = np.concatenate([(p_spec - s.real)[non_slack], (q_spec - s.imag)[pq]])
        history.append(float(np.abs(mismatch).max()))
        if history[-1] < tol:
            return theta, vm, history, True
        j11, j12, j21, j22 = jacobian_blocks(theta, vm, ybus)
        jac = np.block(
            [
                [j11[np.ix_(non_slack, non_slack)], j12[np.ix_(non_slack, pq)]],
                [j21[np.ix_(pq, non_slack)], j22[np.ix_(pq, pq)]],
            ]
        )
        step = np.linalg.solve(jac, mismatch)
        theta[non_slack] += step[: len(non_slack)]
        vm[pq] += step[len(non_slack) :]
    return theta, vm, history, False


def flat_start(ppc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Zero angles, set-point magnitudes at the slack and PV buses, 1 pu at the PQ buses."""
    types, _, _, vset = specified(ppc)
    vm = np.where(types == PQ, 1.0, vset)
    return np.zeros(len(types)), vm


def dc_start(ppc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Angles of the DC power flow (b = 1/x, slack angle 0) with the magnitudes of the flat start."""
    types, p, _, _ = specified(ppc)
    n = len(types)
    bbus = np.zeros((n, n))
    for f, t, _r, x in ppc['branch'][:, :4]:
        i, k = int(f) - 1, int(t) - 1
        bbus[[i, k], [i, k]] += 1 / x
        bbus[[i, k], [k, i]] -= 1 / x
    keep = types != SLACK
    theta = np.zeros(n)
    theta[keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], p[keep])
    return theta, flat_start(ppc)[1]


def solve(
    ppc: dict, start: tuple[np.ndarray, np.ndarray] | None = None, **kwargs
) -> tuple[np.ndarray, np.ndarray, list[float], bool]:
    """Newton-Raphson on a case dict from ``start`` (default: flat start)."""
    types, p, q, _ = specified(ppc)
    theta0, vm0 = start if start is not None else flat_start(ppc)
    return newton_pf(make_ybus(ppc), types, p, q, theta0, vm0, **kwargs)


def scaled_loads(ppc: dict, factor: float) -> dict:
    """A copy of the case with every load (P and Q) multiplied by ``factor``."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}
    out['bus'][:, 2:4] *= factor
    return out


def without_branches(ppc: dict, rows: list[int]) -> dict:
    """A copy of the case with the branches in ``rows`` (0-based) removed."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}
    out['branch'] = np.delete(out['branch'], rows, axis=0)
    return out


def pv_to_pq(ppc: dict, bus: int) -> dict:
    """A copy of the case where the generator at ``bus`` (1-based) no longer holds its voltage: Qg is fixed at 0."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}
    out['bus'][bus - 1, 1] = PQ
    return out


def pandapower_net(ppc: dict, f_hz: float = 50.0) -> pp.pandapowerNet:
    """The same case as a pandapower network: per-unit branch data become ohms and nF on the bus voltage base."""
    net = pp.create_empty_network(sn_mva=ppc['baseMVA'], f_hz=f_hz)
    for bus_i, _type, pd, qd, *_rest in ppc['bus']:
        b = pp.create_bus(net, vn_kv=ppc['bus'][int(bus_i) - 1, 9], index=int(bus_i) - 1)
        if pd or qd:
            pp.create_load(net, b, p_mw=pd, q_mvar=qd)
    types = ppc['bus'][:, 1].astype(int)
    for g in ppc['gen'][ppc['gen'][:, 7] > 0]:
        bus = int(g[0]) - 1
        if types[bus] == SLACK:
            pp.create_ext_grid(net, bus, vm_pu=g[5])
        elif types[bus] == PV:
            pp.create_gen(net, bus, p_mw=g[1], vm_pu=g[5])
        else:
            pp.create_sgen(net, bus, p_mw=g[1], q_mvar=g[2])
    for f, t, r, x, b, *_rest in ppc['branch']:
        z_base = ppc['bus'][int(f) - 1, 9] ** 2 / ppc['baseMVA']  # ohm
        c_nf = b / z_base / (2 * np.pi * f_hz) * 1e9
        pp.create_line_from_parameters(
            net,
            int(f) - 1,
            int(t) - 1,
            length_km=1.0,
            r_ohm_per_km=r * z_base,
            x_ohm_per_km=x * z_base,
            c_nf_per_km=c_nf,
            max_i_ka=1.0,
        )
    return net


def pandapower_solution(ppc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Voltage angles (rad) and magnitudes (pu) of ``pandapower.runpp`` on the same case."""
    net = pandapower_net(ppc)
    pp.runpp(net, tolerance_mva=1e-10, init='flat')
    return np.deg2rad(net.res_bus.va_degree.to_numpy()), net.res_bus.vm_pu.to_numpy()


def max_loading(ppc: dict, step: float = 0.05, limit: float = 10.0) -> float:
    """The largest load multiple (in steps of ``step``) for which Newton-Raphson from a flat start converges."""
    factor = 1.0
    while factor + step <= limit and solve(scaled_loads(ppc, factor + step))[3]:
        factor += step
    return factor


def main() -> None:
    """Print every number quoted in the Lab solutions of steps 1-4."""
    theta, vm, hist, ok = solve(CASE5)
    print('Step 1: mismatch inf-norm per iteration (pu):', ' '.join(f'{h:.1e}' for h in hist), 'converged', ok)
    print('  angles (deg):', np.round(np.rad2deg(theta), 4), ' magnitudes (pu):', np.round(vm, 5))
    th_pp, vm_pp = pandapower_solution(CASE5)
    dv, dth = np.abs(vm - vm_pp).max(), np.rad2deg(np.abs(theta - th_pp)).max()
    print(f'Step 2: pandapower  max |dV| = {dv:.1e} pu, max |dtheta| = {dth:.1e} deg')
    lam = max_loading(CASE5)
    print(f'Step 3a: loads can be scaled by {lam:.2f}; at {lam + 0.05:.2f} the iteration fails:')
    th_f, vm_f, hist_f, ok_f = solve(scaled_loads(CASE5, lam + 0.05))
    print('  mismatch history:', ' '.join(f'{h:.1e}' for h in hist_f), 'converged', ok_f, ' |V| =', np.round(vm_f, 3))
    _, vm_l, hist_l, _ = solve(scaled_loads(CASE5, lam))
    print(f'  at {lam:.2f}: {len(hist_l) - 1} iterations, lowest |V| = {vm_l.min():.3f} pu')
    no_q = pv_to_pq(CASE5, 2)
    lam_q = max_loading(no_q)
    _, vm_q, hist_q, ok_q = solve(no_q)
    print(f'Step 3b: bus 2 as PQ (no voltage support): nominal load converges={ok_q} in {len(hist_q) - 1} iterations,')
    print(f'  |V2| = {vm_q[1]:.4f};', end='')
    print(f'  the largest load multiple drops from {lam:.2f} to {lam_q:.2f}')
    try:
        solve(without_branches(CASE5, [4, 6]))
        print('Step 3c: islanding bus 5 did not raise')
    except np.linalg.LinAlgError as err:
        print('Step 3c: lines 2-5 and 4-5 removed, bus 5 islanded with its load:', err)
    heavy = scaled_loads(CASE5, lam)
    previous = solve(scaled_loads(CASE5, lam - 0.1))[:2]
    for name, start in (('flat', flat_start(heavy)), ('DC angles', dc_start(heavy)), ('previous solution', previous)):
        th_s, vm_s, hist_s, ok_s = solve(heavy, start)
        print(f'Step 4: {name:18s} {len(hist_s) - 1} iterations, converged={ok_s}, |V| = {np.round(vm_s, 4)}')


if __name__ == '__main__':
    main()
