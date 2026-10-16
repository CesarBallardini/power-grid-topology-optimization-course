"""The 7-bus case of Chapter 19: DC OPF and SCOPF with SciPy, N-1 and bus splits with pandapower.

Run it to print the numbers quoted in the chapter (worked example, Lab Part C steps 1-3, Exercise 10)::

    python case7.py

Everything is in MW, $/MWh and per unit on 100 MVA. Bus 1 is the slack. Needs NumPy, SciPy and pandapower.
"""

import numpy as np
import pandapower as pp
from scipy.optimize import linprog

# --8<-- [start:dc]
# (from bus, to bus, x p.u., limit MW)
LINES = [
    (1, 2, 0.10, 240),
    (1, 3, 0.15, 240),
    (2, 3, 0.10, 200),
    (2, 5, 0.12, 240),
    (3, 4, 0.10, 200),
    (3, 6, 0.14, 200),
    (4, 5, 0.10, 200),
    (4, 6, 0.12, 200),
    (5, 6, 0.10, 200),
    (5, 7, 0.15, 200),
    (6, 7, 0.12, 200),
]
N_BUS, N_LINE = 7, len(LINES)
NAMES = [f'L{i}-{j}' for i, j, *_ in LINES]
LOAD = np.array([0, 0, 80, 100, 90, 70, 0.0])  # MW per bus
GENS = {1: (10, 250), 2: (20, 200), 7: (35, 200)}  # bus: ($/MWh, MW max)
LIMIT = np.array([f for *_, f in LINES], float)
COST = np.array([c for c, _ in GENS.values()], float)
KV = 110.0  # for pandapower


def _ptdf() -> np.ndarray:
    """PTDF of the intact grid with bus 1 as slack (Chapter 17)."""
    a = np.zeros((N_LINE, N_BUS))  # branch-bus incidence
    for k, (i, j, *_) in enumerate(LINES):
        a[k, i - 1], a[k, j - 1] = 1, -1
    b = 1 / np.array([x for _, _, x, _ in LINES])
    bbus = a.T @ np.diag(b) @ a
    ptdf = np.zeros((N_LINE, N_BUS))
    ptdf[:, 1:] = np.diag(b) @ a[:, 1:] @ np.linalg.inv(bbus[1:, 1:])
    return ptdf


PTDF = _ptdf()
GEN_TO_BUS = np.zeros((N_BUS, len(GENS)))
for _g, _bus in enumerate(GENS):
    GEN_TO_BUS[_bus - 1, _g] = 1


def lodf(k: int) -> np.ndarray:
    """Flow change on every line per MW that line ``k`` carried before its outage."""
    h = PTDF[:, LINES[k][0] - 1] - PTDF[:, LINES[k][1] - 1]
    return h / (1 - h[k])


def flows(pg: np.ndarray, out: int | None = None) -> np.ndarray:
    """DC line flows for the dispatch ``pg`` (one value per generator), optionally after the outage of line ``out``."""
    f = PTDF @ (GEN_TO_BUS @ pg - LOAD)
    if out is None:
        return f
    f = f + lodf(out) * f[out]
    f[out] = 0
    return f


def dispatch(contingencies=(), limit: np.ndarray = LIMIT):
    """Minimum-cost dispatch with base-case limits and, for each line in ``contingencies``, its post-outage limits.

    ``dispatch()`` is the OPF, ``dispatch(range(N_LINE))`` the N-1 SCOPF. Returns the ``linprog`` result; ``x`` is
    the dispatch and ``fun`` the cost in $/h.
    """
    fmat, d = PTDF @ GEN_TO_BUS, PTDF @ LOAD  # base flow = fmat @ pg - d
    rows, rhs = [fmat, -fmat], [limit + d, limit - d]
    for k in contingencies:
        keep = np.arange(N_LINE) != k
        fk, dk = fmat + np.outer(lodf(k), fmat[k]), d + lodf(k) * d[k]
        rows += [fk[keep], -fk[keep]]
        rhs += [limit[keep] + dk[keep], limit[keep] - dk[keep]]
    return linprog(
        COST,
        A_ub=np.vstack(rows),
        b_ub=np.concatenate(rhs),
        A_eq=np.ones((1, len(GENS))),
        b_eq=[LOAD.sum()],
        bounds=[(0, p) for _, p in GENS.values()],
        method='highs',
    )


def n1_loading(pg: np.ndarray, limit: np.ndarray = LIMIT) -> np.ndarray:
    """Post-contingency loading in percent, shape (outage, line); the outaged line itself reads 0."""
    return np.array([100 * np.abs(flows(pg, k)) / limit for k in range(N_LINE)])


# --8<-- [end:dc]
def worst(pct: np.ndarray) -> str:
    """Describe the largest entry of an (outage, line) loading table."""
    o, k = np.unravel_index(pct.argmax(), pct.shape)
    return f'{pct.max():.1f} % on {NAMES[k]} after {NAMES[o]} out'


def binding(pg: np.ndarray, limit: np.ndarray = LIMIT, tol: float = 1e-6) -> list[tuple[str, str]]:
    """(outaged line, loaded line) pairs whose post-contingency loading is exactly 100 % (Lab Part C, step 1)."""
    pct = n1_loading(pg, limit)
    return [(NAMES[o], NAMES[k]) for o, k in zip(*np.nonzero(np.abs(pct - 100) < tol), strict=True)]


# --8<-- [start:ac]
def pandapower_net(pg: np.ndarray, limit: np.ndarray = LIMIT) -> pp.pandapowerNet:
    """The same grid for pandapower: 110 kV lines with r = 0.1 x, loads with Q = 0.3 P, bus 1 as ``ext_grid``."""
    net = pp.create_empty_network(sn_mva=100)
    for i in range(N_BUS):
        pp.create_bus(net, vn_kv=KV, name=f'bus{i + 1}')
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    for (bus, _), p in zip(GENS.items(), pg, strict=True):
        if bus != 1:  # bus 1 is the slack and also covers the losses
            pp.create_gen(net, bus - 1, p_mw=p, vm_pu=1.0, name=f'gen{bus}')
    for bus, p in enumerate(LOAD):
        if p:
            pp.create_load(net, bus, p_mw=p, q_mvar=0.3 * p)
    for (i, j, x, _), lim, name in zip(LINES, limit, NAMES, strict=True):
        z = x * KV**2 / 100  # ohm
        pp.create_line_from_parameters(
            net,
            i - 1,
            j - 1,
            length_km=1,
            r_ohm_per_km=0.1 * z,
            x_ohm_per_km=z,
            c_nf_per_km=0,
            max_i_ka=lim / (np.sqrt(3) * KV),
            name=name,
        )
    return net


def split_bus(net: pp.pandapowerNet, bus: int, lines: list[str]) -> pp.pandapowerNet:
    """Move the given lines from ``bus`` (1-based) to a new busbar; the load and any generator stay behind."""
    new = pp.create_bus(net, vn_kv=KV, name=f'bus{bus}b')
    for name in lines:
        k = NAMES.index(name)
        col = 'from_bus' if net.line.at[k, 'from_bus'] == bus - 1 else 'to_bus'
        net.line.at[k, col] = new
    return net


def pandapower_n1(net: pp.pandapowerNet, ac: bool = False) -> np.ndarray:
    """Worst value of every line over all single-line outages: DC flow in MW, or AC loading in percent."""
    result = np.zeros(len(net.line))
    for k in range(len(net.line)):
        net.line.at[k, 'in_service'] = False
        if ac:
            pp.runpp(net)
            value = net.res_line.loading_percent.to_numpy(copy=True)  # writable under pandas 3 copy-on-write too
        else:
            pp.rundcpp(net)
            value = np.abs(net.res_line.p_from_mw.to_numpy())
        value[k] = 0
        result = np.maximum(result, value)
        net.line.at[k, 'in_service'] = True
    return result


# --8<-- [end:ac]
def ac_secure_dispatch(limit: np.ndarray = LIMIT, max_iter: int = 10) -> tuple[list[float], np.ndarray]:
    """Lab Part C, step 2: derate the limits of the AC-overloaded lines until the SCOPF dispatch is AC secure.

    Returns the SCOPF cost after each iteration (the first entry is the DC SCOPF) and the final dispatch.
    """
    lim = limit.copy()
    costs = []
    for _ in range(max_iter):
        r = dispatch(range(N_LINE), lim)
        costs.append(r.fun)
        ac = pandapower_n1(pandapower_net(r.x, limit), ac=True)
        if ac.max() <= 100:
            break
        dc = n1_loading(r.x, limit).max(axis=0) * limit / 100  # worst DC post-contingency flow, MW
        over = ac > 100
        lim[over] = dc[over] * 100 / ac[over]
    return costs, r.x


def main() -> None:
    """Print the worked example, Lab Part C steps 1-3 and the SciPy and pandapower parts of Exercise 10."""
    opf, scopf = dispatch(), dispatch(range(N_LINE))
    for name, r in (('OPF', opf), ('SCOPF', scopf)):
        print(f'{name:5s} cost {r.fun:6.0f} $/h  Pg {np.round(r.x, 1)}  worst N-1 loading {worst(n1_loading(r.x))}')
    for name, r in (('OPF', opf), ('SCOPF', scopf)):
        print(f'{name:5s} AC worst N-1 loading {pandapower_n1(pandapower_net(r.x), ac=True).max():.1f} %')
    print('Part C step 1, binding (outage, line) pairs of the SCOPF:', binding(scopf.x))
    costs, _ = ac_secure_dispatch()
    print('Part C step 2, SCOPF cost per derating iteration:', np.round(costs))
    lim = LIMIT.copy()
    lim[NAMES.index('L1-3')] = 150
    opf3, scopf3 = dispatch((), lim), dispatch(range(N_LINE), lim)
    print(
        f'Part C step 3, L1-3 limited to 150 MW: OPF {opf3.fun:.0f} $/h, SCOPF {scopf3.fun:.0f} $/h, '
        f'SCOPF Pg {np.round(scopf3.x, 1)}, binding {binding(scopf3.x, lim)}, '
        f'OPF base-case flow on L1-3 {flows(opf3.x)[NAMES.index("L1-3")]:.1f} MW'
    )

    print('\nExercise 10: SCOPF dispatch, L3-6 limited to 100 MW')
    lim = LIMIT.copy()
    lim[NAMES.index('L3-6')] = 100
    print('  old SCOPF dispatch now has worst N-1 loading', worst(n1_loading(scopf.x, lim)))
    r = dispatch(range(N_LINE), lim)
    print(f'  new SCOPF: cost {r.fun:.0f} $/h, Pg {np.round(r.x, 1)}, binding {binding(r.x, lim)}')
    for moved in (['L3-6', 'L6-7'], ['L3-6', 'L4-6'], ['L4-6', 'L6-7']):
        net = split_bus(pandapower_net(scopf.x, lim), 6, moved)
        dc = pandapower_n1(net)
        ac = pandapower_n1(net, ac=True)
        print(
            f'  bus 6 split, {moved} on their own busbar: DC worst N-1 flow on L3-6 {dc[NAMES.index("L3-6")]:.1f} MW, '
            f'worst DC loading {(100 * dc / lim).max():.1f} %, '
            f'DC worst N-1 flow on L3-4 {dc[NAMES.index("L3-4")]:.1f} MW, '
            f'worst AC loading {ac.max():.1f} % on {NAMES[ac.argmax()]}'
        )


if __name__ == '__main__':
    main()
