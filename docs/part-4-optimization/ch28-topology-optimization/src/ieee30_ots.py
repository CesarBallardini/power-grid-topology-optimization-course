"""DC-OPF, optimal transmission switching and lazy N-1 constraints for IEEE 30 with PuLP and HiGHS (Chapter 28, Lab A).

The network data (lines, reactances, ratings) come from ``ch27_ieee30.load_case`` in the Chapter 27 folder, which is
put on ``sys.path`` below; generators and loads are read from pandapower's ``case30`` again. MATPOWER's case30 has
quadratic generation costs, so the LP uses linear costs: ten times the linear coefficient ``cp1`` of each generator
(20, 17.5, 10, 32.5, 30 and 30 $/MWh). Run it to print every number of the Lab A solutions (about a minute, the full
sweep of step 2 included)::

    python ieee30_ots.py
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandapower.networks as pn
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ch27-pareto-multi-objective' / 'src'))
import ch27_ieee30  # noqa: E402

TIME_LIMIT = 20.0  # seconds per MILP in the sweep


# --8<-- [start:data]
def data(scale: float = 1.0) -> dict:
    """IEEE 30 for the OPF: the Chapter 27 network dict plus ``load`` (MW per bus) and ``gens`` (bus, MW, $/MWh)."""
    case = ch27_ieee30.load_case(scale)
    net = pn.case30()
    load = np.zeros(case['n_bus'])
    np.add.at(load, net.load.bus.to_numpy(), scale * net.load.p_mw.to_numpy())
    buses = [int(net.ext_grid.bus.iloc[0]), *net.gen.bus.astype(int).tolist()]
    p_max = [float(net.ext_grid.max_p_mw.iloc[0]), *net.gen.max_p_mw.tolist()]
    cost = (10 * net.poly_cost.cp1_eur_per_mw).tolist()  # ext_grid first, then the gens, as in poly_cost
    case.update(load=load, gens=list(zip(buses, p_max, cost, strict=True)), b=1 / case['x'])
    return case


def _base_model(case: dict, name: str) -> tuple[pulp.LpProblem, dict]:
    """Variables and constraints shared by the DC-OPF and the switching MILP; returns the problem and its parts."""
    n_bus, lines = case['n_bus'], case['lines']
    m = pulp.LpProblem(name, pulp.LpMinimize)
    p = [m.add_variable(f'p_{g}', 0, p_max) for g, (_, p_max, _) in enumerate(case['gens'])]
    theta = [m.add_variable(f'theta_{n}', None, None) for n in range(n_bus)]
    f = [m.add_variable(f'f_{k}', -r, r) for k, r in enumerate(case['rating'])]
    m += theta[0] == 0
    balance = []
    for n in range(n_bus):
        generation = pulp.lpSum(p[g] for g, (bus, _, _) in enumerate(case['gens']) if bus == n)
        out_flow = pulp.lpSum(f[k] for k in range(len(lines)) if lines[k][0] == n)
        in_flow = pulp.lpSum(f[k] for k in range(len(lines)) if lines[k][1] == n)
        c = generation - case['load'][n] == out_flow - in_flow
        m += c, f'balance_{n}'
        balance.append(c)
    m += pulp.lpSum(cost * p[g] for g, (_, _, cost) in enumerate(case['gens']))
    return m, {'p': p, 'theta': theta, 'f': f, 'balance': balance}


def dc_opf(case: dict, contingencies: list[int] | None = None) -> dict:
    """DC-OPF as an LP; ``contingencies`` adds the post-outage limits of those lines through the LODF.

    Returns cost ($/h), dispatch ``p`` (MW), flows ``f`` (MW), angles ``theta`` (rad) and the nodal prices ``lmp``
    ($/MWh, the duals of the balance constraints).
    """
    m, v = _base_model(case, 'dc_opf')
    for k, (i, j) in enumerate(case['lines']):
        m += v['f'][k] == case['b'][k] * (v['theta'][i] - v['theta'][j])
    if contingencies:
        lodf = _lodf(case)
        for c in contingencies:
            if np.isnan(lodf[:, c]).any():
                continue  # a bridge: its outage islands the grid and has no LODF, ToOp drops it too
            for k in range(len(case['lines'])):
                if k != c:
                    post = v['f'][k] + lodf[k, c] * v['f'][c]
                    m += post <= case['rating'][k]
                    m += -post <= case['rating'][k]
    m.solve(pulp.HiGHS(msg=False))
    return {
        'status': pulp.LpStatus[m.status],
        'cost': pulp.value(m.objective),
        'p': np.array([x.value() for x in v['p']]),
        'f': np.array([x.value() for x in v['f']]),
        'theta': np.array([x.value() for x in v['theta']]),
        'lmp': np.array([c.pi for c in v['balance']]),
    }


# --8<-- [end:data]
# --8<-- [start:ots]
def big_m(case: dict, mode: str = 'global') -> np.ndarray:
    """A bound on |b_l (theta_i - theta_j) - f_l| when line ``l`` is open: ``b_l`` times a bound on the angle gap.

    ``'global'``: the angle gap is at most the sum over every other line of rating / b (a simple path uses each
    line at most once), valid for every connected topology. ``'cycle'``: the shortest path around the line in the
    intact network with weights rating / b, the bound of Exercise 1, valid while every other line stays closed.
    """
    rating_over_b = case['rating'] / case['b']
    if mode == 'global':
        return case['b'] * (rating_over_b.sum() - rating_over_b)
    import networkx as nx  # noqa: PLC0415

    g = nx.Graph()
    for k, (i, j) in enumerate(case['lines']):
        g.add_edge(int(i), int(j), weight=rating_over_b[k], key=k)
    bound = big_m(case, 'global')  # a bridge has no cycle: its opening islands the grid, keep the global bound
    for k, (i, j) in enumerate(case['lines']):
        h = g.copy()
        h.remove_edge(int(i), int(j))
        if nx.has_path(h, int(i), int(j)):
            bound[k] = case['b'][k] * nx.shortest_path_length(h, int(i), int(j), weight='weight')
    return bound


def ots(case: dict, k_max: int, switchable: list[int], time_limit: float = TIME_LIMIT, mode: str = 'global') -> dict:
    """Optimal transmission switching: at most ``k_max`` of the ``switchable`` lines open, big-M formulation."""
    m, v = _base_model(case, 'ots')
    big = big_m(case, mode)
    z = {k: m.add_variable(f'z_{k}', cat='Binary') for k in switchable}
    for k, (i, j) in enumerate(case['lines']):
        gap = case['b'][k] * (v['theta'][i] - v['theta'][j]) - v['f'][k]
        if k in z:
            m += v['f'][k] <= case['rating'][k] * z[k]
            m += -v['f'][k] <= case['rating'][k] * z[k]
            m += gap <= big[k] * (1 - z[k])
            m += -gap <= big[k] * (1 - z[k])
        else:
            m += gap == 0
    m += pulp.lpSum(1 - z[k] for k in switchable) <= k_max
    t0 = time.perf_counter()
    m.solve(pulp.HiGHS(msg=False, timeLimit=time_limit))
    info = m.solverModel.getInfo()
    return {
        'status': pulp.LpStatus[m.status],
        'cost': pulp.value(m.objective),
        'switched': sorted(k for k in switchable if z[k].value() is not None and z[k].value() < 0.5),
        'time_s': time.perf_counter() - t0,
        'gap': info.mip_gap,
        'p': np.array([x.value() for x in v['p']]),
    }


def sweep(case: dict, ranked: list[int], time_limit: float = TIME_LIMIT) -> pd.DataFrame:
    """Lab A step 2: k = 1, 2, 3 for the 5, 10, 20 and all 41 switchable lines (first entries of ``ranked``)."""
    rows = []
    for n_switchable in (5, 10, 20, 41):
        for k in (1, 2, 3):
            r = ots(case, k, ranked[:n_switchable], time_limit)
            rows.append(
                {'switchable': n_switchable, 'k': k, **{key: r[key] for key in ('cost', 'switched', 'time_s', 'gap')}}
            )
    return pd.DataFrame(rows)


def brute_force(case: dict, k_max: int, switchable: list[int]) -> tuple[float, tuple[int, ...], int, float]:
    """One DC-OPF per set of at most ``k_max`` open lines: best cost, its set, the number of LPs and the time."""
    t0 = time.perf_counter()
    best, arg, count = np.inf, (), 0
    for k in range(k_max + 1):
        for subset in itertools.combinations(switchable, k):
            count += 1
            reduced = _without(case, subset)
            if reduced is None:
                continue
            r = dc_opf(reduced)
            if r['status'] == 'Optimal' and r['cost'] < best - 1e-6:
                best, arg = r['cost'], subset
    return best, arg, count, time.perf_counter() - t0


def _without(case: dict, switched: tuple[int, ...]) -> dict | None:
    keep = np.ones(len(case['lines']), dtype=bool)
    keep[list(switched)] = False
    if ch27_ieee30.ptdf(case, keep) is None:
        return None
    return {
        **case,
        'lines': case['lines'][keep],
        'x': case['x'][keep],
        'b': case['b'][keep],
        'rating': case['rating'][keep],
    }


# --8<-- [end:ots]
# --8<-- [start:lazy]
def _lodf(case: dict) -> np.ndarray:
    """LODF matrix (line, outage) of the intact network; a bridge outage gets a column of NaN (its own entry -1)."""
    h = ch27_ieee30.ptdf(case, np.ones(len(case['lines']), dtype=bool))
    hk = h[:, case['lines'][:, 0]] - h[:, case['lines'][:, 1]]
    denominator = 1 - np.diag(hk)
    with np.errstate(divide='ignore', invalid='ignore'):
        lodf = np.where(np.abs(denominator) > 1e-9, hk / denominator, np.nan)
    np.fill_diagonal(lodf, -1.0)
    return lodf


def post_contingency_loading(case: dict, f: np.ndarray) -> np.ndarray:
    """Loading in percent after every single outage, shape (outage, line); a bridge gives a row of NaN."""
    lodf = _lodf(case)
    post = f[None, :] + lodf.T * f[:, None]  # (outage, line)
    return 100 * np.abs(post) / case['rating']


def lazy_scopf(case: dict, n_initial: int = 5) -> tuple[dict, list[dict]]:
    """Lab A step 3: start with the ``n_initial`` worst contingencies, add violated ones until none is left."""
    base = dc_opf(case)
    loading = post_contingency_loading(case, base['f'])
    worst = np.nan_to_num(loading, nan=0).max(axis=1)
    active = list(np.argsort(-worst)[:n_initial])
    history = []
    while True:
        r = dc_opf(case, active)
        loading = post_contingency_loading(case, r['f'])
        worst = np.nan_to_num(loading, nan=0).max(axis=1)
        violated = [int(c) for c in np.flatnonzero(worst > 100 + 1e-6) if c not in active]
        history.append(
            {'constraints': len(active), 'cost': r['cost'], 'worst_loading': worst.max(), 'violated': violated}
        )
        if not violated:
            return r, history
        active.append(violated[int(np.argmax(worst[violated]))])


# --8<-- [end:lazy]
def main() -> None:
    """Lab A, steps 1 to 4."""
    pd.set_option('display.width', 200)
    case = data()
    opf = dc_opf(case)
    h = ch27_ieee30.ptdf(case, np.ones(41, dtype=bool))
    gen_bus = np.zeros(case['n_bus'])
    np.add.at(gen_bus, [bus for bus, _, _ in case['gens']], opf['p'])
    print(f'step 1: DC-OPF cost {opf["cost"]:.1f} $/h, dispatch {np.round(opf["p"], 1)} MW')
    print(f'  max |LP flow - PTDF flow| = {np.abs(opf["f"] - h @ (gen_bus - case["load"])).max():.2e} MW')
    loading = 100 * np.abs(opf['f']) / case['rating']
    binding = [f'{i + 1}-{j + 1}' for (i, j), load in zip(case['lines'], loading, strict=True) if load > 100 - 1e-6]
    print(f'  lines at their rating: {binding}')
    lmp = opf['lmp']
    print(f'  LMP range {lmp.min():.2f} to {lmp.max():.2f} $/MWh', end='; ')
    print(f'bus 1 {lmp[0]:.2f}, bus 22 {lmp[21]:.2f}, bus 24 {lmp[23]:.2f}')
    print('  LMP per bus:', np.round(lmp, 2))

    ranked = [int(k) for k in np.argsort(-loading)]
    global_m, cycle_m = big_m(case), big_m(case, 'cycle')
    print(f'step 2: big-M global {global_m.min():.0f} to {global_m.max():.0f} MW', end=', ')
    print(f'cycle {cycle_m.min():.0f} to {cycle_m.max():.0f} MW')
    table = sweep(case, ranked)
    table['switched'] = table['switched'].map(
        lambda s: [f'{case["lines"][k][0] + 1}-{case["lines"][k][1] + 1}' for k in s]
    )
    print(table.to_string(index=False, float_format=lambda v: f'{v:.1f}' if v > 1 else f'{v:.2f}'))

    r, history = lazy_scopf(case)
    print(f'step 3: {len(history)} iterations')
    for step in history:
        print(
            f'  {step["constraints"]:2d} contingencies: cost {step["cost"]:.1f} $/h, '
            f'worst loading {step["worst_loading"]:.1f} %, violated {step["violated"]}'
        )
    print(f'  SCOPF dispatch {np.round(r["p"], 1)} MW')

    best, arg, count, dt = brute_force(case, 2, list(range(41)))
    milp = ots(case, 2, list(range(41)))
    names = [f'{i + 1}-{j + 1}' for i, j in case['lines']]
    print(f'step 4: brute force |D| <= 2: {count} LPs in {dt:.1f} s', end=', ')
    print(f'best {best:.1f} $/h with lines {[names[k] for k in arg]}')
    print(
        f'  MILP k = 2, 41 switchable: {milp["cost"]:.1f} $/h in {milp["time_s"]:.1f} s, gap {milp["gap"]:.3g}, '
        f'lines {[names[k] for k in milp["switched"]]}'
    )


if __name__ == '__main__':
    main()
