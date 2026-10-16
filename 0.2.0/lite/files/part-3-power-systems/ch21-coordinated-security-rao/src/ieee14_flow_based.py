"""Flow-based domain of the IEEE 14-bus grid: nodal and zonal PTDF, CNEC margins, one topology action (Chapter 21 Lab).

Run ``python ieee14_flow_based.py`` to print every number quoted in the Lab solutions. The grid comes from
``pypowsybl.network.create_ieee14()``; the DC model uses nominal transformer ratios
(``dc_use_transformer_ratio=False``) so that the hand-built PTDF and pypowsybl's sensitivities agree to machine
precision.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pypowsybl as pp

S_BASE = 100.0  # MVA
ZONES = {'A': [1, 2, 3, 4, 5], 'B': [6, 11, 12, 13], 'C': [7, 8, 9, 10, 14]}  # bus numbers of the three zones
PARAMETERS = pp.loadflow.Parameters(distributed_slack=False, dc_use_transformer_ratio=False)
F_MAX_FACTOR, F_MAX_FLOOR, FRM_SHARE = 1.3, 20.0, 0.05  # Lab step 4 assumptions
DIRECTIONS = ['A>B', 'B>A', 'A>C', 'C>A', 'B>C', 'C>B']


@dataclass
class Grid:
    """A DC model of the grid: branch ends, reactances in pu, nodal injections in MW and the pypowsybl network."""

    names: list[str]
    frm: np.ndarray
    to: np.ndarray
    x: np.ndarray
    p: np.ndarray
    net: pp.network.Network

    @property
    def n_bus(self) -> int:
        """Number of buses."""
        return len(self.p)


def load_ieee14(open_line: str | None = None) -> Grid:
    """IEEE 14 from pypowsybl as a branch list; ``open_line`` removes one line before building it."""
    net = pp.network.create_ieee14()
    if open_line is not None:
        net.remove_elements(open_line)
    buses = net.get_buses()
    index = {bus: i for i, bus in enumerate(buses.index)}  # VL1_0 -> 0, ..., VL14_0 -> 13
    nominal = buses.voltage_level_id.map(net.get_voltage_levels().nominal_v)
    lines, trafos = net.get_lines(), net.get_2_windings_transformers()
    v1, v2 = nominal[lines.bus1_id].to_numpy(), nominal[lines.bus2_id].to_numpy()
    x_lines = lines.x.to_numpy() * S_BASE / (v1 * v2)  # pypowsybl's per-unit base for a line is V1 * V2
    x_trafos = trafos.x.to_numpy() * S_BASE / nominal[trafos.bus2_id].to_numpy() ** 2  # ohms are on side 2
    branches = pd.concat([lines[['bus1_id', 'bus2_id']], trafos[['bus1_id', 'bus2_id']]])
    p = np.zeros(len(buses))
    for gen in net.get_generators().itertuples():
        p[index[gen.bus_id]] += gen.target_p
    for load in net.get_loads().itertuples():
        p[index[load.bus_id]] -= load.p0
    p[0] -= p.sum()  # the slack generator at bus 1 covers the 13.4 MW imbalance of the data (lossless DC model)
    return Grid(
        names=list(branches.index),
        frm=branches.bus1_id.map(index).to_numpy(),
        to=branches.bus2_id.map(index).to_numpy(),
        x=np.concatenate([x_lines, x_trafos]),
        p=p,
        net=net,
    )


def ptdf_matrix(grid: Grid, slack: int = 0) -> np.ndarray:
    """Nodal PTDF (branches x buses): flow on every branch per MW injected at a bus and taken out at the slack."""
    m = len(grid.frm)
    a = np.zeros((m, grid.n_bus))
    a[np.arange(m), grid.frm] = 1.0
    a[np.arange(m), grid.to] = -1.0
    bf = a / grid.x[:, None]
    bbus = a.T @ bf
    keep = np.arange(grid.n_bus) != slack
    h = np.zeros_like(a)
    h[:, keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], bf[:, keep].T).T
    return h


def zone_of_bus(n_bus: int = 14) -> np.ndarray:
    """Zone index (0, 1, 2 for A, B, C) of every bus."""
    zone = np.empty(n_bus, dtype=int)
    for k, buses in enumerate(ZONES.values()):
        zone[np.array(buses) - 1] = k
    return zone


def glsk(grid: Grid, kind: str) -> np.ndarray:
    """GLSK matrix (buses x zones) whose columns sum to one.

    ``'flat'`` spreads a zone's net position evenly over its buses; ``'generation'`` spreads it in proportion to the
    generators' ``target_p`` where the zone has generation in the base case, and falls back to flat elsewhere.
    """
    zone = zone_of_bus(grid.n_bus)
    g = np.zeros((grid.n_bus, len(ZONES)))
    gen_p = np.zeros(grid.n_bus)
    buses = list(grid.net.get_buses().index)
    for gen in grid.net.get_generators().itertuples():
        gen_p[buses.index(gen.bus_id)] += gen.target_p
    for k in range(len(ZONES)):
        members = zone == k
        weights = gen_p * members if kind == 'generation' and gen_p[members].sum() > 0 else members.astype(float)
        g[:, k] = weights / weights.sum()
    return g


def pypowsybl_zone_factors(grid: Grid, zone_id: str, injection_ids: list[str], keys: list[float]) -> np.ndarray:
    """Zonal PTDF of one zone from pypowsybl's DC sensitivity analysis, one value per branch of ``grid``."""
    analysis = pp.sensitivity.create_dc_analysis()
    analysis.set_zones([pp.sensitivity.create_zone_from_injections_and_shift_keys(zone_id, injection_ids, keys)])
    analysis.add_branch_flow_factor_matrix(branches_ids=grid.names, variables_ids=[zone_id])
    result = analysis.run(grid.net, PARAMETERS)
    return result.get_sensitivity_matrix().loc[zone_id, grid.names].to_numpy()


def net_positions(grid: Grid) -> np.ndarray:
    """Net position of every zone at the reference injections."""
    zone = zone_of_bus(grid.n_bus)
    return np.array([grid.p[zone == k].sum() for k in range(len(ZONES))])


def f_max_rule(f_ref: np.ndarray) -> np.ndarray:
    """The Lab's stand-in for missing operational limits: 1.3 times the base-case flow, at least 20 MW."""
    return np.maximum(F_MAX_FACTOR * np.abs(f_ref), F_MAX_FLOOR)


def cnec_table(grid: Grid, h: np.ndarray, g: np.ndarray, f_max: np.ndarray) -> pd.DataFrame:
    """One CNEC per branch and direction: zonal PTDFs, reference flow, F0 and RAM (all in MW, zone columns per MW)."""
    f_ref = h @ grid.p
    ptdf_z = h @ g
    f_0 = f_ref - ptdf_z @ net_positions(grid)
    frm = FRM_SHARE * f_max
    rows = []
    for sign, direction in ((1.0, '+'), (-1.0, '-')):
        for k, name in enumerate(grid.names):
            rows.append(
                {
                    'cnec': f'{name}{direction}',
                    'ptdf_A': sign * ptdf_z[k, 0],
                    'ptdf_B': sign * ptdf_z[k, 1],
                    'ptdf_C': sign * ptdf_z[k, 2],
                    'f_ref': f_ref[k],
                    'f_max': f_max[k],
                    'f_0': f_0[k],
                    'ram': f_max[k] - frm[k] - sign * f_0[k],
                }
            )
    return pd.DataFrame(rows).set_index('cnec')


def half_planes(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Constraint rows ``a @ (NP_A, NP_B) <= ram`` after eliminating NP_C = -NP_A - NP_B."""
    a = np.column_stack([table.ptdf_A - table.ptdf_C, table.ptdf_B - table.ptdf_C])
    return a, table.ram.to_numpy()


def domain_vertices(a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Vertices, in counter-clockwise order, of the bounded polygon ``{u : a @ u <= rhs}`` in the plane."""
    points = []
    for i, j in itertools.combinations(range(len(rhs)), 2):
        pair = a[[i, j]]
        if abs(np.linalg.det(pair)) < 1e-12:
            continue
        u = np.linalg.solve(pair, rhs[[i, j]])
        if np.all(a @ u <= rhs + 1e-6):
            points.append(u)
    if not points:  # two half-planes exclude each other: no net position satisfies every CNEC
        return np.zeros((0, 2))
    pts = np.unique(np.round(points, 6), axis=0)
    centre = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))]


def binding_cnecs(table: pd.DataFrame, vertices: np.ndarray) -> list[str]:
    """The CNECs whose half-plane boundary carries an edge of the domain."""
    a, rhs = half_planes(table)
    active = np.zeros(len(rhs), dtype=bool)
    for u in vertices:
        active |= np.abs(a @ u - rhs) < 1e-4
    return list(table.index[active])


def max_exchange(table: pd.DataFrame, np_ref: np.ndarray, direction: str) -> tuple[float, str]:
    """Largest extra exchange from the reference point along ``'A>B'`` (A exports, B imports) and its binding CNEC.

    Each CNEC allows ``slack / s`` MW, with ``slack`` its margin at the reference point and ``s`` its sensitivity to
    the exchange; the smallest ratio wins. The result is the flow-based counterpart of an available transfer capacity.
    """
    source, sink = direction.split('>')
    s = table[f'ptdf_{source}'] - table[f'ptdf_{sink}']
    a, rhs = half_planes(table)
    slack = pd.Series(rhs - a @ np_ref[:2], index=table.index)
    positive = s > 1e-9
    limits = slack[positive] / s[positive]
    return float(limits.min()), str(limits.idxmin())


def exchanges(table: pd.DataFrame, np_ref: np.ndarray) -> pd.Series:
    """Largest extra exchange from the reference point in each of the six zone-to-zone directions, MW."""
    return pd.Series({d: max_exchange(table, np_ref, d)[0] for d in DIRECTIONS})


def security_metrics(grid: Grid, h: np.ndarray, f_max: np.ndarray) -> dict[str, float | str]:
    """Worst N-0 loading, worst N-1 loading with its branch and outage, and the N-1 overload energy (Chapter 19)."""
    f_0 = h @ grid.p
    worst_flow = np.abs(f_0).copy()  # per branch, over the base case and every non-islanding outage
    worst, worst_branch, worst_outage = 0.0, '', ''
    for k in range(len(grid.frm)):
        col = h[:, grid.frm[k]] - h[:, grid.to[k]]
        if abs(1.0 - col[k]) < 1e-9:  # bridge: its outage islands part of the grid
            continue
        lodf = col / (1.0 - col[k])
        lodf[k] = -1.0
        flow = np.abs(f_0 + lodf * f_0[k])
        worst_flow = np.maximum(worst_flow, flow)
        b = int((flow / f_max).argmax())
        if flow[b] / f_max[b] > worst:
            worst, worst_branch, worst_outage = float(flow[b] / f_max[b]), grid.names[b], grid.names[k]
    return {
        'n0_worst': float((np.abs(f_0) / f_max).max()),
        'n1_worst': worst,
        'n1_branch': worst_branch,
        'n1_outage': worst_outage,
        'n1_overload_energy': float(np.maximum(worst_flow - f_max, 0.0).sum()),
    }


# --8<-- [start:main]
def topology_action(grid: Grid, f_max: np.ndarray, line: str) -> tuple[Grid, np.ndarray, pd.DataFrame]:
    """Open ``line`` and rebuild the DC model, the PTDF and the CNEC table; the limits stay those of the lines."""
    opened = load_ieee14(open_line=line)
    h_open = ptdf_matrix(opened)
    f_max_open = f_max[[grid.names.index(n) for n in opened.names]]
    return opened, h_open, cnec_table(opened, h_open, glsk(opened, 'generation'), f_max_open)


def rounded(metrics: dict[str, float | str]) -> dict[str, float | str]:
    """The security metrics with three decimals, for printing."""
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}


def main() -> None:
    """Steps 1 to 6 of the Lab: the base topology, then lines 13-14 and 3-4 opened one at a time."""
    grid = load_ieee14()
    h = ptdf_matrix(grid)
    pp.loadflow.run_dc(grid.net, PARAMETERS)
    f_pp = np.concatenate([grid.net.get_lines().p1.to_numpy(), grid.net.get_2_windings_transformers().p1.to_numpy()])
    np_ref = net_positions(grid)
    print('step 1: max |PTDF p - pypowsybl DC flow| =', f'{np.abs(h @ grid.p - f_pp).max():.1e}', 'MW')
    print('        slack generator output', round(grid.p[0], 1), 'MW; net positions A, B, C:', np.round(np_ref, 1))

    g_gen, g_flat = glsk(grid, 'generation'), glsk(grid, 'flat')
    print('step 2: generation GLSK of zone A over buses 1-5:', np.round(g_gen[:5, 0], 3))
    print('        flat GLSK of zone B over buses 6, 11, 12, 13:', np.round(g_flat[[5, 10, 11, 12], 1], 3))

    gens = grid.net.get_generators().loc[['B1-G', 'B2-G'], 'target_p']
    zone_a_pp = pypowsybl_zone_factors(grid, 'A', list(gens.index), list(gens / gens.sum()))
    print('step 3: max |zonal PTDF of A (NumPy) - pypowsybl| =', f'{np.abs(h @ g_gen[:, 0] - zone_a_pp).max():.1e}')
    for k in ('L1-2-1', 'L1-5-1', 'L2-3-1', 'T5-6-1'):
        i = grid.names.index(k)
        print(f'        {k}: generation GLSK {np.round(h[i] @ g_gen, 3)}, flat GLSK {np.round(h[i] @ g_flat, 3)}')

    f_max = f_max_rule(h @ grid.p)
    table = cnec_table(grid, h, g_gen, f_max)
    print('step 4: CNECs with the smallest RAM (MW):')
    print(table.sort_values('ram').head(4).round(3).to_string())

    vertices = domain_vertices(*half_planes(table))
    print(f'step 5: the domain has {len(vertices)} vertices; binding CNECs: {binding_cnecs(table, vertices)}')
    print('        vertices (NP_A, NP_B):', np.round(vertices, 1).tolist())
    print('        reference point (NP_A, NP_B) =', np.round(np_ref[:2], 1))
    print('        extra exchange from the reference point (MW):', exchanges(table, np_ref).round(1).to_dict())
    print('        base topology:', rounded(security_metrics(grid, h, f_max)))

    opened, h_open, table_open = topology_action(grid, f_max, 'L13-14-1')
    vertices_open = domain_vertices(*half_planes(table_open))
    print(f'step 6: L13-14-1 open: {len(vertices_open)} vertices; binding: {binding_cnecs(table_open, vertices_open)}')
    both = pd.DataFrame(
        {'base': exchanges(table, np_ref), 'L13-14 open': exchanges(table_open, net_positions(opened))}
    )
    both['change'] = both['L13-14 open'] - both['base']
    print(both.round(1).to_string())
    print(
        '        L13-14 open:',
        rounded(security_metrics(opened, h_open, table_open.f_max.to_numpy()[: len(opened.names)])),
    )

    opened, h_open, table_open = topology_action(grid, f_max, 'L3-4-1')
    print(f'        L3-4-1 open instead: {len(domain_vertices(*half_planes(table_open)))} vertices (empty domain)')
    print(table_open.loc[table_open.ram < 0].round(3).to_string())
    print(
        '        L3-4 open:',
        rounded(security_metrics(opened, h_open, table_open.f_max.to_numpy()[: len(opened.names)])),
    )


# --8<-- [end:main]


if __name__ == '__main__':
    main()
