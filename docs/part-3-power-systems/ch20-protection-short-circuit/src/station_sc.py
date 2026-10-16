"""Lab Part A of Chapter 20: the worked example's double-busbar station in pandapower, with ``calc_sc`` and ``runpp``.

Two 220 kV infeeds (external grids G1 and G2) reach the station busbars S1 and S2 over 40 km and 60 km of
j0.4 ohm/km line; S1 and S2 are joined by a bus-bus switch of type CB (the coupler). The extra paths the hand
calculation neglects are two 30 km lines from S1 and S2 to a load bus L and a 150 km line G1-G2.
"""

import pandapower as pp
import pandapower.shortcircuit as sc
import pandas as pd

BUSES = ('G1', 'G2', 'S1', 'S2', 'L')


def station(coupler_closed: bool = True, extra_paths: bool = True, r_ohm_per_km: float = 0.03) -> pp.pandapowerNet:
    """Build the station grid; ``extra_paths=False`` and ``r_ohm_per_km=0`` reproduce the hand calculation."""
    net = pp.create_empty_network(sn_mva=100)
    bus = {name: pp.create_bus(net, vn_kv=220, name=name) for name in BUSES}
    pp.create_ext_grid(net, bus['G1'], vm_pu=1.0, s_sc_max_mva=8000, s_sc_min_mva=6000, rx_max=0.1, rx_min=0.1)
    pp.create_ext_grid(net, bus['G2'], vm_pu=1.0, s_sc_max_mva=6000, s_sc_min_mva=4500, rx_max=0.1, rx_min=0.1)

    def line(a: str, b: str, km: float) -> None:
        pp.create_line_from_parameters(
            net,
            bus[a],
            bus[b],
            length_km=km,
            r_ohm_per_km=r_ohm_per_km,
            x_ohm_per_km=0.4,
            c_nf_per_km=0,
            max_i_ka=1.0,
            endtemp_degree=80,  # needed by the minimum case (line resistance at end temperature)
            name=f'{a}-{b}',
        )

    line('G1', 'S1', 40)
    line('G2', 'S2', 60)
    if extra_paths:
        line('S1', 'L', 30)
        line('S2', 'L', 30)
        line('G1', 'G2', 150)
        pp.create_load(net, bus['L'], p_mw=200, q_mvar=50)
    pp.create_switch(net, bus['S1'], bus['S2'], et='b', type='CB', closed=coupler_closed, name='coupler')
    return net


def fault_levels(net: pp.pandapowerNet, case: str = 'max') -> pd.DataFrame:
    """``calc_sc`` with peak current and no branch results; ``ikss_ka``, ``skss_mw`` and ``ip_ka`` per bus."""
    sc.calc_sc(net, case=case, ip=True, branch_results=False)
    table = net.res_bus_sc[['ikss_ka', 'skss_mw', 'ip_ka']].copy()
    table.index = net.bus.name
    return table


def station_table(extra_paths: bool = True, r_ohm_per_km: float = 0.03) -> pd.DataFrame:
    """Steps 2 and 3: I''k at S1 and S2 for both cases, coupler closed and open."""
    rows = {}
    for case in ('max', 'min'):
        for closed in (True, False):
            table = fault_levels(station(closed, extra_paths, r_ohm_per_km), case)
            rows[(case, 'closed' if closed else 'open')] = table.loc[['S1', 'S2'], 'ikss_ka']
    return pd.DataFrame(rows).T


def recoupling_angle() -> tuple[float, float]:
    """Step 4: voltage angle difference S1 - S2 in degrees with the coupler closed and open (AC power flow)."""
    out = []
    for closed in (True, False):
        net = station(closed)
        pp.runpp(net)
        va = net.res_bus.va_degree
        out.append(float(va[net.bus.name == 'S1'].iloc[0] - va[net.bus.name == 'S2'].iloc[0]))
    return out[0], out[1]


def verdict(ik_max_ka: float, ik_min_ka: float, breaker_ka: float, protection_min_ka: float) -> str:
    """Step 5: is a busbar operable with the given maximum and minimum fault currents?"""
    problems = []
    if ik_max_ka > breaker_ka:
        problems.append(f"I''k max {ik_max_ka:.2f} kA exceeds the {breaker_ka:.0f} kA breaker rating")
    if ik_min_ka < protection_min_ka:
        problems.append(f"I''k min {ik_min_ka:.2f} kA is below the {protection_min_ka:.0f} kA the protection needs")
    return 'acceptable' if not problems else '; '.join(problems)


def main() -> None:
    """Print steps 2 to 5 of Lab Part A."""
    pd.set_option('display.width', 120)
    print("full grid (r = 0.03 ohm/km, extra paths), I''k in kA at S1 and S2:")
    print(station_table().round(3))
    for closed in (True, False):
        print('maximum case, coupler', 'closed' if closed else 'open', '(kA, MVA, kA):')
        print(fault_levels(station(closed), 'max').loc[['S1', 'S2']].round(2))
    print('hand-calculation grid (no extra paths, r = 0):')
    print(station_table(extra_paths=False, r_ohm_per_km=0.0).round(3))
    closed, opened = recoupling_angle()
    print(f'angle S1 - S2: {closed:.3f} deg with the coupler closed, {opened:.3f} deg open')
    full = station_table()
    for busbar in ('S1', 'S2'):
        ik_max, ik_min = full.loc[('max', 'open'), busbar], full.loc[('min', 'open'), busbar]
        print(f'split, {busbar}: 8 kA breakers, 5 kA minimum for protection -> {verdict(ik_max, ik_min, 8.0, 5.0)}')
    ik_max, ik_min = full.loc[('max', 'closed'), 'S1'], full.loc[('min', 'closed'), 'S1']
    print(f'coupled: 8 kA breakers, 5 kA minimum -> {verdict(ik_max, ik_min, 8.0, 5.0)}')


if __name__ == '__main__':
    main()
