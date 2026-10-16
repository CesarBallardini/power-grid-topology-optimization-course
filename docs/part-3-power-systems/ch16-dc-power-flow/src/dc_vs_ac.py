"""DC flows of a pandapower case against ``rundcpp`` and ``runpp`` (Chapter 16, Lab steps 2 and 3).

The branch table (from, to, x, tap, shift) comes from the internal MATPOWER-style structure that pandapower builds
for its own solvers (``net._ppc``), so the per-unit base is pandapower's ``sn_mva`` and the orientation is
pandapower's (lines: from bus to to bus; transformers: hv bus to lv bus).
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
from dc_power_flow import dc_power_flow

F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, PF = 0, 1, 2, 3, 8, 9, 13


def branch_table(net: pp.pandapowerNet) -> pd.DataFrame:
    """Branches of a solved network: from, to, r, x, tap, shift (deg) and the DC flow pandapower computed (MW)."""
    br = net._ppc['branch'].real
    names = [f'line {i}' for i in net.line.index] + [f'trafo {i}' for i in net.trafo.index]
    table = pd.DataFrame(
        {
            'from': br[:, F_BUS].astype(int),
            'to': br[:, T_BUS].astype(int),
            'r': br[:, BR_R],
            'x': br[:, BR_X],
            'tap': np.where(br[:, TAP] == 0, 1.0, br[:, TAP]),
            'shift': br[:, SHIFT],
            'p_pandapower': br[:, PF],
        },
        index=names,
    )
    table['kind'] = ['line'] * len(net.line) + ['trafo'] * len(net.trafo)
    return table


def dc_case(name: str) -> tuple[pp.pandapowerNet, pd.DataFrame, np.ndarray, int]:
    """Run ``rundcpp`` on ``pandapower.networks.<name>()``: the net, the branch table, injections (MW) and slack."""
    net = getattr(pn, name)()
    pp.rundcpp(net)
    table = branch_table(net)
    p = -net.res_bus.p_mw.to_numpy()  # res_bus is a demand: minus it is the net injection, balanced in DC
    return net, table, p, int(net.ext_grid.bus.iloc[0])


def own_dc_flows(table: pd.DataFrame, p: np.ndarray, slack: int, with_taps: bool = True) -> np.ndarray:
    """DC flows (MW) with the worked example's function, with or without the tap ratio in b = 1/(x tau)."""
    x = table['x'].to_numpy() * (table['tap'].to_numpy() if with_taps else 1.0)
    return dc_power_flow(table['from'].to_numpy(), table['to'].to_numpy(), x, p, slack)


def dc_ac_frame(name: str) -> pd.DataFrame:
    """One row per branch: DC flow, AC flow, AC loading (%), r/x and the relative DC error against the AC flow."""
    net, table, p, slack = dc_case(name)
    table['p_dc'] = own_dc_flows(table, p, slack)
    pp.runpp(net)
    table['p_ac'] = np.concatenate([net.res_line.p_from_mw.to_numpy(), net.res_trafo.p_hv_mw.to_numpy()])
    table['loading_ac'] = np.concatenate(
        [net.res_line.loading_percent.to_numpy(), net.res_trafo.loading_percent.to_numpy()]
    )
    table['losses_ac'] = np.concatenate([net.res_line.pl_mw.to_numpy(), net.res_trafo.pl_mw.to_numpy()])
    table['r_over_x'] = table['r'] / table['x']
    table['abs_error'] = (table['p_dc'] - table['p_ac']).abs()
    table['rel_error'] = table['abs_error'] / table['p_ac'].abs()
    return table


def main() -> None:
    """Print the comparison of Lab steps 2 and 3 for case57."""
    pd.set_option('display.width', 140)
    _, table, p, slack = dc_case('case57')
    n_tap = int((table['tap'] != 1).sum())
    print(
        f'case57: {len(table)} branches, {n_tap} with a tap ratio, shifts {table["shift"].unique()}, slack bus {slack}'
    )
    no_tap, with_tap = own_dc_flows(table, p, slack, with_taps=False), own_dc_flows(table, p, slack)
    ref = table['p_pandapower'].to_numpy()
    err_no_tap, err_tap = np.abs(no_tap - ref).max(), np.abs(with_tap - ref).max()
    print(f'Step 2: max |own - rundcpp| ignoring taps: {err_no_tap:.3f} MW; with b = 1/(x tau): {err_tap:.1e} MW')
    worst = int(np.abs(no_tap - ref).argmax())
    tap = table['tap'].iloc[worst]
    print(f'  largest tap effect on {table.index[worst]} (tap {tap:.3f}): {no_tap[worst]:.2f} vs {ref[worst]:.2f} MW')
    print(f'  for the Octave comparison (step 5): first five rundcpp flows {np.round(ref[:5], 4)}, ', end='')
    print(f'sum |F| {np.abs(ref).sum():.4f}, max |F| {np.abs(ref).max():.4f} MW')
    frame = dc_ac_frame('case57')
    err = frame['abs_error']
    print(f'Step 3: AC losses {frame["losses_ac"].sum():.2f} MW of {-p[p < 0].sum():.1f} MW of load; ', end='')
    print(f'mean |DC - AC| {err.mean():.2f} MW, max {err.max():.2f} MW on {err.idxmax()}')
    heavy = frame[frame['p_ac'].abs() > 10]
    rel = 100 * heavy['rel_error']
    print(
        f'  branches above 10 MW: median relative error {rel.median():.1f} %, max {rel.max():.1f} % ({rel.idxmax()})'
    )
    print(f'  correlation of |DC - AC| with |AC flow|: {err.corr(frame["p_ac"].abs()):.2f}; ', end='')
    print(f'of the relative error with r/x (branches above 10 MW): {heavy["rel_error"].corr(heavy["r_over_x"]):.2f}')
    lines = frame[frame['kind'] == 'line']
    n_trafo_r = int((frame.loc[frame['kind'] == 'trafo', 'r'] > 0).sum())
    print(f'  r/x of the lines: {lines["r_over_x"].min():.2f} to {lines["r_over_x"].max():.2f}; ', end='')
    print(f'transformers with r > 0: {n_trafo_r}')
    print('  five worst absolute errors:')
    print(frame.nlargest(5, 'abs_error')[['kind', 'p_dc', 'p_ac', 'loading_ac', 'r_over_x', 'rel_error']].round(3))


if __name__ == '__main__':
    main()
