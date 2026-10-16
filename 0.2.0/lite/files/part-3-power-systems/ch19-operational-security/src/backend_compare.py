"""Lab Part B, step 5 of Chapter 19: the same single-branch AC N-1 of IEEE 14 in pandapower and in pypowsybl.

pandapower's ``case14`` is exported with ``to_mpc`` and loaded by pypowsybl from the MATPOWER file, as
``test_backend_comparison.py`` in ToOp does. Both engines then outage every branch one at a time; the branch flows
``p`` at the from side are compared per (contingency, branch). Needs only pandapower and pypowsybl.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import pypowsybl
import pypowsybl.security as security
from pandapower.converter.matpower import to_mpc

# One slack at bus 1 and no reactive limits, as pandapower's runpp does by default (enforce_q_lims=False).
LF_PARAMS = pypowsybl.loadflow.Parameters(distributed_slack=False, use_reactive_limits=False)


def branch_ids(net: pp.pandapowerNet) -> list[str]:
    """Pypowsybl's id of every pandapower branch, lines first then transformers: ``LINE-<from>-<to>`` style."""
    ids = [f'{f + 1}-{t + 1}' for f, t in zip(net.line.from_bus, net.line.to_bus, strict=True)]
    ids += [f'{h + 1}-{lv + 1}' for h, lv in zip(net.trafo.hv_bus, net.trafo.lv_bus, strict=True)]
    return ids


def powsybl_network(net: pp.pandapowerNet) -> pypowsybl.network.Network:
    """Round trip through a MATPOWER file (``to_mpc`` wants a solved network to initialise voltages from)."""
    pp.rundcpp(net)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'ieee14.mat'
        to_mpc(net, str(path))
        return pypowsybl.network.load(str(path))


def powsybl_ids(pnet: pypowsybl.network.Network) -> dict[str, str]:
    """Map ``<from>-<to>`` to pypowsybl's element id (``LINE-4-7`` or ``TWT-4-7``, depending on the tap ratio)."""
    out = {}
    for prefix, table in (('LINE-', pnet.get_lines()), ('TWT-', pnet.get_2_windings_transformers())):
        for element in table.index:
            out[element.removeprefix(prefix)] = element
    return out


def pandapower_n1(net: pp.pandapowerNet) -> pd.DataFrame:
    """Flows ``p`` at the from/hv side in MW: one row per outaged branch, one column per branch, NaN when diverged."""
    ids = branch_ids(net)
    rows = {}
    for k, key in enumerate(ids):
        table, idx = ('line', k) if k < len(net.line) else ('trafo', k - len(net.line))
        net[table].at[idx, 'in_service'] = False
        try:
            pp.runpp(net)
            flows = np.concatenate([net.res_line.p_from_mw.to_numpy(), net.res_trafo.p_hv_mw.to_numpy()])
        except pp.LoadflowNotConverged:
            flows = np.full(len(ids), np.nan)
        net[table].at[idx, 'in_service'] = True
        flows[k] = np.nan
        rows[key] = flows
    return pd.DataFrame(rows, index=ids).T


def powsybl_n1(pnet: pypowsybl.network.Network, ids: list[str]) -> pd.DataFrame:
    """The same table from pypowsybl's security analysis, columns in the pandapower order."""
    names = powsybl_ids(pnet)
    analysis = security.create_analysis()
    analysis.add_single_element_contingencies([names[i] for i in ids])
    analysis.add_monitored_elements(branch_ids=[names[i] for i in ids])
    result = analysis.run_ac(pnet, LF_PARAMS)
    table = result.branch_results.reset_index()
    out = pd.DataFrame(np.nan, index=ids, columns=ids)
    for key in ids:
        rows = table[table.contingency_id == names[key]].set_index('branch_id')
        for col in ids:
            if names[col] in rows.index:
                out.at[key, col] = rows.at[names[col], 'p1']
    return out


def compare() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Both N-1 tables and the absolute differences per (contingency, branch), as a flat series without NaNs."""
    net = pn.case14()
    pnet = powsybl_network(net)
    pp_table = pandapower_n1(net)
    py_table = powsybl_n1(pnet, branch_ids(net))
    diff = (pp_table - py_table).abs().stack()
    return pp_table, py_table, diff


def main() -> None:
    """Print the distribution of the branch-flow differences between the two engines."""
    pp_table, py_table, diff = compare()
    print(f'{pp_table.shape[0]} contingencies x {pp_table.shape[1]} branches; compared entries: {len(diff)}')
    print(
        '|p_pandapower - p_pypowsybl| in MW: quantiles 50 / 90 / 99 / max =',
        np.round(diff.quantile([0.5, 0.9, 0.99]).to_list() + [diff.max()], 4),
    )
    worst = diff.idxmax()
    print(
        f'largest difference: outage {worst[0]}, branch {worst[1]}: '
        f'{pp_table.loc[worst]:.3f} vs {py_table.loc[worst]:.3f} MW'
    )
    print('contingencies not converged in pandapower:', pp_table.index[pp_table.isna().all(axis=1)].tolist())
    print('entries missing in pypowsybl (outaged branch itself or disconnected):', int(py_table.isna().sum().sum()))


if __name__ == '__main__':
    main()
