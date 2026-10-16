"""Exercise 3 of Chapter 20: maximum and minimum short-circuit currents of pandapower's ``case9``.

``case9`` ships without short-circuit data. ``with_sc_data`` adds what ``calc_sc`` needs: the external grid's
``s_sc_max_mva``, ``s_sc_min_mva``, ``rx_max`` and ``rx_min``; the generators' ``vn_kv``, ``sn_mva``, ``xdss_pu``,
``rdss_ohm`` and ``cos_phi``; and the lines' ``endtemp_degree`` for the minimum case. The voltage factor is c_max
= 1.1 for the maximum case and c_min = 1.0 for the minimum case at this voltage level (IEC 60909 table 1).
"""

import pandapower as pp
import pandapower.networks as pn
import pandapower.shortcircuit as sc
import pandas as pd


def with_sc_data(
    s_sc_max_mva: float = 5000.0,
    s_sc_min_mva: float = 3000.0,
    gen_sn_mva: float = 250.0,
    xdss_pu: float = 0.2,
    endtemp_degree: float = 80.0,
) -> pp.pandapowerNet:
    """``case9`` with the short-circuit parameters of the external grid, the generators and the lines."""
    net = pn.case9()
    net.ext_grid['s_sc_max_mva'] = s_sc_max_mva
    net.ext_grid['s_sc_min_mva'] = s_sc_min_mva
    net.ext_grid['rx_max'] = 0.1
    net.ext_grid['rx_min'] = 0.1
    net.gen['vn_kv'] = net.bus.vn_kv[net.gen.bus].to_numpy()
    net.gen['sn_mva'] = gen_sn_mva
    net.gen['xdss_pu'] = xdss_pu
    net.gen['rdss_ohm'] = 0.05
    net.gen['cos_phi'] = 0.85
    net.line['endtemp_degree'] = endtemp_degree
    return net


def max_min_table(net: pp.pandapowerNet | None = None) -> pd.DataFrame:
    """``ikss_ka`` for ``case="max"`` and ``case="min"`` at every bus, and the ratio min/max."""
    net = with_sc_data() if net is None else net
    out = {}
    for case in ('max', 'min'):
        sc.calc_sc(net, case=case, ip=True, branch_results=False)
        out[case] = net.res_bus_sc.ikss_ka.copy()
    table = pd.DataFrame(out)
    table['min/max'] = table['min'] / table['max']
    return table


def endtemp_effect(temps=(20.0, 80.0, 160.0)) -> pd.Series:
    """Minimum-case I''k at bus 5 (0-based 4, in the middle of a long line) for several line end temperatures."""
    out = {}
    for t in temps:
        net = with_sc_data(endtemp_degree=t)
        sc.calc_sc(net, case='min', branch_results=False)
        out[t] = float(net.res_bus_sc.ikss_ka[4])
    return pd.Series(out, name='ikss_min_bus5_ka')


def main() -> None:
    """Print the comparison and the role of the end temperature."""
    table = max_min_table()
    print("case9 (345 kV): I''k in kA, max (c = 1.1, S''kQ 5000 MVA) and min (c = 1.0, S''kQ 3000 MVA, hot lines)")
    print(table.round(3).to_string())
    print(
        'ratio min/max: smallest',
        round(table['min/max'].min(), 3),
        'at bus',
        int(table['min/max'].idxmin()),
        '; largest',
        round(table['min/max'].max(), 3),
        'at bus',
        int(table['min/max'].idxmax()),
    )
    print(
        'minimum case at bus 5 for line end temperatures 20, 80, 160 degC (kA):', endtemp_effect().round(4).to_dict()
    )


if __name__ == '__main__':
    main()
