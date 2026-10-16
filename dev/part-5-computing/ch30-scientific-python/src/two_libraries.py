"""The same two operations in pandapower and pypowsybl: load a case, run a power flow, read a result table."""

import pandapower as pp
import pandapower.networks as pn
import pypowsybl as pypo


def pandapower_case57():
    """IEEE 57-bus case: element counts, then AC and DC from-end active flows per line (pandas Series)."""
    net = pn.case57()  # element tables are pandas DataFrames
    counts = (len(net.bus), len(net.line), len(net.trafo))
    pp.runpp(net)  # AC power flow; results land in net.res_line, net.res_bus, ...
    ac_p = net.res_line.p_from_mw.copy()
    pp.rundcpp(net)  # DC power flow overwrites the same result tables
    dc_p = net.res_line.p_from_mw
    return counts, ac_p, dc_p


def pypowsybl_ieee14():
    """IEEE 14-bus case in pypowsybl: the load flow status and the line results p1, i1."""
    n = pypo.network.create_ieee14()
    result = pypo.loadflow.run_ac(n)
    lines = n.get_lines()[['p1', 'i1']]  # results are columns of the element tables
    return result[0].status, lines


def main() -> None:
    """Run both and print the headline numbers."""
    counts, ac_p, dc_p = pandapower_case57()
    print(counts)
    print((ac_p - dc_p).abs().max())  # largest AC-DC difference of a line flow, MW
    status, lines = pypowsybl_ieee14()
    print(status)  # ComponentStatus.CONVERGED
    print(lines.head())


if __name__ == '__main__':
    main()
