import pypowsybl as pypo
from two_libraries import pandapower_case57, pypowsybl_ieee14


def test_pandapower_case57_ac_and_dc_agree_roughly():
    counts, ac_p, dc_p = pandapower_case57()
    assert counts[0] == 57 and counts[1] + counts[2] == 80
    assert (ac_p - dc_p).abs().max() < 25  # DC drops losses and reactive flow, but is in the same ballpark


def test_pypowsybl_ieee14_converges():
    status, lines = pypowsybl_ieee14()
    assert status == pypo.loadflow.ComponentStatus.CONVERGED
    assert len(lines) == 17 and lines['i1'].gt(0).all()
