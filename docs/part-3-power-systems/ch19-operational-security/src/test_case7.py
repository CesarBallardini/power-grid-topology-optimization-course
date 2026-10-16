import numpy as np
import pytest
from case7 import (
    LIMIT,
    N_LINE,
    NAMES,
    binding,
    dispatch,
    flows,
    lodf,
    n1_loading,
    pandapower_n1,
    pandapower_net,
    split_bus,
)


def test_opf_and_scopf_of_the_worked_example():
    opf, scopf = dispatch(), dispatch(range(N_LINE))
    assert opf.fun == pytest.approx(4300) and scopf.fun == pytest.approx(4400)
    np.testing.assert_allclose(opf.x, [250, 90, 0], atol=1e-6)
    np.testing.assert_allclose(scopf.x, [240, 100, 0], atol=1e-6)
    assert n1_loading(opf.x).max() == pytest.approx(250 / 240 * 100)
    assert n1_loading(scopf.x).max() == pytest.approx(100)
    assert binding(scopf.x) == [('L1-2', 'L1-3'), ('L1-3', 'L1-2')]


@pytest.mark.parametrize('k', range(N_LINE))
def test_lodf_post_contingency_flows_satisfy_kirchhoff(k):
    pg = dispatch().x
    f = flows(pg, out=k)
    assert f[k] == 0
    inj = np.zeros(7)
    for g, bus in zip(pg, (1, 2, 7), strict=True):
        inj[bus - 1] += g
    inj -= np.array([0, 0, 80, 100, 90, 70, 0.0])
    a = np.zeros((N_LINE, 7))
    for i, (frm, to) in enumerate([(n.split('-')[0][1:], n.split('-')[1]) for n in NAMES]):
        a[i, int(frm) - 1], a[i, int(to) - 1] = 1, -1
    np.testing.assert_allclose(a.T @ f, inj, atol=1e-9)
    # the other lines pick up the lost flow in proportion to the LODF column
    base = flows(pg)
    keep = np.arange(N_LINE) != k
    np.testing.assert_allclose(f[keep], base[keep] + lodf(k)[keep] * base[k], atol=1e-9)


def test_dc_flows_agree_with_pandapower():
    pg = dispatch().x
    net = pandapower_net(pg)
    import pandapower as pp

    pp.rundcpp(net)
    np.testing.assert_allclose(net.res_line.p_from_mw.to_numpy(), flows(pg), atol=1e-6)


def test_split_bus_moves_the_lines_and_lowers_line_3_6():
    lim = LIMIT.copy()
    lim[NAMES.index('L3-6')] = 100
    pg = dispatch(range(N_LINE)).x
    assert n1_loading(pg, lim).max() == pytest.approx(111.5, abs=0.05)
    net = split_bus(pandapower_net(pg, lim), 6, ['L3-6', 'L6-7'])
    on_new_bus = net.line.index[(net.line.from_bus == 7) | (net.line.to_bus == 7)]
    assert len(net.bus) == 8 and list(on_new_bus) == [NAMES.index('L3-6'), NAMES.index('L6-7')]
    worst = pandapower_n1(net)
    assert worst[NAMES.index('L3-6')] == pytest.approx(60.2, abs=0.05)
    assert (100 * worst / lim).max() == pytest.approx(100, abs=1e-6)
