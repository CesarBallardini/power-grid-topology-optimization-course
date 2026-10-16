import numpy as np
import pytest
from ieee30_ots import big_m, brute_force, data, dc_opf, lazy_scopf, ots, post_contingency_loading


@pytest.fixture(scope='module')
def case():
    return data()


@pytest.fixture(scope='module')
def opf(case):
    return dc_opf(case)


def test_data(case):
    assert len(case['gens']) == 6 and case['load'].sum() == pytest.approx(189.2)
    assert [c for _, _, c in case['gens']] == pytest.approx([20, 17.5, 10, 32.5, 30, 30])


def test_dc_opf_matches_the_dc_power_flow(case, opf):
    assert opf['status'] == 'Optimal'
    assert opf['cost'] == pytest.approx(3101.0, abs=0.1)
    assert opf['p'].sum() == pytest.approx(case['load'].sum())
    from ch27_ieee30 import ptdf  # noqa: PLC0415

    gen_bus = np.zeros(30)
    np.add.at(gen_bus, [bus for bus, _, _ in case['gens']], opf['p'])
    np.testing.assert_allclose(opf['f'], ptdf(case, np.ones(41, dtype=bool)) @ (gen_bus - case['load']), atol=1e-8)
    loading = 100 * np.abs(opf['f']) / case['rating']
    assert np.flatnonzero(loading > 100 - 1e-6).tolist() == [30]  # line 22-24 binding


def test_lmps_differ_only_because_of_congestion(case, opf):
    assert opf['lmp'][21] == pytest.approx(11.95, abs=0.01)  # bus 22, behind the congested line
    assert opf['lmp'][23] == pytest.approx(38.88, abs=0.01)  # bus 24, in front of it
    assert opf['lmp'][0] == pytest.approx(20.0, abs=1e-6)  # the slack bus generator is marginal
    relaxed = {**case, 'rating': np.full(41, 1e4)}
    np.testing.assert_allclose(dc_opf(relaxed)['lmp'], 20.0, atol=1e-6)  # uncongested: one price, the marginal unit


def test_big_m_bounds(case):
    cycle, global_ = big_m(case, 'cycle'), big_m(case, 'global')
    assert (cycle <= global_ + 1e-9).all()
    assert cycle.min() > 0


def test_switching_lowers_the_cost(case, opf):
    r = ots(case, 1, list(range(41)))
    assert r['status'] == 'Optimal' and r['gap'] == pytest.approx(0, abs=1e-6)
    assert r['cost'] == pytest.approx(3084.0, abs=0.1)
    assert r['cost'] < opf['cost'] and len(r['switched']) == 1
    assert ots(case, 0, list(range(41)))['cost'] == pytest.approx(opf['cost'], abs=1e-6)


def test_brute_force_agrees_with_the_milp(case):
    best, arg, count, _ = brute_force(case, 1, list(range(41)))
    assert count == 42 and len(arg) == 1
    assert best == pytest.approx(3084.0, abs=0.1)


def test_lazy_scopf_terminates_secure(case, opf):
    assert np.nanmax(post_contingency_loading(case, opf['f'])) > 100
    r, history = lazy_scopf(case)
    assert len(history) == 2 and history[-1]['violated'] == []
    assert history[-1]['worst_loading'] <= 100 + 1e-6
    assert r['cost'] == pytest.approx(3369.2, abs=0.1) and r['cost'] > opf['cost']
