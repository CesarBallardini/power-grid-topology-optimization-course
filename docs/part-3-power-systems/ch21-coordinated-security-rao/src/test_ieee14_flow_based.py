import numpy as np
import pypowsybl as pp
import pytest
from ieee14_flow_based import (
    PARAMETERS,
    binding_cnecs,
    cnec_table,
    domain_vertices,
    exchanges,
    f_max_rule,
    glsk,
    half_planes,
    load_ieee14,
    max_exchange,
    net_positions,
    ptdf_matrix,
    pypowsybl_zone_factors,
    security_metrics,
    topology_action,
)


@pytest.fixture(scope='module')
def base():
    grid = load_ieee14()
    h = ptdf_matrix(grid)
    f_max = f_max_rule(h @ grid.p)
    return grid, h, f_max, cnec_table(grid, h, glsk(grid, 'generation'), f_max)


def test_ptdf_reproduces_pypowsybl_dc_flows(base):
    grid, h, _, _ = base
    pp.loadflow.run_dc(grid.net, PARAMETERS)
    f_pp = np.concatenate([grid.net.get_lines().p1.to_numpy(), grid.net.get_2_windings_transformers().p1.to_numpy()])
    np.testing.assert_allclose(h @ grid.p, f_pp, atol=1e-9)
    assert grid.p.sum() == pytest.approx(0.0, abs=1e-9)
    assert grid.p[0] == pytest.approx(219.0)


def test_glsk_columns_sum_to_one_and_follow_generation(base):
    grid = base[0]
    for kind in ('generation', 'flat'):
        g = glsk(grid, kind)
        np.testing.assert_allclose(g.sum(axis=0), 1.0)
        assert (g >= 0).all()
    g = glsk(grid, 'generation')
    np.testing.assert_allclose(g[:2, 0], [232.4 / 272.4, 40 / 272.4])
    np.testing.assert_allclose(g[[5, 10, 11, 12], 1], 0.25)  # zone B has no generation: flat


def test_zonal_ptdf_matches_pypowsybl(base):
    grid, h, _, _ = base
    gens = grid.net.get_generators().loc[['B1-G', 'B2-G'], 'target_p']
    from_pp = pypowsybl_zone_factors(grid, 'A', list(gens.index), list(gens / gens.sum()))
    np.testing.assert_allclose(h @ glsk(grid, 'generation')[:, 0], from_pp, atol=1e-12)


def test_net_positions_balance_and_reference_point_is_inside(base):
    grid, _, _, table = base
    np_ref = net_positions(grid)
    assert np_ref.sum() == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(np_ref, [87.7, -34.3, -53.4])
    a, rhs = half_planes(table)
    assert np.all(a @ np_ref[:2] <= rhs + 1e-9)
    assert table.ram.min() < 0  # L3-4-1 in the negative direction: the zero-exchange point is outside


def test_domain_and_exchanges(base):
    grid, _, _, table = base
    vertices = domain_vertices(*half_planes(table))
    assert len(vertices) == 5
    assert set(binding_cnecs(table, vertices)) == {'L10-11-1+', 'T4-7-1+', 'T5-6-1+', 'L3-4-1-', 'L10-11-1-'}
    value, cnec = max_exchange(table, net_positions(grid), 'A>B')
    assert value == pytest.approx(16.5, abs=0.05)
    assert cnec == 'T5-6-1+'
    assert (exchanges(table, net_positions(grid)) > 0).all()


def test_security_metrics_of_base_topology(base):
    grid, h, f_max, _ = base
    m = security_metrics(grid, h, f_max)
    assert m['n0_worst'] == pytest.approx(1 / 1.3)
    assert m['n1_worst'] == pytest.approx(3.0, abs=0.01)
    assert (m['n1_branch'], m['n1_outage']) == ('L3-4-1', 'L2-3-1')


def test_opening_13_14_keeps_a_domain_and_3_4_empties_it(base):
    grid, _, f_max, table = base
    opened, _, table_open = topology_action(grid, f_max, 'L13-14-1')
    assert len(domain_vertices(*half_planes(table_open))) == 5
    change = exchanges(table_open, net_positions(opened)) - exchanges(table, net_positions(grid))
    assert change['A>B'] > 0 > change['B>C']
    _, _, table_34 = topology_action(grid, f_max, 'L3-4-1')
    assert len(domain_vertices(*half_planes(table_34))) == 0
    assert table_34.loc['L2-3-1+', 'ram'] < 0
    np.testing.assert_allclose(table_34.loc['L2-3-1+', ['ptdf_A', 'ptdf_B', 'ptdf_C']], 0.0, atol=1e-12)
