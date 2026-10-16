import grid_graph as gg
import matplotlib
import networkx as nx
import numpy as np
import pytest

matplotlib.use('Agg')


@pytest.fixture(scope='module')
def ieee14():
    return gg.build_graph(gg.load_edges())


def test_edge_list_builds_the_14_bus_multigraph(ieee14):
    assert isinstance(ieee14, nx.MultiGraph)
    assert ieee14.number_of_nodes() == 14 and ieee14.number_of_edges() == 20


def test_one_bridge_and_one_articulation_point(ieee14):
    assert gg.bridges_and_articulation_points(ieee14) == ([(7, 8)], [7])


def test_parallel_circuit_removes_the_bridge(ieee14):
    g = ieee14.copy()
    g.add_edge(7, 8)
    assert list(nx.bridges(g)) == []
    assert g.number_of_edges() == 21


def test_zero_eigenvalues_count_islands(ieee14):
    assert gg.count_zero_eigenvalues(ieee14) == 1
    cut = ieee14.copy()
    cut.remove_edge(7, 8)
    assert gg.count_zero_eigenvalues(cut) == 2
    assert nx.number_connected_components(cut) == 2


def test_effective_resistance_is_one_only_across_the_bridge(ieee14):
    reff = gg.effective_resistances(ieee14).set_index(['from_bus', 'to_bus'])['r_eff']
    assert reff[(7, 8)] == pytest.approx(1.0)
    assert (reff.drop((7, 8)) < 1).all()
    simple = nx.Graph(ieee14)
    for (u, v), r in reff.items():
        assert r == pytest.approx(nx.resistance_distance(simple, u, v))


def test_algebraic_connectivity_drops_to_zero_without_the_bridge(ieee14):
    base = gg.algebraic_connectivity(ieee14)
    assert base == pytest.approx(nx.algebraic_connectivity(nx.Graph(ieee14)), rel=1e-6)
    table = gg.connectivity_after_removal(ieee14)
    assert len(table) == 20
    worst = table.iloc[0]
    assert (worst['from_bus'], worst['to_bus']) == (7, 8) and worst['lambda2'] == pytest.approx(0, abs=1e-9)
    assert (table['lambda2'].iloc[1:] > 0).all() and (table['lambda2'] <= base + 1e-12).all()
    assert table['drop'].iloc[0] == pytest.approx(base)


def test_two_hop_neighbourhood_of_bus_5(ieee14):
    hops = gg.k_hop_neighbourhood(ieee14, 5, 2)
    assert hops == {1: 1, 2: 1, 3: 2, 4: 1, 5: 0, 6: 1, 7: 2, 9: 2, 11: 2, 12: 2, 13: 2}
    assert max(hops.values()) == 2 and 8 not in hops


def test_figures_are_built(ieee14):
    bridges, points = gg.bridges_and_articulation_points(ieee14)
    fig = gg.draw_grid(ieee14, bridges, points)
    assert len(fig.axes) == 1
    table = gg.connectivity_after_removal(ieee14)
    fig2 = gg.plot_connectivity(table, gg.algebraic_connectivity(ieee14))
    assert len(fig2.axes[0].patches) == 20
    matplotlib.pyplot.close('all')


def test_laplacian_rows_sum_to_zero(ieee14):
    lap = gg.laplacian(ieee14)
    np.testing.assert_allclose(lap.sum(axis=1), 0)
    assert lap[6, 7] == -1 and lap[7, 7] == 1  # bus 8 has degree 1
