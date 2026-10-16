import grid_graph as gg
import networkx as nx
import pandapower.networks as pn
import pytest
from ieee118 import edges_from_pandapower


@pytest.fixture(scope='module')
def ieee118():
    edges = edges_from_pandapower(pn.case118())
    return edges, gg.build_graph(edges)


def test_118_bus_edge_list(ieee118):
    edges, graph = ieee118
    assert len(edges) == 186 and graph.number_of_nodes() == 118
    assert edges.min().min() == 1 and edges.max().max() == 118
    assert nx.is_connected(graph)


def test_nine_bridges_and_their_effective_resistance(ieee118):
    _, graph = ieee118
    bridges, points = gg.bridges_and_articulation_points(graph)
    assert len(bridges) == 9 and (8, 9) in bridges and (9, 10) in bridges
    assert len(points) == 9 and 9 in points
    reff = gg.effective_resistances(graph).set_index(['from_bus', 'to_bus'])['r_eff']
    assert set(reff[reff > 1 - 1e-9].index) == set(bridges)
    assert gg.count_zero_eigenvalues(graph) == 1
