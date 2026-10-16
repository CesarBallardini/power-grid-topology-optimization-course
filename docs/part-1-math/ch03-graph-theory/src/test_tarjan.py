import networkx as nx
import pytest
from tarjan import TWO_TRIANGLES, tarjan


def test_two_triangles_table():
    result = tarjan(TWO_TRIANGLES)
    assert result['disc'] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert result['low'] == {0: 0, 1: 0, 2: 0, 3: 3, 4: 3, 5: 3}
    assert result['bridges'] == [(2, 3)]
    assert result['articulation_points'] == [2, 3]


def test_agrees_with_networkx_on_two_triangles():
    g = nx.Graph(TWO_TRIANGLES)
    assert sorted(nx.bridges(g)) == [(2, 3)]
    assert sorted(nx.articulation_points(g)) == [2, 3]


@pytest.mark.parametrize('seed', range(5))
def test_agrees_with_networkx_on_random_connected_graphs(seed):
    g = nx.connected_watts_strogatz_graph(12, 3, 0.4, seed=seed)
    result = tarjan(list(g.edges()))
    assert {frozenset(b) for b in result['bridges']} == {frozenset(b) for b in nx.bridges(g)}
    assert result['articulation_points'] == sorted(nx.articulation_points(g))


def test_root_with_two_children_is_an_articulation_point():
    result = tarjan([(0, 1), (0, 2)])
    assert result['articulation_points'] == [0] and len(result['bridges']) == 2
