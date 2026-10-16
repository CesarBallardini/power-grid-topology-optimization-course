import csv
from pathlib import Path

import networkx as nx


def test_ieee14_edge_list_is_the_connected_14_bus_graph():
    with (Path(__file__).parent / 'ieee14_edges.csv').open(newline='') as f:
        edges = [(int(r['from_bus']), int(r['to_bus'])) for r in csv.DictReader(f)]
    g = nx.Graph(edges)
    assert len(edges) == 20 and g.number_of_nodes() == 14
    assert nx.is_connected(g)
    assert list(nx.bridges(g)) == [(7, 8)]  # bus 8 hangs off bus 7
