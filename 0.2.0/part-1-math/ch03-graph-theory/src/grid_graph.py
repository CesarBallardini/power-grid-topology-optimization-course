"""Bridges, islands and Laplacian measures of a grid read from an edge list (Chapter 3, Lab and Exercises 3 and 6).

The grid is a plain CSV with the columns ``from_bus`` and ``to_bus``; the IEEE 14-bus branch list ships next to
this module. Every function takes a NetworkX graph, so the same code runs on the 118-bus list of lab step 6.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EDGES_CSV = HERE / 'ieee14_edges.csv'


def load_edges(path: Path = EDGES_CSV) -> pd.DataFrame:
    """Branch list ``from_bus, to_bus`` of a grid, as a DataFrame (lab step 1)."""
    return pd.read_csv(path)


def build_graph(edges: pd.DataFrame) -> nx.MultiGraph:
    """Multigraph of the branch list; parallel circuits become parallel edges (lab step 2)."""
    return nx.from_pandas_edgelist(edges, 'from_bus', 'to_bus', create_using=nx.MultiGraph)


def bridges_and_articulation_points(graph: nx.Graph) -> tuple[list[tuple[int, int]], list[int]]:
    """Branches whose outage islands the grid and buses whose loss islands it (lab step 2)."""
    return sorted(nx.bridges(graph)), sorted(nx.articulation_points(graph))


def laplacian(graph: nx.Graph) -> np.ndarray:
    """Dense Laplacian L = D − W with unit branch weights; parallel edges add up."""
    return nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes)).toarray().astype(float)


def count_zero_eigenvalues(graph: nx.Graph, tol: float = 1e-9) -> int:
    """Number of zero eigenvalues of the Laplacian, which equals the number of islands (lab step 3)."""
    return int(np.sum(np.abs(np.linalg.eigvalsh(laplacian(graph))) < tol))


def effective_resistances(graph: nx.Graph) -> pd.DataFrame:
    """Effective resistance across every branch with unit resistances (lab step 3).

    R_ij = (e_i − e_j)ᵀ L⁺ (e_i − e_j), with L⁺ the pseudo-inverse of the Laplacian. A bridge has R = 1, its own
    resistance, because there is no parallel path; every other branch has R < 1.
    """
    nodes = sorted(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    lplus = np.linalg.pinv(laplacian(graph))
    rows = []
    for u, v in sorted({tuple(sorted(e)) for e in graph.edges()}):
        e = np.zeros(len(nodes))
        e[index[u]], e[index[v]] = 1, -1
        rows.append({'from_bus': u, 'to_bus': v, 'r_eff': float(e @ lplus @ e)})
    return pd.DataFrame(rows)


def algebraic_connectivity(graph: nx.Graph) -> float:
    """Second-smallest Laplacian eigenvalue; zero when the graph is disconnected (Exercise 3)."""
    return float(np.linalg.eigvalsh(laplacian(graph))[1])


def connectivity_after_removal(graph: nx.Graph) -> pd.DataFrame:
    """Algebraic connectivity of the grid after removing each branch in turn (Exercise 3)."""
    base = algebraic_connectivity(graph)
    rows = []
    for u, v, key in sorted(graph.edges(keys=True)):
        h = graph.copy()
        h.remove_edge(u, v, key=key)
        rows.append({'from_bus': min(u, v), 'to_bus': max(u, v), 'lambda2': algebraic_connectivity(h)})
    out = pd.DataFrame(rows)
    out['drop'] = base - out['lambda2']
    return out.sort_values('lambda2', ignore_index=True)


def k_hop_neighbourhood(graph: nx.Graph, source: int, hops: int) -> dict[int, int]:
    """Buses within ``hops`` of ``source`` with unit weights, with their distance (Exercise 6)."""
    lengths = nx.single_source_dijkstra_path_length(graph, source, cutoff=hops)
    return dict(sorted(lengths.items()))


def draw_grid(graph: nx.Graph, bridges: list[tuple[int, int]], points: list[int]) -> matplotlib.figure.Figure:
    """Draw the grid with bridges in red and articulation points as larger red nodes (lab step 2)."""
    pos = nx.kamada_kawai_layout(graph)
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ['tab:red' if n in points else 'tab:blue' for n in graph.nodes]
    sizes = [500 if n in points else 250 for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_color=colours, node_size=sizes, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    bridge_set = {tuple(sorted(b)) for b in bridges}
    plain = [(u, v) for u, v in graph.edges() if tuple(sorted((u, v))) not in bridge_set]
    nx.draw_networkx_edges(graph, pos, edgelist=plain, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=list(bridge_set), edge_color='tab:red', width=3, ax=ax)
    ax.set_axis_off()
    return fig


def plot_connectivity(table: pd.DataFrame, base: float) -> matplotlib.figure.Figure:
    """Bar chart of the algebraic connectivity after each single-branch removal (Exercise 3)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f'{u}-{v}' for u, v in zip(table['from_bus'], table['to_bus'], strict=True)]
    ax.bar(labels, table['lambda2'])
    ax.axhline(base, color='k', linestyle='--', label=f'intact grid: {base:.3f}')
    ax.set_ylabel(r'$\lambda_2$ after removal')
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    return fig


def main() -> None:
    """Print every number quoted in the Lab (steps 1–3) and Exercises 3 and 6 of the solutions."""
    edges = load_edges()
    graph = build_graph(edges)
    print(f'{len(edges)} branches, {graph.number_of_nodes()} buses')
    bridges, points = bridges_and_articulation_points(graph)
    print('bridges:', bridges, ' articulation points:', points)
    graph.add_edge(7, 8)
    print('bridges with a parallel circuit on (7, 8):', list(nx.bridges(graph)))
    graph.remove_edge(7, 8)
    print('zero eigenvalues, intact grid:', count_zero_eigenvalues(graph))
    cut = graph.copy()
    cut.remove_edge(7, 8)
    print('zero eigenvalues without (7, 8):', count_zero_eigenvalues(cut))
    reff = effective_resistances(graph)
    print(reff.sort_values('r_eff', ascending=False).round(4).to_string(index=False))
    base = algebraic_connectivity(graph)
    print(f'algebraic connectivity: {base:.4f}')
    table = connectivity_after_removal(graph)
    print(table.round(4).to_string(index=False))
    print('within 2 hops of bus 5:', k_hop_neighbourhood(graph, 5, 2))


if __name__ == '__main__':
    main()
