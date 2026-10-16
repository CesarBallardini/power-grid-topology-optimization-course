"""Tarjan's depth-first search for bridges and articulation points, with its discovery times and low-links.

Chapter 3, Exercise 1. The example graph is two triangles joined by a single edge. The DFS visits neighbours in
increasing order, so the table it prints is the one a student gets by hand with the same convention.
"""

import networkx as nx

TWO_TRIANGLES = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)]


def tarjan(edges: list[tuple[int, int]], root: int = 0) -> dict:
    """Discovery times, low-links, bridges and articulation points of a connected undirected graph.

    ``low[v]`` is the smallest discovery time reachable from the DFS subtree of ``v`` through at most one back
    edge. Tree edge (u, v) is a bridge when ``low[v] > disc[u]``; ``u`` is an articulation point when it is the
    root with two or more DFS children, or a non-root with a child ``v`` such that ``low[v] >= disc[u]``.
    """
    graph = nx.Graph(edges)
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {root: None}
    children = dict.fromkeys(graph.nodes, 0)
    bridges: list[tuple[int, int]] = []
    points: set[int] = set()
    clock = [0]

    def visit(u: int) -> None:
        disc[u] = low[u] = clock[0]
        clock[0] += 1
        for v in sorted(graph.neighbors(u)):
            if v not in disc:
                parent[v] = u
                children[u] += 1
                visit(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
                if parent[u] is not None and low[v] >= disc[u]:
                    points.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    visit(root)
    if children[root] >= 2:
        points.add(root)
    return {'disc': disc, 'low': low, 'bridges': bridges, 'articulation_points': sorted(points)}


def main() -> None:
    """Print the DFS table of the two-triangle graph and the bridges and cut vertices it reveals."""
    result = tarjan(TWO_TRIANGLES)
    print('vertex  disc  low')
    for v in sorted(result['disc']):
        print(f'{v:6d}  {result["disc"][v]:4d}  {result["low"][v]:3d}')
    print('bridges:', result['bridges'])
    print('articulation points:', result['articulation_points'])


if __name__ == '__main__':
    main()
