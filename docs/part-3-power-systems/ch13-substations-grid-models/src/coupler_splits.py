"""Which coupler states split a four-busbar substation into exactly two electrical nodes (Chapter 13)."""

import itertools

import networkx as nx

COUPLERS = [(0, 1), (1, 2), (2, 3)]  # busbar pairs joined by a coupler


def two_node_configurations(couplers: list[tuple[int, int]], n_busbars: int) -> list[tuple[tuple[bool, ...], list]]:
    """Enumerate the open/closed states of the couplers that leave exactly two connected groups of busbars.

    Returns ``(is_open, parts)`` pairs: the state of each coupler (``True`` = open) and the two groups.
    """
    found = []
    for is_open in itertools.product([True, False], repeat=len(couplers)):
        g = nx.Graph()
        g.add_nodes_from(range(n_busbars))
        g.add_edges_from(c for c, o in zip(couplers, is_open, strict=True) if not o)
        parts = list(nx.connected_components(g))
        if any(is_open) and len(parts) == 2:
            found.append((is_open, parts))
    return found


def main() -> None:
    """Print the splits of a four-busbar substation whose couplers form a chain."""
    for is_open, parts in two_node_configurations(COUPLERS, 4):
        print(is_open, parts)


if __name__ == '__main__':
    main()
