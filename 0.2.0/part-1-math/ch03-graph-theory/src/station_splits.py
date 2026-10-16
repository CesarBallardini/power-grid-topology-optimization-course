"""Coupler states of a small station and the busbar bipartitions they produce (Chapter 3, lab step 5, Exercise 4).

A station is a set of busbars joined by couplers. Opening couplers splits the station graph; ToOp keeps only the
coupler states that leave exactly two electrical busbars (``make_separation_set`` in
``dc_solver/preprocess/preprocess_switching.py``). This module enumerates those states by brute force.
"""

import itertools

import networkx as nx

RING = [(0, 1), (1, 2), (2, 3), (3, 0)]
RING_WITH_DIAGONAL = [*RING, (0, 2)]
N_BUSBARS = 4

Bipartition = tuple[frozenset[int], frozenset[int]]


def canonical(components: list[set[int]]) -> Bipartition:
    """The two components as a pair ordered so that the one holding the smallest busbar comes first."""
    a, b = (frozenset(c) for c in components)
    return (a, b) if min(a) < min(b) else (b, a)


def two_busbar_states(couplers: list[tuple[int, int]], n_busbars: int) -> list[tuple[tuple[int, ...], Bipartition]]:
    """Every open/closed coupler state whose closed couplers leave exactly two connected components.

    A state is a tuple with one 1 (closed) or 0 (open) per coupler. The all-closed state is skipped, since it is
    the unsplit station, and so are states with one or with more than two components.
    """
    out = []
    for state in itertools.product((0, 1), repeat=len(couplers)):
        if all(state):
            continue
        graph = nx.Graph()
        graph.add_nodes_from(range(n_busbars))
        graph.add_edges_from(c for c, closed in zip(couplers, state, strict=True) if closed)
        components = list(nx.connected_components(graph))
        if len(components) == 2:
            out.append((state, canonical(components)))
    return out


def count_states_and_bipartitions(couplers: list[tuple[int, int]], n_busbars: int) -> tuple[int, int]:
    """Number of two-busbar coupler states and of distinct bipartitions they realize."""
    states = two_busbar_states(couplers, n_busbars)
    return len(states), len({bipartition for _, bipartition in states})


def all_bipartitions(n_busbars: int) -> list[Bipartition]:
    """The 2^(n−1) − 1 non-trivial bipartitions of the busbars (busbar 0 always on side A) (Exercise 4)."""
    others = range(1, n_busbars)
    out = []
    for size in range(n_busbars - 1):
        for side_a in itertools.combinations(others, size):
            a = frozenset((0, *side_a))
            b = frozenset(others) - a
            if b:
                out.append((a, b))
    return out


def unreachable_bipartitions(couplers: list[tuple[int, int]], n_busbars: int) -> list[Bipartition]:
    """Non-trivial bipartitions that no coupler state realizes, because one side is not connected (Exercise 4)."""
    reachable = {bipartition for _, bipartition in two_busbar_states(couplers, n_busbars)}
    return [b for b in all_bipartitions(n_busbars) if b not in reachable]


def describe(bipartition: Bipartition) -> str:
    """A bipartition as ``{0, 1} | {2, 3}``."""
    a, b = bipartition
    return f'{set(sorted(a))} | {set(sorted(b))}'


def main() -> None:
    """Print the counts of lab step 5 and the unreachable bipartition of Exercise 4."""
    for name, couplers in (('ring', RING), ('ring + diagonal (0, 2)', RING_WITH_DIAGONAL)):
        n_states, n_bipartitions = count_states_and_bipartitions(couplers, N_BUSBARS)
        print(f'{name}: {n_states} two-busbar states, {n_bipartitions} distinct bipartitions')
        for state, bipartition in two_busbar_states(couplers, N_BUSBARS):
            print(f'  closed={state}  {describe(bipartition)}')
        print('  unreachable:', [describe(b) for b in unreachable_bipartitions(couplers, N_BUSBARS)])
    n_assignments, n_nontrivial = 2 ** (N_BUSBARS - 1), len(all_bipartitions(N_BUSBARS))
    print(f'{N_BUSBARS} busbars: {n_assignments} assignments up to swapping, {n_nontrivial} non-trivial')


if __name__ == '__main__':
    main()
