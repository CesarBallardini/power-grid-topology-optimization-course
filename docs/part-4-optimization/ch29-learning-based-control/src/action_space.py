"""Counting busbar assignments and joint topology actions (Chapter 29, Exercise 2).

Run it to print the numbers quoted in the solution::

    python action_space.py
"""


# --8<-- [start:count]
def assignments(elements: int, busbars: int = 2) -> int:
    """Every assignment of ``elements`` to labelled ``busbars``."""
    return busbars**elements


def assignments_up_to_symmetry(elements: int) -> int:
    """Two busbars whose swap gives the same topology: half of the labelled assignments."""
    return 2 ** (elements - 1)


def toop_splits(elements: int) -> int:
    """ToOp's valid splits of a substation with ``elements`` branches: both busbars keep at least two branches.

    Choosing the branches of the second busbar with 2 <= |B| <= d - 2 gives 2^d - 2 - 2d labelled assignments, each
    split counted twice (B and its complement), so 2^(d-1) - d - 1 splits; the unsplit configuration is one more.
    """
    return 2 ** (elements - 1) - elements - 1


def unitary_topologies(elements: int) -> int:
    """Grid2Op's ``get_all_unitary_topologies_set`` for one substation, counted by enumeration.

    Every assignment up to the busbar swap in which no busbar holds exactly one element, the unsplit one included;
    equals ``toop_splits + 1`` because both rules drop the same configurations.
    """
    configurations = set()
    for code in range(2**elements):
        second = frozenset(i for i in range(elements) if code >> i & 1)
        if len(second) in (1, elements - 1):
            continue
        configurations.add(frozenset([frozenset(range(elements)) - second, second]))
    return len(configurations)


def joint(per_substation: int, substations: int = 14) -> int:
    """Joint action space of independent choices at every substation."""
    return per_substation**substations


# --8<-- [end:count]
def main() -> None:
    """Six elements, two busbars, fourteen substations."""
    d = 6
    print(f'{d} elements on 2 labelled busbars: {assignments(d)} assignments', end='; ')
    print(f'up to the busbar swap: {assignments_up_to_symmetry(d)}')
    print(f'ToOp valid splits (both busbars with >= 2 branches): {toop_splits(d)}', end=', ')
    print(f'with the unsplit one: {toop_splits(d) + 1}')
    print(f'Grid2Op unitary topologies of the substation, by enumeration: {unitary_topologies(d)}')
    for name, n in (
        ('labelled', assignments(d)),
        ('up to symmetry', assignments_up_to_symmetry(d)),
        ('ToOp', toop_splits(d) + 1),
    ):
        print(f'  14 substations, {name}: {joint(n):.3e} = {joint(n)}')
    print(f'  14 substations, one changed at a time: 14 x {toop_splits(d)} = {14 * toop_splits(d)} unitary actions')


if __name__ == '__main__':
    main()
