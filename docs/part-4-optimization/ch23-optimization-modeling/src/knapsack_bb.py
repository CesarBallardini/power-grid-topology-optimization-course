"""Branch and bound by hand on a three-item 0-1 knapsack, and the PuLP check (Chapter 23, Exercise 3).

max 5 a + 4 b + 3 c subject to 2 a + 3 b + c <= 5, a, b, c in {0, 1}. The bound at a node is the LP relaxation of the
remaining knapsack: items sorted by value per unit weight, taken whole until one no longer fits, which is taken
fractionally. Branching fixes that fractional item to 0 and to 1.
"""

from dataclasses import dataclass, field

import pulp

VALUES = {'a': 5.0, 'b': 4.0, 'c': 3.0}
WEIGHTS = {'a': 2.0, 'b': 3.0, 'c': 1.0}
CAPACITY = 5.0


@dataclass
class Node:
    """A node of the tree: fixed variables, its LP bound, the fractional item and how it was closed."""

    fixed: dict[str, int]
    bound: float = 0.0
    fractional: str | None = None
    status: str = ''
    children: list[Node] = field(default_factory=list)


def lp_bound(fixed: dict[str, int]) -> tuple[float, str | None, bool, dict[str, int]]:
    """Greedy fractional knapsack over the free items.

    Returns the bound, the fractional item (``None`` when the relaxation is integer), whether the fixed items fit
    and the integer assignment of the free items the greedy pass made.
    """
    capacity = CAPACITY - sum(WEIGHTS[k] for k, v in fixed.items() if v == 1)
    if capacity < 0:
        return float('-inf'), None, False, {}
    value = sum(VALUES[k] for k, v in fixed.items() if v == 1)
    taken = {}
    free = sorted((k for k in VALUES if k not in fixed), key=lambda k: VALUES[k] / WEIGHTS[k], reverse=True)
    for k in free:
        if WEIGHTS[k] <= capacity:
            capacity -= WEIGHTS[k]
            value += VALUES[k]
            taken[k] = 1
        elif capacity > 1e-12:
            return value + VALUES[k] * capacity / WEIGHTS[k], k, True, taken
        else:
            taken[k] = 0  # no room left: the item is out, no fraction needed
    return value, None, True, taken


def branch_and_bound() -> tuple[Node, float, dict[str, int]]:
    """Depth-first branch and bound (branch 0 first); returns the tree, the incumbent value and solution."""
    best = {'value': float('-inf'), 'x': {}}

    def expand(node: Node) -> None:
        node.bound, node.fractional, feasible, taken = lp_bound(node.fixed)
        if not feasible:
            node.status = 'infeasible (over capacity)'
            return
        if node.bound <= best['value'] + 1e-9:
            node.status = f'pruned by bound ({node.bound:.2f} <= incumbent {best["value"]:.0f})'
            return
        if node.fractional is None:
            node.status = f'integer, new incumbent {node.bound:.0f}'
            best['value'] = node.bound
            best['x'] = {k: {**node.fixed, **taken}[k] for k in VALUES}
            return
        node.status = f'bound {node.bound:.2f}, branch on {node.fractional}'
        for value in (0, 1):
            child = Node({**node.fixed, node.fractional: value})
            node.children.append(child)
            expand(child)

    root = Node({})
    expand(root)
    return root, best['value'], best['x']


def print_tree(node: Node, depth: int = 0) -> None:
    """Indented listing of the tree."""
    fixed = ', '.join(f'{k}={v}' for k, v in node.fixed.items()) or 'root'
    print('  ' * depth + f'[{fixed}] {node.status}')
    for child in node.children:
        print_tree(child, depth + 1)


def pulp_check() -> tuple[float, dict[str, int]]:
    """The same knapsack through PuLP and HiGHS."""
    m = pulp.LpProblem('knapsack', pulp.LpMaximize)
    x = m.add_variable_dicts('x', list(VALUES), cat='Binary')
    m += pulp.lpSum(VALUES[k] * x[k] for k in VALUES)
    m += pulp.lpSum(WEIGHTS[k] * x[k] for k in VALUES) <= CAPACITY
    m.solve(pulp.HiGHS(msg=False))
    return pulp.value(m.objective), {k: int(round(x[k].value())) for k in VALUES}


def main() -> None:
    """The tree, the incumbent and the PuLP check."""
    root, value, x = branch_and_bound()
    print_tree(root)
    print(f'optimum {value:.0f} at {x}')
    print('PuLP + HiGHS:', pulp_check())


if __name__ == '__main__':
    main()
