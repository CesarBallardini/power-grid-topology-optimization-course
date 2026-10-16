"""The O(nW) dynamic program for 0-1 knapsack and why it is pseudo-polynomial (Chapter 23, Exercise 5)."""

import time

import numpy as np


def knapsack_dp(values: list[int], weights: list[int], capacity: int) -> tuple[int, list[int], int]:
    """Best value, the chosen items and the number of table cells filled (n times W + 1)."""
    n = len(values)
    table = np.zeros(
        (n + 1, capacity + 1), dtype=np.int64
    )  # table[i, w]: best value with the first i items, weight <= w
    for i in range(1, n + 1):
        v, wt = values[i - 1], weights[i - 1]
        table[i] = table[i - 1]
        if wt <= capacity:
            table[i, wt:] = np.maximum(table[i - 1, wt:], table[i - 1, : capacity + 1 - wt] + v)
    chosen = []
    w = capacity
    for i in range(n, 0, -1):
        if table[i, w] != table[i - 1, w]:
            chosen.append(i - 1)
            w -= weights[i - 1]
    return int(table[n, capacity]), sorted(chosen), n * (capacity + 1)


def verify_certificate(values: list[int], weights: list[int], capacity: int, chosen: list[int], target: int) -> bool:
    """The polynomial check of the decision version: the certificate is the chosen set, checked in O(n)."""
    return sum(weights[i] for i in chosen) <= capacity and sum(values[i] for i in chosen) >= target


def timing(
    n: int = 50, capacities: tuple[int, ...] = (1_000, 10_000, 100_000), seed: int = 5
) -> list[tuple[int, float]]:
    """Seconds of the dynamic program for one instance family as W grows ten times per step."""
    rng = np.random.default_rng(seed)
    out = []
    for capacity in capacities:
        values = [int(v) for v in rng.integers(1, 100, n)]
        weights = [int(w) for w in rng.integers(1, capacity // 2, n)]
        start = time.perf_counter()
        knapsack_dp(values, weights, capacity)
        out.append((capacity, time.perf_counter() - start))
    return out


def main() -> None:
    """Exercise 3's instance through the table, then the growth of the running time with W."""
    value, chosen, cells = knapsack_dp([5, 4, 3], [2, 3, 1], 5)
    print(f'max 5a + 4b + 3c with 2a + 3b + c <= 5: value {value}, items {chosen}, {cells} table cells')
    print('certificate check:', verify_certificate([5, 4, 3], [2, 3, 1], 5, chosen, 9))
    for capacity, seconds in timing():
        print(f'n = 50, W = {capacity:>7,}: {seconds * 1000:8.1f} ms  (W has {capacity.bit_length()} bits)')


if __name__ == '__main__':
    main()
