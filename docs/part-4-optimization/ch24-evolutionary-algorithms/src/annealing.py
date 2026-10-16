"""Hill climbing and simulated annealing on the k-subset neighborhood (Chapter 24, Exercises 1 and 2).

The neighborhood of a genome is "change one slot": overwrite one of the k slots with another line or with the
sentinel, then normalize. Hill climbing only accepts improvements and stops at a local optimum. Simulated
annealing accepts a worse neighbor with probability exp(delta / T), T following a geometric cooling schedule.
"""

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
from genome_mutation import mutate, normalize
from line_switching import Case, load_case30
from switching_ga import BUDGET, K_MAX, Evaluator, Run, random_genome

if TYPE_CHECKING:
    from collections.abc import Callable


def neighbors(genome: np.ndarray, n: int) -> set[tuple[int, ...]]:
    """Every distinct genome reached by overwriting one slot with any line or the sentinel."""
    out = set()
    for slot in range(len(genome)):
        for value in range(n + 1):
            g = genome.copy()
            g[slot] = value
            out.add(tuple(normalize(g, n)))
    out.discard(tuple(genome))
    return out


def count_neighbors(n: int, k: int, j: int) -> int:
    """Distinct neighbors of a genome with j lines in k slots: j (n - j) swaps, j removals, n - j additions (j < k)."""
    return j * (n - j) + j + (n - j if j < k else 0)


# --8<-- [start:toy]
TOY_N, TOY_K = 4, 2  # four lines, at most two switched off


def toy_fitness(genome: np.ndarray) -> float:
    """A deceptive two-line landscape: {0, 1} is a local optimum, {2, 3} the global one, every mixed pair is poor."""
    lines = frozenset(genome[genome < TOY_N].tolist())
    table = {frozenset({0, 1}): 10.0, frozenset({2, 3}): 20.0}
    return table.get(lines, 5.0 if len(lines) == 1 else 0.0)


def hill_climb(objective: Callable[[np.ndarray], float], start: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Steepest ascent over the whole neighborhood; returns the local optimum and the number of evaluations."""
    current, value, evaluations = start, objective(start), 1
    while True:
        best = None
        for g in neighbors(current, n):
            candidate = np.array(g)
            v = objective(candidate)
            evaluations += 1
            if v > value and (best is None or v > best[1]):
                best = (candidate, v)
        if best is None:
            return current, evaluations
        current, value = best


# --8<-- [end:toy]


def simulated_annealing(  # noqa: PLR0913
    objective: Callable[[np.ndarray], float],
    start: np.ndarray,
    n: int,
    rng: np.random.Generator,
    budget: int,
    t0: float,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Geometric cooling T_i = t0 alpha^i, one mutated neighbor per step; returns the best genome and value."""
    current, value = start, objective(start)
    best, best_value = current, value
    temperature = t0
    for _ in range(budget - 1):
        candidate = mutate(current, n, rng)
        v = objective(candidate)
        delta = v - value
        if delta >= 0 or (math.isfinite(delta) and rng.random() < math.exp(delta / temperature)):
            current, value = candidate, v
            if v > best_value:
                best, best_value = candidate, v
        temperature *= alpha
    return best, best_value


def toy_runs(seed: int = 0, n_runs: int = 20) -> tuple[float, int]:
    """Hill climbing from {0, 1} stays there; the fraction of seeded annealing runs that reach {2, 3}."""
    start = normalize(np.array([0, 1]), TOY_N)
    stuck, _ = hill_climb(toy_fitness, start, TOY_N)
    rng = np.random.default_rng(seed)
    escaped = sum(
        simulated_annealing(toy_fitness, start, TOY_N, rng, budget=200, t0=10.0, alpha=0.97)[1] == 20.0
        for _ in range(n_runs)
    )
    return toy_fitness(stuck), escaped


def anneal_case(case: Case, seed: int, k: int = K_MAX, budget: int = BUDGET, t0: float = 20.0, alpha=0.995) -> Run:
    """Simulated annealing on the line-switching fitness with the GA's evaluation budget, as a ``Run``."""
    rng = np.random.default_rng(seed)
    evaluator = Evaluator(case)
    simulated_annealing(evaluator, random_genome(k, case.n_branch, rng), case.n_branch, rng, budget, t0, alpha)
    return Run(seed, evaluator, False, {'k': k, 'budget': budget, 't0': t0, 'alpha': alpha})


def main() -> None:
    """Exercise 1 (neighbor count and toy landscape) and Exercise 2 (annealing against the GA on IEEE 30)."""
    from line_switching import brute_force
    from switching_ga import genetic_algorithm, seed_table

    n, k = 41, 3
    full = normalize(np.array([0, 1, 2]), n)
    print(
        f'neighbors of a full genome (n={n}, k={k}): {len(neighbors(full, n))} = {count_neighbors(n, k, k)}; '
        f'raw slot overwrites k (n + 1) = {k * (n + 1)}'
    )
    two = normalize(np.array([0, 1, n]), n)
    print(f'neighbors with two lines: {len(neighbors(two, n))} = {count_neighbors(n, k, 2)}')
    stuck_value, escaped = toy_runs()
    print(
        f'toy: hill climbing from {{0, 1}} ends with fitness {stuck_value}; annealing reaches 20 in {escaped}/20 runs'
    )
    for pair in itertools.combinations(range(TOY_N), 2):
        print(f'  toy fitness {set(pair)}: {toy_fitness(np.array(pair))}')
    case = load_case30()
    _, target = brute_force(case, k).best
    sa = [anneal_case(case, seed) for seed in range(10)]
    ga = [genetic_algorithm(case, seed) for seed in range(10)]
    table_sa, table_ga = seed_table(sa, target, case), seed_table(ga, target, case)
    print(table_sa.to_string())
    for name, table in (('SA', table_sa), ('GA', table_ga)):
        reached = int((table['evals to optimum'] != 'not reached').sum())
        print(
            f'{name}: mean best E {table["best E [MW]"].mean():.3f} MW, median {table["best E [MW]"].median():.3f}, '
            f'optimum reached in {reached}/10 runs'
        )


if __name__ == '__main__':
    main()
