"""A genetic algorithm for "switch off at most k lines" on IEEE 30, checked against brute force (Chapter 24).

The genotype and the mutation are those of ``genome_mutation.py``: k sorted slots padded with the sentinel n.
Crossover samples k unique indices from the union of two parents. Selection is a tournament, the best genomes
survive unchanged (elitism), and the run stops after a fixed number of fitness evaluations. Islanding candidates
either get fitness -inf (penalty) or are repaired by dropping lines until the grid is connected.
"""

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from genome_mutation import mutate, normalize
from line_switching import Case, fitness, is_connected, load_case30

K_MAX = 3  # at most three switched lines
BUDGET = 1152  # 10 % of the 11 522 sets with at most three lines


class Evaluator:
    """Counts fitness evaluations and records the best value seen after each one, with the elapsed time."""

    def __init__(self, case: Case):
        """Wrap the fitness of ``case``."""
        self.case = case
        self.n_evaluations = 0
        self.n_islanding = 0
        self.best = -np.inf
        self.best_genome: np.ndarray | None = None
        self.history: list[tuple[int, float, float]] = []  # (evaluations, best so far, seconds)
        self.start = time.perf_counter()

    def __call__(self, genome: np.ndarray) -> float:
        """Fitness of the set ``genome[genome < n]``, counted once per call."""
        value = fitness(self.case, genome[genome < self.case.n_branch])
        self.n_evaluations += 1
        if value == -np.inf:
            self.n_islanding += 1
        if value > self.best:
            self.best, self.best_genome = value, genome.copy()
        self.history.append((self.n_evaluations, self.best, time.perf_counter() - self.start))
        return value


def random_genome(k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """A normalized genome whose slots are random lines or the sentinel, each with equal weight."""
    return normalize(rng.integers(n + 1, size=k), n)


def crossover(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample up to k unique indices from the union of the two parents' lines."""
    union = np.union1d(a[a < n], b[b < n])
    child = rng.choice(union, size=min(len(a), len(union)), replace=False) if len(union) else np.array([], dtype=int)
    return normalize(np.concatenate([child, np.full(len(a) - len(child), n)]).astype(int), n)


def tournament(values: np.ndarray, size: int, rng: np.random.Generator) -> int:
    """Index of the best of ``size`` genomes drawn at random."""
    contenders = rng.choice(len(values), size=size, replace=False)
    return int(contenders[np.argmax(values[contenders])])


def repair(case: Case, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Drop lines from the genome in random order until the grid is connected (no fitness evaluation needed)."""
    n = case.n_branch
    lines = genome[genome < n]
    order = rng.permutation(len(lines))
    while len(lines) and not is_connected(case, lines):
        lines = np.delete(lines, order[0] % len(lines))
        order = order[1:]
    return normalize(np.concatenate([lines, np.full(len(genome) - len(lines), n)]).astype(int), n)


@dataclass
class Run:
    """One seeded run: the evaluation counter with its history, and the configuration."""

    seed: int
    evaluator: Evaluator
    repaired: bool
    config: dict = field(default_factory=dict)

    @property
    def best(self) -> float:
        """Best fitness found."""
        return self.evaluator.best

    @property
    def seconds(self) -> float:
        """Wall-clock time of the run."""
        return self.evaluator.history[-1][2]

    def evaluations_to_reach(self, target: float, tol: float = 1e-9) -> int | None:
        """Number of evaluations after which the best value first reached ``target``; None if never."""
        for n_eval, best, _ in self.evaluator.history:
            if best >= target - tol:
                return n_eval
        return None

    def curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Best-so-far fitness against evaluations."""
        h = np.array(self.evaluator.history)
        return h[:, 0], h[:, 1]


def genetic_algorithm(  # noqa: PLR0913
    case: Case,
    seed: int,
    k: int = K_MAX,
    budget: int = BUDGET,
    pop_size: int = 32,
    n_elite: int = 2,
    tournament_size: int = 3,
    p_crossover: float = 0.7,
    repaired: bool = False,
) -> Run:
    """Run the GA until ``budget`` fitness evaluations have been spent; the run keeps the whole history."""
    rng = np.random.default_rng(seed)
    n = case.n_branch
    evaluator = Evaluator(case)
    fix = (lambda g: repair(case, g, rng)) if repaired else (lambda g: g)
    population = [fix(random_genome(k, n, rng)) for _ in range(pop_size)]
    values = np.array([evaluator(g) for g in population])
    while evaluator.n_evaluations < budget:
        elite = [population[i] for i in np.argsort(values)[::-1][:n_elite]]
        children = []
        while len(children) < pop_size - n_elite and evaluator.n_evaluations + len(children) < budget:
            parent = population[tournament(values, tournament_size, rng)]
            if rng.random() < p_crossover:
                parent = crossover(parent, population[tournament(values, tournament_size, rng)], n, rng)
            children.append(fix(mutate(parent, n, rng)))
        child_values = np.array([evaluator(g) for g in children])
        population = elite + children
        values = np.concatenate([[values[i] for i in np.argsort(values)[::-1][:n_elite]], child_values])
    config = {'k': k, 'budget': budget, 'pop_size': pop_size, 'n_elite': n_elite, 'p_crossover': p_crossover}
    return Run(seed, evaluator, repaired, config)


def seed_table(runs: list[Run], target: float, case: Case) -> pd.DataFrame:
    """One row per run: best overload energy, evaluations to first reach ``target``, time and wasted evaluations."""
    rows = []
    for run in runs:
        reached = run.evaluations_to_reach(target)
        genome = run.evaluator.best_genome
        rows.append(
            {
                'seed': run.seed,
                'best E [MW]': round(-run.best, 3),
                'lines': ', '.join(case.names[i] for i in genome[genome < case.n_branch]),
                'evals to optimum': 'not reached' if reached is None else reached,
                'seconds': round(run.seconds, 2),
                'islanding evals': run.evaluator.n_islanding,
            }
        )
    return pd.DataFrame(rows).set_index('seed')


def plot_convergence(runs: list[Run], target: float | None = None, ax=None) -> None:
    """Best-so-far overload energy against evaluations, one line per seed."""
    ax = ax or plt.gca()
    for run in runs:
        evals, best = run.curve()
        ax.plot(evals, -best, lw=1, label=f'seed {run.seed}')
    if target is not None:
        ax.axhline(-target, color='k', ls='--', lw=1, label='brute-force optimum')
    ax.set_xlabel('fitness evaluations')
    ax.set_ylabel('best N-1 overload energy [MW]')
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)


def main() -> None:
    """Lab steps 3 and 4 and Exercise 4: ten seeds against the |D| <= 3 optimum, penalty against repair."""
    from line_switching import brute_force

    case = load_case30()
    bf = brute_force(case, K_MAX)
    best, target = bf.best
    print(f'brute force: E = {-target:.3f} MW, budget for the GA {BUDGET} evaluations')
    runs = [genetic_algorithm(case, seed) for seed in range(10)]
    table = seed_table(runs, target, case)
    print(table.to_string())
    reached = table['evals to optimum'] != 'not reached'
    print(f'{int(reached.sum())} of 10 seeds reached the optimum; mean best E {table["best E [MW]"].mean():.3f} MW')
    repaired = [genetic_algorithm(case, seed, repaired=True) for seed in range(10)]
    table_r = seed_table(repaired, target, case)
    reached_r = table_r['evals to optimum'] != 'not reached'
    print(table_r.to_string())
    print(
        f'wasted (islanding) evaluations per run: penalty {table["islanding evals"].mean():.1f}, '
        f'repair {table_r["islanding evals"].mean():.1f}; mean best E: penalty {table["best E [MW]"].mean():.3f}, '
        f'repair {table_r["best E [MW]"].mean():.3f}; '
        f'optimum reached: penalty {int(reached.sum())}, repair {int(reached_r.sum())}'
    )


if __name__ == '__main__':
    main()
