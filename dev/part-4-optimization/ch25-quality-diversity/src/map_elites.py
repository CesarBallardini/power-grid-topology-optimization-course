"""MAP-Elites in NumPy for the line-switching problem of Chapter 24, and a deceptive 1-D landscape (Chapter 25).

The archive is a grid with two integer descriptors, the number of switched lines (0 to 3) and the maximum N-0
loading after switching binned into 5 cells; each cell keeps the best genome that landed in it. Coverage is the
fraction of filled cells and the QD-score the sum over filled cells of (fitness - fitness of the unswitched grid),
so that a cell only adds to the score when it beats doing nothing.

The fitness and the genotype come from Chapter 24's ``src`` folder, put on the import path below.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ch24-evolutionary-algorithms' / 'src'))

from genome_mutation import mutate  # noqa: E402
from line_switching import Case, kept_branches, load_case30, n1_flows  # noqa: E402
from switching_ga import BUDGET, K_MAX, crossover, random_genome  # noqa: E402

LOADING_EDGES = (90.0, 100.0, 110.0, 130.0)  # N-0 loading bins: <90, 90-100, 100-110, 110-130, >=130 %
CELLS = (K_MAX + 1, len(LOADING_EDGES) + 1)


def evaluate(case: Case, genome: np.ndarray) -> tuple[float, tuple[int, int] | None]:
    """Fitness and descriptor cell (lines switched, N-0 loading bin) of a genome; the cell is None if islanded."""
    lines = genome[genome < case.n_branch]
    result = n1_flows(case, lines)
    if result is None:
        return -np.inf, None
    f0, f1, bridge = result
    keep = kept_branches(case, lines)
    over = np.maximum(0.0, np.abs(f1) - case.limit[keep][:, None])
    over[:, bridge] = 0.0
    loading = float((100.0 * np.abs(f0) / case.limit[keep]).max())
    return -float(over.sum()), (len(lines), int(np.searchsorted(LOADING_EDGES, loading, side='right')))


@dataclass
class Archive:
    """The grid archive and the history of coverage, QD-score and best fitness against evaluations."""

    cells: tuple[int, int]
    baseline: float
    fitness: np.ndarray = field(init=False)
    genomes: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    history: list[tuple[int, float, float, float]] = field(default_factory=list)
    n_evaluations: int = 0

    def __post_init__(self):
        """Start with every cell empty."""
        self.fitness = np.full(self.cells, -np.inf)

    def insert(self, genome: np.ndarray, value: float, cell: tuple[int, int] | None) -> None:
        """Count the evaluation and keep the genome if its cell is empty or held a worse one."""
        self.n_evaluations += 1
        if cell is not None and value > self.fitness[cell]:
            self.fitness[cell] = value
            self.genomes[cell] = genome.copy()
        self.history.append((self.n_evaluations, self.coverage, self.qd_score, self.best))

    @property
    def filled(self) -> np.ndarray:
        """Boolean mask of the filled cells."""
        return np.isfinite(self.fitness)

    @property
    def coverage(self) -> float:
        """Filled cells over all cells."""
        return float(self.filled.mean())

    @property
    def qd_score(self) -> float:
        """Sum over filled cells of the fitness gain over the unswitched grid (negative for cells worse than it)."""
        return float((self.fitness[self.filled] - self.baseline).sum())

    @property
    def best(self) -> float:
        """Best fitness in the archive."""
        return float(self.fitness.max()) if self.filled.any() else -np.inf

    def table(self) -> np.ndarray:
        """Overload energy per cell (rows: lines switched, columns: loading bin), NaN where empty."""
        return np.where(self.filled, -self.fitness, np.nan)


def map_elites(case: Case, seed: int, budget: int = BUDGET, batch: int = 32, p_crossover: float = 0.5) -> Archive:
    """Fill an archive with ``budget`` evaluations: random genomes first, then mutated (and crossed) elites."""
    rng = np.random.default_rng(seed)
    n = case.n_branch
    archive = Archive(CELLS, baseline=evaluate(case, np.full(K_MAX, n))[0])
    for _ in range(batch):
        g = random_genome(K_MAX, n, rng)
        archive.insert(g, *evaluate(case, g))
    while archive.n_evaluations < budget:
        elites = list(archive.genomes.values())
        parent = elites[rng.integers(len(elites))]
        if rng.random() < p_crossover:
            parent = crossover(parent, elites[rng.integers(len(elites))], n, rng)
        child = mutate(parent, n, rng)
        archive.insert(child, *evaluate(case, child))
    return archive


# --8<-- [start:deceptive]
N_POINTS = 100  # the 1-D genotype is an integer 0 <= x < 100


def deceptive_fitness(x: np.ndarray) -> np.ndarray:
    """A wide hill of height 10 at x = 20, a flat valley, and a narrow peak of height 20 at x = 85."""
    x = np.asarray(x, dtype=float)
    hill = 10.0 * np.exp(-(((x - 20.0) / 15.0) ** 2))
    peak = 20.0 * np.exp(-(((x - 85.0) / 2.0) ** 2))
    return hill + peak


STEP = 10  # a mutation moves x by up to STEP, the width of one archive cell


def step(x: int, rng: np.random.Generator, step_size: int = STEP) -> int:
    """Mutation: move by a nonzero integer in [-step_size, step_size], clipped to the domain."""
    delta = int(rng.integers(1, step_size + 1)) * int(rng.choice([-1, 1]))
    return int(np.clip(x + delta, 0, N_POINTS - 1))


def ga_1d(seed: int, budget: int = 2000, pop_size: int = 20, n_elite: int = 2, step_size: int = STEP) -> int:
    """Elitist GA with tournament selection; returns the best x found."""
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 60, size=pop_size)  # start on the left, where the hill is
    values = deceptive_fitness(population)
    evaluations = pop_size
    while evaluations < budget:
        order = np.argsort(values)[::-1]
        children = []
        for _ in range(pop_size - n_elite):
            a, b = rng.choice(pop_size, size=2, replace=False)
            parent = population[a] if values[a] >= values[b] else population[b]
            children.append(step(int(parent), rng, step_size))
        population = np.concatenate([population[order[:n_elite]], children])
        values = deceptive_fitness(population)
        evaluations += pop_size - n_elite
    return int(population[np.argmax(values)])


def map_elites_1d(seed: int, budget: int = 2000, n_cells: int = 10, step_size: int = STEP) -> int:
    """MAP-Elites with the descriptor "which tenth of the domain x lies in"; returns the best x found."""
    rng = np.random.default_rng(seed)
    archive = {}
    for x in rng.integers(0, 60, size=20):
        archive.setdefault(int(x) * n_cells // N_POINTS, int(x))
    evaluations = 20
    while evaluations < budget:
        parent = list(archive.values())[rng.integers(len(archive))]
        child = step(parent, rng, step_size)
        cell = child * n_cells // N_POINTS
        if cell not in archive or deceptive_fitness(child) > deceptive_fitness(archive[cell]):
            archive[cell] = child
        evaluations += 1
    return max(archive.values(), key=lambda x: float(deceptive_fitness(x)))


# --8<-- [end:deceptive]


def main() -> None:
    """Exercise 1 (MAP-Elites on IEEE 30, ten seeds) and Exercise 3 (the deceptive landscape, twenty seeds)."""
    case = load_case30()
    archives = [map_elites(case, seed) for seed in range(10)]
    print(f'archive {CELLS[0]} x {CELLS[1]} cells, baseline E = {-archives[0].baseline:.3f} MW')
    for a in archives[:1]:
        for n_eval in (32, 100, 300, 600, 1152):
            e, cov, qd, best = a.history[n_eval - 1]
            print(f'  seed 0 after {e:4d} evaluations: coverage {cov:.2f}, QD-score {qd:8.1f}, best E {-best:.3f}')
        print(np.round(a.table(), 1))
    best = np.array([-a.best for a in archives])
    cov = np.array([a.coverage for a in archives])
    qd = np.array([a.qd_score for a in archives])
    print(
        f'ten seeds: best E min {best.min():.3f} median {np.median(best):.3f} max {best.max():.3f}; '
        f'coverage {cov.min():.2f}-{cov.max():.2f}; QD-score {qd.min():.1f}-{qd.max():.1f}'
    )
    print(f'seeds reaching 14.391 MW: {int((best < 14.392).sum())} of 10')
    never = [cell for cell in np.ndindex(CELLS) if not any(a.filled[cell] for a in archives)]
    print('cells never filled by any seed:', never)
    ga = [ga_1d(seed) for seed in range(20)]
    me = [map_elites_1d(seed) for seed in range(20)]
    print(
        f'deceptive 1-D: GA finds x=85 in {sum(x == 85 for x in ga)}/20 seeds, '
        f'MAP-Elites in {sum(x == 85 for x in me)}/20; GA results {sorted(set(ga))}'
    )
    short = [map_elites_1d(seed, step_size=2) for seed in range(20)]
    print(f'with steps of at most 2 (a fifth of a cell) MAP-Elites finds x=85 in {sum(x == 85 for x in short)}/20')
    print('fitness at x = 20, 50, 60, 70, 80, 83, 85:', np.round(deceptive_fitness([20, 50, 60, 70, 80, 83, 85]), 4))


if __name__ == '__main__':
    main()
