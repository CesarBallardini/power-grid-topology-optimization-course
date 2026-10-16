"""Surrogate-assisted pre-screening in a genetic algorithm for IEEE 30 line switching (Chapter 29, Exercise 4).

The problem and the genotype are those of Chapter 24 (at most ``K_MAX`` switched lines, N-1 overload energy as the
cost, evaluated with ``ch27_ieee30.overload_energy`` from the Chapter 27 folder). The plain GA evaluates every
offspring exactly; the assisted GA fits a linear surrogate (one weight per line plus a bias, least squares on
every genome evaluated so far), scores five times as many offspring with it and evaluates exactly only the best
20 %. Run it to print the comparison over ten seeds::

    python surrogate_ga.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ch27-pareto-multi-objective' / 'src'))
import ch27_ieee30  # noqa: E402

K_MAX = 2
TARGET = 44.7  # the exact optimum with two switched lines is 44.6 MW (Chapter 27)
BUDGET = 400  # exact evaluations per run
POP = 20
PENALTY = 1e4


# --8<-- [start:ga]
def features(genome: np.ndarray, n_line: int) -> np.ndarray:
    """One-hot of the switched lines plus a bias: the linear surrogate assumes additive switching effects."""
    f = np.zeros(n_line + 1)
    f[genome[genome < n_line]] = 1
    f[-1] = 1
    return f


def exact(case: dict, genome: np.ndarray, n_line: int) -> float:
    """N-1 overload energy of the switched lines of a genome, islanding capped at ``PENALTY``."""
    return min(ch27_ieee30.overload_energy(case, tuple(int(g) for g in genome if g < n_line)), PENALTY)


def random_genome(rng: np.random.Generator, n_line: int) -> np.ndarray:
    """One to ``K_MAX`` distinct lines, sorted and padded with the sentinel ``n_line``."""
    k = rng.integers(1, K_MAX + 1)
    g = np.full(K_MAX, n_line)
    g[:k] = rng.choice(n_line, k, replace=False)
    return np.sort(g)


def offspring(rng: np.random.Generator, parents: np.ndarray, n_line: int) -> np.ndarray:
    """Tournament of two, crossover by sampling from the union of the parents, mutation of one slot."""
    a, b = parents[rng.integers(len(parents))], parents[rng.integers(len(parents))]
    pool = np.unique(np.concatenate([a[a < n_line], b[b < n_line]]))
    k = min(len(pool), rng.integers(1, K_MAX + 1)) if len(pool) else 0
    g = np.full(K_MAX, n_line)
    if k:
        g[:k] = rng.choice(pool, k, replace=False)
    if rng.random() < 0.5:
        g[rng.integers(K_MAX)] = n_line if rng.random() < 0.3 else rng.integers(n_line)
    g = np.unique(g[g < n_line])
    return np.concatenate([g, np.full(K_MAX - len(g), n_line)])


def run(case: dict, seed: int, assisted: bool, budget: int = BUDGET, pop: int = POP) -> dict:
    """One GA run: best value, exact evaluations spent and the evaluation at which ``TARGET`` was first reached."""
    rng = np.random.default_rng(seed)
    n_line = len(case['lines'])
    genomes = np.array([random_genome(rng, n_line) for _ in range(pop)])
    values = np.array([exact(case, g, n_line) for g in genomes])
    archive_x, archive_y = [features(g, n_line) for g in genomes], list(values)
    used, reached = pop, None
    best = values.min()
    if best <= TARGET:
        reached = int(np.argmax(values <= TARGET)) + 1
    while used < budget:
        order = np.argsort(values)
        parents = genomes[order[: pop // 2]]
        if assisted:
            candidates = np.array([offspring(rng, parents, n_line) for _ in range(5 * pop)])
            w, *_ = np.linalg.lstsq(np.array(archive_x), np.array(archive_y), rcond=None)
            scores = np.array([features(g, n_line) for g in candidates]) @ w
            children = candidates[np.argsort(scores)[:pop]]
        else:
            children = np.array([offspring(rng, parents, n_line) for _ in range(pop)])
        child_values = np.array([exact(case, g, n_line) for g in children])
        for g, v in zip(children, child_values, strict=True):
            used += 1
            archive_x.append(features(g, n_line))
            archive_y.append(v)
            if reached is None and v <= TARGET:
                reached = used
        genomes = np.vstack([genomes, children])
        values = np.concatenate([values, child_values])
        keep = np.argsort(values)[:pop]
        genomes, values = genomes[keep], values[keep]
        best = min(best, values.min())
    return {'best': float(best), 'evaluations': used, 'reached_at': reached}


def compare(case: dict, seeds: range = range(10)) -> dict[str, list[dict]]:
    """Plain against assisted GA on the same seeds."""
    return {'plain': [run(case, s, False) for s in seeds], 'assisted': [run(case, s, True) for s in seeds]}


# --8<-- [end:ga]
def main() -> None:
    """Evaluations to target over ten seeds."""
    case = ch27_ieee30.load_case()
    results = compare(case)
    for name, runs in results.items():
        hits = [r['reached_at'] for r in runs if r['reached_at'] is not None]
        spread = f'median {np.median(hits):.0f}, min {min(hits)}, max {max(hits)}' if hits else 'never'
        print(
            f'{name:9s} reached {TARGET} MW in {len(hits)}/10 runs; evaluations to target: {spread}; '
            f'best value median {np.median([r["best"] for r in runs]):.1f} MW'
        )


if __name__ == '__main__':
    main()
