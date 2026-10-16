"""Chapter 27, Lab 5: MOME (QDax's ``MOMERepertoire``) on the IEEE 30 line-switching problem of Lab 6.

Runs in the labs container (``make test-labs``, see ``docker/labs``), on the CPU. A ToOp run was not available, so
the problem is the one of ``ch27_ieee30.py``: switch off at most three of the 41 lines of IEEE 30 (loads and
generation scaled by 1.5). The two objectives are those the lab names, negated because QDax maximizes: the N-1
overload energy and the critical branch count (lines overloaded in at least one N-1 case), each divided by its value
on the intact network, so the hypervolume reference point (1.1, 1.1) only credits switchings that are at most 10 %
worse than doing nothing. The descriptors play the role of ToOp's ``switching_distance`` and
``split_subs``: the number of switched lines and the number of buses at their ends. Every feasible switching is
enumerated once on the host (the same 11,522 sets as Lab 6), so the JAX scoring function is a table lookup and the
exact per-cell Pareto fronts are known. Run it to print every number quoted in the solutions (about a minute on
one CPU core)::

    python mome_labs.py
"""

import itertools
import time
import warnings
from functools import partial

import ch27_ieee30 as ieee30
import jax
import jax.numpy as jnp
import moocore
import numpy as np
import pareto
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.mome import MOME
from qdax.utils.metrics import default_moqd_metrics

warnings.filterwarnings('ignore', message='This type of repertoire does not store the extra scores')

K_MAX = 3  # at most this many switched lines, as in Lab 6
FRONT_LEN = 8  # maximum size of the Pareto front kept per cell
BATCH = 32  # offspring per generation
ITERATIONS = 40  # generations: 32 + 40 x 32 = 1,312 evaluations, about 11 % of the enumeration
SEEDS = range(5)
REFERENCE = np.array([-1.1, -1.1])  # hypervolume reference point of the normalized, negated objectives


# --8<-- [start:table]
def n1_metrics(case: dict, switched_off: tuple[int, ...] = ()) -> tuple[float, int]:
    """N-1 overload energy (MW) and critical branch count of the network without ``switched_off``.

    The same post-contingency flows as ``ch27_ieee30.overload_energy`` (bridge outages left out); a branch is critical
    when its flow exceeds its rating in at least one outage. ``(inf, -1)`` when the network is islanded.
    """
    in_service = np.ones(len(case['lines']), dtype=bool)
    in_service[list(switched_off)] = False
    h = ieee30.ptdf(case, in_service)
    if h is None:
        return np.inf, -1
    lines, rating = case['lines'][in_service], case['rating'][in_service]
    f = h @ case['p']
    hk = h[:, lines[:, 0]] - h[:, lines[:, 1]]
    denominator = 1 - np.diag(hk)
    not_bridge = np.abs(denominator) > 1e-9
    post = f[:, None] + hk[:, not_bridge] / denominator[not_bridge] * f[not_bridge]
    post[np.flatnonzero(not_bridge), np.arange(not_bridge.sum())] = 0
    excess = np.maximum(0, np.abs(post) - rating[:, None])
    return float(excess.sum()), int(np.any(excess > 1e-9, axis=1).sum())


def build_table(case: dict, k_max: int = K_MAX) -> dict:
    """Objectives and descriptors of every switching of at most ``k_max`` lines, indexed by its sorted genotype.

    A genotype is ``k_max`` line indices sorted ascending and padded with the sentinel ``n_line`` (Chapter 24's
    encoding), so ``table['energy'][g0, g1, g2]`` is its overload energy. Returns the dense arrays, the list of
    feasible genotypes (not islanded) and the normalization of the objectives (their intact-network values).
    """
    n_line = len(case['lines'])
    shape = (n_line + 1,) * k_max
    energy, critical = np.full(shape, np.inf), np.full(shape, -1)
    descriptor = np.zeros((*shape, 2))
    feasible = []
    for k in range(k_max + 1):
        for subset in itertools.combinations(range(n_line), k):
            g = (*subset, *(n_line,) * (k_max - k))
            energy[g], critical[g] = n1_metrics(case, subset)
            descriptor[g] = (k, len(np.unique(case['lines'][list(subset)]))) if k else (0, 0)
            if np.isfinite(energy[g]):
                feasible.append(g)
    feasible = np.array(feasible)
    scale = np.array(n1_metrics(case), dtype=float)
    fitness = -np.stack([energy, critical], axis=-1) / scale  # QDax maximizes
    fitness[~np.isfinite(energy)] = -np.inf
    return {
        'n_line': n_line,
        'energy': energy,
        'critical': critical,
        'fitness': fitness,
        'descriptor': descriptor,
        'feasible': feasible,
        'scale': scale,
    }


def cell_list(k_max: int = K_MAX) -> np.ndarray:
    """The reachable descriptor cells: ``k`` switched lines touch between ``k + 1`` and ``2 k`` buses."""
    return np.array([(0, 0)] + [(k, b) for k in range(1, k_max + 1) for b in range(k + 1, 2 * k + 1)], dtype=float)


def exact_cells(table: dict) -> dict[tuple[int, int], np.ndarray]:
    """Exact Pareto front of every cell, as minimized normalized objectives (energy, critical) / scale."""
    g = tuple(table['feasible'].T)
    points = -table['fitness'][g]
    descriptors = table['descriptor'][g].astype(int)
    out = {}
    for cell in cell_list().astype(int):
        members = points[np.all(descriptors == cell, axis=1)]
        out[tuple(cell)] = np.unique(members[pareto.non_dominated_mask(members)], axis=0)
    return out


# --8<-- [end:table]
# --8<-- [start:mome]
def make_scoring(table: dict):
    """JAX scoring function ``(genotypes, key) -> (fitnesses, descriptors, extra)``: a lookup in the table.

    The repertoire stores genotypes as floats (QDax fills empty slots with zeros of the default dtype), so the
    integer line indices are recovered by rounding and sorting before the lookup.
    """
    fitness, descriptor = jnp.asarray(table['fitness']), jnp.asarray(table['descriptor'])

    def scoring(genotypes, key):
        g = jnp.sort(jnp.round(genotypes).astype(jnp.int32), axis=1)
        index = tuple(g[:, i] for i in range(g.shape[1]))
        return fitness[index], descriptor[index], {}

    return scoring


def make_mutation(table: dict):
    """Chapter 24's mutation in JAX: one slot becomes the sentinel (p = 0.3) or a random line; duplicates removed.

    A child that islands the grid is replaced by its parent, so every genotype in the repertoire is feasible and its
    fitness finite (QDax marks empty slots with -inf).
    """
    n_line = table['n_line']
    feasible = jnp.asarray(np.isfinite(table['energy']))

    def mutate_one(genotype, key):
        k_slot, k_kind, k_line = jax.random.split(key, 3)
        g = jnp.round(genotype).astype(jnp.int32)
        slot = jax.random.randint(k_slot, (), 0, g.shape[0])
        value = jnp.where(jax.random.uniform(k_kind) < 0.3, n_line, jax.random.randint(k_line, (), 0, n_line))
        child = jnp.sort(g.at[slot].set(value))
        duplicate = jnp.concatenate([jnp.array([False]), child[1:] == child[:-1]])
        child = jnp.sort(jnp.where(duplicate, n_line, child))
        ok = feasible[tuple(child[i] for i in range(child.shape[0]))]
        return jnp.where(ok, child, g).astype(genotype.dtype)

    def mutation(genotypes, key):
        return jax.vmap(mutate_one)(genotypes, jax.random.split(key, genotypes.shape[0]))

    return mutation


def run_mome(table: dict, seed: int, iterations: int = ITERATIONS, batch: int = BATCH):
    """One MOME run; returns the final repertoire and its metrics (QDax's ``default_moqd_metrics``)."""
    rng = np.random.default_rng(seed)
    start = table['feasible'][rng.choice(len(table['feasible']), batch, replace=False)]
    emitter = MixingEmitter(
        mutation_fn=make_mutation(table),
        variation_fn=lambda x1, x2, key: x1,  # unused: variation_percentage is 0
        variation_percentage=0.0,
        batch_size=batch,
    )
    mome = MOME(
        scoring_function=make_scoring(table),
        emitter=emitter,
        metrics_function=partial(default_moqd_metrics, reference_point=jnp.asarray(REFERENCE)),
        pareto_front_max_length=FRONT_LEN,
    )
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, metrics = mome.init(
        jnp.asarray(start, dtype=jnp.float32), jnp.asarray(cell_list()), subkey
    )
    update = jax.jit(mome.update)
    for _ in range(iterations):
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = update(repertoire, emitter_state, subkey)
    return repertoire, metrics


def repertoire_points(repertoire) -> list[np.ndarray]:
    """The stored front of every cell, as minimized normalized objectives, empty slots dropped.

    QDax keeps copies of the same point (equal points do not dominate each other), so the rows are made unique.
    """
    fitnesses = np.asarray(repertoire.fitnesses)
    return [np.unique(-f[np.all(np.isfinite(f), axis=1)], axis=0) for f in fitnesses]


def moqd_score(fronts) -> float:
    """MOQD score: the sum of the per-cell hypervolumes with the reference point ``-REFERENCE`` (minimized).

    Points beyond the reference add nothing here; QDax's ``compute_hypervolume`` does not clip them and counts a
    negative area instead, so its ``moqd_score`` metric is lower whenever a cell holds such a point.
    """
    return float(sum(pareto.hypervolume_2d(f, -REFERENCE) for f in fronts if len(f)))


def three_objective_hv(points: np.ndarray) -> float:
    """Hypervolume in (energy, critical, switched lines / K_MAX), minimized, with moocore on the host (outside jit)."""
    return float(moocore.hypervolume(points, ref=np.full(3, 1.1)))


def repertoire_bytes(repertoire) -> int:
    """Memory of the per-cell fronts: genotypes, fitnesses and descriptors of every cell and slot."""
    return int(sum(np.asarray(x).nbytes for x in (repertoire.genotypes, repertoire.fitnesses, repertoire.descriptors)))


# --8<-- [end:mome]
def main() -> None:
    """Lab 5: exact per-cell fronts, MOME on five seeds, the scalar repertoire's losses, 3-objective hypervolume."""
    case = ieee30.load_case()
    t0 = time.perf_counter()
    table = build_table(case)
    energy0, critical0 = table['scale']
    print(
        f'table of {len(table["feasible"])} feasible switchings (|D| <= {K_MAX}) in {time.perf_counter() - t0:.1f} s; '
        f'intact network: overload energy {energy0:.1f} MW, {int(critical0)} critical branches'
    )
    exact = exact_cells(table)
    print(f'{len(exact)} reachable cells (switched lines, buses touched); exact per-cell fronts (MW, branches):')
    for cell, front in exact.items():
        values = [(round(float(e * energy0), 1), round(float(c * critical0))) for e, c in front]
        print(f'  {tuple(int(v) for v in cell)}: {values}')
    n_exact = sum(len(f) for f in exact.values())
    print(
        f'exact: {n_exact} non-dominated points over the cells; a scalar repertoire (one elite per cell) keeps '
        f'{len(exact)}, so it discards {n_exact - len(exact)}'
    )
    exact_score = moqd_score(exact.values())
    print(f'exact MOQD score {exact_score:.4f} (reference {(-REFERENCE).tolist()} in units of the intact values)')
    feasible = tuple(table['feasible'].T)
    all_points = np.hstack([-table['fitness'][feasible], table['descriptor'][feasible][:, :1] / K_MAX])
    exact_3d = np.unique(all_points[pareto.non_dominated_mask(all_points)], axis=0)
    print(
        f'exact 3-objective front (energy, critical, switched lines / {K_MAX}): {len(exact_3d)} points, '
        f'moocore HV {three_objective_hv(exact_3d):.4f} (reference 1.1 each)'
    )

    exact_all = np.vstack(list(exact.values()))
    scores, found, extra, hv3, seconds = [], [], [], [], []
    for seed in SEEDS:
        t0 = time.perf_counter()
        repertoire, metrics = run_mome(table, seed)
        seconds.append(time.perf_counter() - t0)
        fronts = repertoire_points(repertoire)
        qdax_score = float(np.where(np.isfinite(metrics['moqd_score']), metrics['moqd_score'], 0).sum())
        scores.append(moqd_score(fronts))
        stored = np.vstack([f for f in fronts if len(f)])
        found.append(int(sum(np.any(np.all(np.isclose(stored, p), axis=1)) for p in exact_all)))
        extra.append(len(stored) - found[-1])
        beyond = int(np.any(stored >= -REFERENCE, axis=1).sum())
        k = np.asarray(repertoire.centroids)[:, :1] / K_MAX
        points_3d = np.vstack(
            [np.hstack([f, np.repeat(k[i][None], len(f), 0)]) for i, f in enumerate(fronts) if len(f)]
        )
        hv3.append(three_objective_hv(points_3d[pareto.non_dominated_mask(points_3d)]))
        print(
            f'seed {seed}: {seconds[-1]:.1f} s, MOQD score {scores[-1]:.4f} (QDax metric {qdax_score:.4f}), '
            f'coverage {float(metrics["coverage"]):.0f} %, {len(stored)} distinct points stored '
            f'({beyond} beyond the reference), '
            f'{found[-1]}/{n_exact} exact points found, 3-objective HV {hv3[-1]:.4f}'
        )
    print(
        f'MOME, {len(SEEDS)} seeds ({BATCH} + {ITERATIONS} x {BATCH} = {BATCH * (ITERATIONS + 1)} evaluations each): '
        f'MOQD score median {np.median(scores):.4f} (min {min(scores):.4f}), {np.median(scores) / exact_score:.1%} '
        f'of the exact {exact_score:.4f}; exact points found median {int(np.median(found))}/{n_exact} '
        f'(min {min(found)}), stored points off the exact fronts median {int(np.median(extra))}; '
        f'3-objective HV median {np.median(hv3):.4f}; {np.median(seconds):.1f} s per run'
    )
    print(
        f'per-cell fronts in memory: {len(exact)} cells x {FRONT_LEN} slots x (3 genes + 2 objectives + 2 descriptors)'
        ' '
        f'x 4 bytes = {repertoire_bytes(repertoire)} bytes'
    )


if __name__ == '__main__':
    main()
