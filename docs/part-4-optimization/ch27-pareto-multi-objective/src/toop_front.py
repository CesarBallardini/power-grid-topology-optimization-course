"""Pareto analysis of real ToOp repertoires (Chapter 27, Labs 1, 3, 4 and 7).

The ``res.json`` files in ``toop_front/`` were written by ``toop_front_toop.py`` in the course's ToOp container
(complex example grid, 60 s of DC search per run; ``toop_front/wall_seconds.json`` holds each run's wall time).
This module only reads them, so it runs anywhere::

    python toop_front.py

``lab1`` extracts the non-dominated set of a repertoire, ``lab3`` sweeps the weight of the critical branch count,
``lab4`` compares two descriptor resolutions over five seeds with hypervolume, IGD+ and the additive epsilon
indicator, ``gpu_vs_cpu`` compares GPU runs with CPU runs (``toop_front_gpu/``), and ``lab7`` picks knee,
pseudo-weight and achievement-function solutions from a front.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pareto
from repertoire_cells import METRICS, cell_index, exact_cell_index
from scipy.stats import mannwhitneyu

RUNS = Path(__file__).resolve().parent / 'toop_front'
GPU_RUNS = Path(__file__).resolve().parent / 'toop_front_gpu'  # GPU runs and their CPU reruns
SEEDS = range(1, 6)
WEIGHTS = (0, 1, 3, 10, 30, 100)  # w = 0 is the run fine_s1
BI = ['overload_energy_n_1', 'switching_distance']  # the two objectives of Labs 4 and 7
DECIMALS = 6  # metrics are compared to 1e-6: equal topologies differ by about 1e-12 MW of float noise


def as_tuple(point) -> tuple[float, ...]:
    """A point as a tuple of plain floats (for printing and comparing)."""
    return tuple(float(v) for v in point)


def load(name: str, runs: Path = RUNS) -> dict:
    """The stored ``res.json`` of one run."""
    return json.loads((runs / f'{name}.json').read_text(encoding='utf-8'))


def n_cells(res: dict) -> tuple[int, ...]:
    """The repertoire shape of a run, from its ``me_descriptors``."""
    return tuple(d['num_cells'] for d in res['args']['ga_config']['me_descriptors'])


# --8<-- [start:table]
def table(name: str, runs: Path = RUNS) -> pd.DataFrame:
    """One row per elite of ``best_topos``: actions, fitness, the four metrics (rounded) and the (clipped) cell.

    ``fitness`` is kept as ToOp wrote it, so its maximum is ToOp's winner; the metrics are rounded to ``DECIMALS``
    so that two topologies with the same flows compare as equal in the dominance tests.
    """
    res = load(name, runs)
    shape = n_cells(res)
    rows = []
    for topo in res['best_topos']:
        scores = topo['metrics']['extra_scores']
        row = {'actions': topo['actions'], 'fitness': topo['metrics']['fitness']}
        row.update({m: round(scores[m], DECIMALS) for m in METRICS})
        descriptor = (int(scores['split_subs']), int(scores['switching_distance']))
        row['cell'] = cell_index(descriptor, shape)
        row['exact_cell'] = exact_cell_index(descriptor, shape)
        rows.append(row)
    return pd.DataFrame(rows)


def lab1(name: str = 'fine_s1') -> dict:
    """Non-dominated elites in the four metrics, the best-fitness elite, and the elites of clipped edge cells."""
    t = table(name)
    nd = pareto.non_dominated_mask(t[METRICS].to_numpy(float))
    best = int(t['fitness'].idxmax())
    clipped = t['exact_cell'].isna().to_numpy()
    return {
        'elites': len(t),
        'non_dominated': int(nd.sum()),
        'dominated': int((~nd).sum()),
        'best': best,
        'best_non_dominated': bool(nd[best]),
        'edge_elites': int(clipped.sum()),
        'edge_elites_dominated': int((clipped & ~nd).sum()),
        'mask': nd,
    }


# --8<-- [end:table]
# --8<-- [start:weights]
def weight_runs() -> dict[int, pd.DataFrame]:
    """The repertoire of each weight ``w`` in ``-(overload_energy_n_1 + w critical_branch_count_n_1)``."""
    return {w: table('fine_s1' if w == 0 else f'w{w}') for w in WEIGHTS}


def lab3() -> dict:
    """The winner of every weight, the joint bi-objective front and the points no weight can reach."""
    runs = weight_runs()
    objectives = ['overload_energy_n_1', 'critical_branch_count_n_1']
    winners = {w: as_tuple(t.loc[t['fitness'].idxmax(), objectives]) for w, t in runs.items()}
    points = np.unique(np.vstack([t[objectives].to_numpy(float) for t in runs.values()]), axis=0)
    front = points[pareto.non_dominated_mask(points)]
    front = front[np.argsort(front[:, 0])]
    # what each weight would pick from everything the six runs found together
    best_known = {w: as_tuple(front[pareto.weighted_sum_argmin(front, np.array([1.0, w]))[0]]) for w in runs}
    dominated_winners = [w for w, p in winners.items() if not np.any(np.all(front == p, axis=1))]
    supported = pareto.supported_mask(front)
    unsupported = front[~supported]
    # the alternatives reach the unsupported points: epsilon-constraint and achievement function
    eps = {
        as_tuple(p): as_tuple(front[pareto.epsilon_constraint(front, 0, np.array([0.0, p[1]]))]) for p in unsupported
    }
    asf = {as_tuple(p): as_tuple(front[pareto.achievement_scalarizing(front, p - 1.0)]) for p in unsupported}
    return {
        'winners': winners,
        'best_known': best_known,
        'dominated_winners': dominated_winners,
        'front': front,
        'supported': supported,
        'unsupported': unsupported,
        'epsilon': eps,
        'asf': asf,
    }


# --8<-- [end:weights]
# --8<-- [start:quality]
def front_of(name: str, objectives=BI, runs: Path = RUNS) -> np.ndarray:
    """The non-dominated set of a run's repertoire in ``objectives``, without duplicates."""
    points = np.unique(table(name, runs)[objectives].to_numpy(float), axis=0)
    return points[pareto.non_dominated_mask(points)]


def lab4(reference_factor: float = 1.1) -> pd.DataFrame:
    """Hypervolume, IGD+ and epsilon of every run, normalized with the ideal and nadir of the joint front."""
    names = [f'{c}_s{s}' for c in ('fine', 'coarse') for s in SEEDS]
    fronts = {n: front_of(n) for n in names}
    wall = json.loads((RUNS / 'wall_seconds.json').read_text(encoding='utf-8'))
    joint = np.vstack(list(fronts.values()))
    joint = np.unique(joint[pareto.non_dominated_mask(joint)], axis=0)
    ideal, nadir = joint.min(axis=0), joint.max(axis=0)
    scale = lambda p: (p - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)  # noqa: E731
    reference = np.full(2, reference_factor)
    rows = []
    for name, front in fronts.items():
        res = load(name)
        rows.append(
            {
                'run': name,
                'config': name.split('_')[0],
                'epochs': res['iteration'],
                'wall_seconds': wall[name]['wall_seconds'],
                'elites': len(res['best_topos']),
                'front': len(front),
                'hv': pareto.hypervolume_2d(scale(front), reference),
                'igd_plus': pareto.igd_plus(scale(front), scale(joint)),
                'epsilon': pareto.epsilon_indicator(scale(front), scale(joint)),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(joint=joint, joint_hv=pareto.hypervolume_2d(scale(joint), reference))
    return out


def compare(results: pd.DataFrame, column: str, configs: tuple[str, str] = ('fine', 'coarse')) -> dict:
    """Median and range per configuration, and the two-sided exact Mann-Whitney p-value of the two."""
    groups = {c: results.loc[results['config'] == c, column].to_numpy() for c in configs}
    summary = {c: (float(np.median(v)), float(v.min()), float(v.max())) for c, v in groups.items()}
    p = mannwhitneyu(*groups.values(), alternative='two-sided', method='exact').pvalue
    return {**summary, 'p': float(p)}


# --8<-- [end:quality]
# --8<-- [start:gpu]
def gpu_vs_cpu(runs: Path = GPU_RUNS, reference_factor: float = 1.1) -> pd.DataFrame:
    """The default repertoire (``fine``), seeds 1 to 5, on the GPU and again on the CPU in the same environment.

    Work done in the 60 s (epochs, and evaluations: ToOp's ``total_branch_combis``), the elites and how many of
    them are non-dominated in the four metrics of Lab 1, and the hypervolume and IGD+ of each bi-objective front,
    normalized with the ideal and nadir of Lab 4's joint front, so that 0.8364 means "Lab 4's front reached".
    """
    r4 = lab4(reference_factor)
    joint = r4.attrs['joint']
    ideal, nadir = joint.min(axis=0), joint.max(axis=0)
    scale = lambda p: (p - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)  # noqa: E731
    reference = np.full(2, reference_factor)
    wall = json.loads((runs / 'wall_seconds.json').read_text(encoding='utf-8'))
    rows, fronts = [], []
    for config in ('gpu', 'cpu'):
        for seed in SEEDS:
            name = f'{config}_s{seed}'
            res, t, front = load(name, runs), table(name, runs), front_of(name, runs=runs)
            fronts.append(front)
            rows.append(
                {
                    'run': name,
                    'config': config,
                    'device': wall[name]['device'],
                    'epochs': res['iteration'],
                    'evaluations': res['total_branch_combis'],
                    'wall_seconds': wall[name]['wall_seconds'],
                    'elites': len(t),
                    'non_dominated': int(pareto.non_dominated_mask(t[METRICS].to_numpy(float)).sum()),
                    'hv': pareto.hypervolume_2d(scale(front), reference),
                    'igd_plus': pareto.igd_plus(scale(front), scale(joint)),
                }
            )
    out = pd.DataFrame(rows)
    together = np.unique(np.vstack(fronts), axis=0)
    together = together[pareto.non_dominated_mask(together)]
    out.attrs.update(
        lab4_joint_hv=r4.attrs['joint_hv'],
        joint=together,
        joint_hv=pareto.hypervolume_2d(scale(together), reference),
    )
    return out


# --8<-- [end:gpu]
# --8<-- [start:decision]
def lab7(name: str = 'fine_s1', switching: float = 3.0, splits: float = 1.0) -> dict:
    """Knee and pseudo-weights on the (overload, switching distance) front; ASF with an operator's reference point.

    The reference is (overload energy, switching distance, split substations) = (the front's lowest overload,
    ``switching``, ``splits``): "the best relief, with at most 3 switching actions and 1 split". The objectives
    are normalized by the ranges of the three-objective front before the ASF.
    """
    front = front_of(name)
    front = front[np.argsort(front[:, 0])]
    knees = pareto.knee_points(front)
    t = table(name)
    three = ['overload_energy_n_1', 'switching_distance', 'split_subs']
    points = np.unique(t[three].to_numpy(float), axis=0)
    points = points[pareto.non_dominated_mask(points)]
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    reference = np.array([lo[0], switching, splits])
    pick = pareto.achievement_scalarizing((points - lo) / span, (np.asarray(reference) - lo) / span)
    feasible = points[(points[:, 1] <= reference[1]) & (points[:, 2] <= reference[2])]
    return {
        'front': front,
        'knees': front[knees],
        'pseudo_weights': pareto.pseudo_weights(front),
        'reference': reference,
        'asf': points[pick],
        'best_feasible': as_tuple(feasible[np.argmin(feasible[:, 0])]) if len(feasible) else None,
        'weighted_sum_winner': as_tuple(t.loc[t['fitness'].idxmax(), three]),
    }


# --8<-- [end:decision]
def main() -> None:
    """Print every number the Lab 1, 3, 4 and 7 solutions quote."""
    pd.set_option('display.width', 160)
    r = lab1()
    t = table('fine_s1')
    print('Lab 1 (fine_s1, 5 x 45 cells):', {k: v for k, v in r.items() if k != 'mask'})
    print(t.loc[r['best'], ['actions', 'fitness', *METRICS]].to_dict())
    print(t.assign(non_dominated=r['mask']).sort_values('fitness', ascending=False).to_string(index=False))
    c = lab1('coarse_s1')
    tc = table('coarse_s1')
    print('Lab 1 (coarse_s1, 5 x 9 cells):', {k: v for k, v in c.items() if k != 'mask'})
    print(tc[tc['exact_cell'].isna()].to_string(index=False))
    r3 = lab3()
    print('Lab 3 winners (overload MW, critical branches) per weight:', r3['winners'])
    print('  best of the joint front per weight:', r3['best_known'])
    print('  weights whose winner is dominated by a point another run found:', r3['dominated_winners'])
    print('joint front:', [as_tuple(p) for p in r3['front']])
    print('supported:', r3['supported'].tolist())
    print('unsupported points:', [as_tuple(p) for p in r3['unsupported']])
    print('epsilon-constraint picks:', r3['epsilon'])
    print('ASF picks (reference 1 MW and 1 branch below the point):', r3['asf'])
    r4 = lab4()
    print('Lab 4 (reference 1.1 x nadir after normalization):')
    print(r4.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('joint front:', [as_tuple(p) for p in r4.attrs['joint']], 'hv', round(r4.attrs['joint_hv'], 4))
    for column in ('hv', 'igd_plus', 'epsilon', 'epochs'):
        print(f'  {column}:', compare(r4, column))
    r4b = lab4(2.0)
    print('  hv with reference 2.0:', compare(r4b, 'hv'))
    g = gpu_vs_cpu()
    print('GPU against CPU (fine, seeds 1-5, same environment; normalized as Lab 4):')
    print(g.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(
        '  joint front of the ten runs:', [as_tuple(p) for p in g.attrs['joint']], 'hv', round(g.attrs['joint_hv'], 4)
    )
    print(
        '  runs reaching the Lab 4 hypervolume',
        round(g.attrs['lab4_joint_hv'], 4),
        ':',
        g.loc[np.isclose(g['hv'], g.attrs['lab4_joint_hv']), 'run'].tolist(),
    )
    for column in ('epochs', 'evaluations', 'non_dominated', 'hv', 'igd_plus'):
        print(f'  {column}:', compare(g, column, ('gpu', 'cpu')))
    r7 = lab7()
    print('Lab 7 front (overload, switching distance):', [as_tuple(p) for p in r7['front']])
    print('  knees:', [as_tuple(p) for p in r7['knees']])
    print('  pseudo-weights:', np.round(r7['pseudo_weights'], 3).tolist())
    print('  ASF pick for the reference', as_tuple(r7['reference']), ':', as_tuple(r7['asf']))
    print('  best with switching <= 3 and split <= 1:', r7['best_feasible'])
    print('  weighted-sum winner (overload, switching, splits):', r7['weighted_sum_winner'])


if __name__ == '__main__':
    main()
