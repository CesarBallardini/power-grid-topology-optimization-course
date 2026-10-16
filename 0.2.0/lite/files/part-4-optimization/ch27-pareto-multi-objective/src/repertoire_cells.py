"""ToOp's repertoire cells as constraints, and the non-dominated set of a repertoire table (Chapter 27, Lab 1, Ex. 5).

ToOp maps the descriptor (split_subs, switching_distance) to a cell with ``ravel_multi_index(mode="clip")``, so the
last cell of each axis also holds every larger value. ``table_from_res_json`` reads a ToOp ``res.json``; no ToOp run
was available when the solutions were written, so the numbers come from ``synthetic_repertoire``. Run it with::

    python repertoire_cells.py [res.json]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pareto

N_CELLS = (5, 45)  # ToOp's default: split_subs with 5 cells, switching_distance with 45
METRICS = ['overload_energy_n_1', 'switching_distance', 'split_subs', 'critical_branch_count_n_1']


# --8<-- [start:cells]
def cell_index(descriptor, n_cells: tuple[int, ...] = N_CELLS) -> int:
    """ToOp's cell of a descriptor: every coordinate is clipped into its axis before raveling."""
    return int(np.ravel_multi_index(np.asarray(descriptor, dtype=int), n_cells, mode='clip'))


def exact_cell_index(descriptor, n_cells: tuple[int, ...] = N_CELLS) -> int | None:
    """The cell under "exactly equal" constraints: ``None`` when a coordinate lies outside its axis."""
    d = np.asarray(descriptor, dtype=int)
    if np.any(d < 0) or np.any(d >= np.asarray(n_cells)):
        return None
    return int(np.ravel_multi_index(d, n_cells))


# --8<-- [end:cells]
# --8<-- [start:table]
def table_from_res_json(path: Path) -> pd.DataFrame:
    """One row per ``best_topos`` entry of a ToOp ``res.json``: fitness and the metrics of ``METRICS``."""
    topos = json.loads(Path(path).read_text(encoding='utf-8'))['best_topos']
    rows = [
        {'fitness': t['metrics']['fitness'], **{m: t['metrics']['extra_scores'][m] for m in METRICS}} for t in topos
    ]
    return pd.DataFrame(rows)


def synthetic_repertoire(seed: int = 27, n: int = 60) -> pd.DataFrame:
    """A made-up repertoire table in the shape ToOp writes, with one elite per cell and a weighted-sum fitness."""
    rng = np.random.default_rng(seed)
    split_subs = rng.integers(1, 8, n)
    switching_distance = split_subs * rng.integers(2, 12, n)
    overload = np.round(300 / (1 + split_subs) + rng.normal(0, 25, n).clip(-60, 60) + 30, 1)
    critical = np.maximum(0, (overload / 40 + rng.normal(0, 1.5, n)).round()).astype(int)
    df = pd.DataFrame(
        {
            'overload_energy_n_1': overload.clip(0),
            'switching_distance': switching_distance,
            'split_subs': split_subs,
            'critical_branch_count_n_1': critical,
        }
    )
    df['fitness'] = -(df['overload_energy_n_1'] + 10 * df['critical_branch_count_n_1'])
    df['cell'] = [cell_index(d) for d in df[['split_subs', 'switching_distance']].to_numpy()]
    return (
        df.sort_values('fitness', ascending=False).drop_duplicates('cell').sort_values('cell').reset_index(drop=True)
    )


def analyse(table: pd.DataFrame, n_cells: tuple[int, ...] = N_CELLS) -> dict:
    """Non-dominated rows, the best-fitness row, how many elites are dominated, and the edge-cell elites."""
    points = table[METRICS].to_numpy(float)
    nd = pareto.non_dominated_mask(points)
    exact = [exact_cell_index(d, n_cells) for d in table[['split_subs', 'switching_distance']].to_numpy()]
    clipped = np.array([c is None for c in exact])
    return {
        'non_dominated': np.flatnonzero(nd),
        'best_fitness': int(table['fitness'].idxmax()),
        'dominated_count': int((~nd).sum()),
        'edge_cell_elites': np.flatnonzero(clipped),  # with exactly-equal constraints they would have no cell
    }


# --8<-- [end:table]
def main() -> None:
    """Exercise 5 cells, then the analysis of a res.json if given, else of the synthetic table."""
    print('Exercise 5, n_cells_per_dim =', N_CELLS)
    for d in ((1, 2), (2, 5), (4, 44), (4, 60), (7, 44)):
        print(f'  descriptor {d}: cell {cell_index(d)}, exact cell {exact_cell_index(d)}')
    table = table_from_res_json(Path(sys.argv[1])) if len(sys.argv) > 1 else synthetic_repertoire()
    print('synthetic' if len(sys.argv) == 1 else sys.argv[1], f'repertoire: {len(table)} elites')
    result = analyse(table)
    best = result['best_fitness']
    print(f'  non-dominated elites: {len(result["non_dominated"])}, dominated: {result["dominated_count"]}')
    print(f'  best-fitness elite is row {best}, non-dominated: {best in result["non_dominated"]}')
    print(table.loc[best, ['fitness', *METRICS]].to_dict())
    edge = result['edge_cell_elites']
    print(f'  elites in clipped edge cells: {len(edge)} (rows {edge.tolist()})')
    print('  they would have no cell under exactly-equal constraints')


if __name__ == '__main__':
    main()
