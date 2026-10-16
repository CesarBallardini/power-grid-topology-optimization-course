"""Read ToOp's ``res.json``: the elites as a table, the repertoire as a heatmap, coverage and QD-score (Chapter 25).

``res.json`` is written by ``summarize`` (``optimizer/dc/genetic_functions/scoring_functions.py``). Its
``best_topos`` list holds one entry per cell elite whose fitness beats the unsplit grid, with the descriptors in
``metrics.extra_scores``. ``res_two_descriptors.json`` next to this file is the trimmed output of the lab's step 2
(``small_grid_toop.py ... --descriptors split_subs:4,switching_distance:20 --plot``, ToOp ba1874e on CPU).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE / 'res_two_descriptors.json'
KEEP_TOP = ('initial_fitness', 'initial_metrics', 'max_fitness', 'iteration', 'total_branch_combis')
KEEP_GA = ('me_descriptors', 'target_metrics', 'runtime_seconds', 'random_seed', 'plot')


def load(path: Path = EXAMPLE) -> dict:
    """The parsed ``res.json``."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def trim(res: dict) -> dict:
    """Only what the chapter reads: headline numbers, the descriptor configuration and each elite's metrics."""
    return {
        **{k: res[k] for k in KEEP_TOP},
        'args': {'ga_config': {k: res['args']['ga_config'][k] for k in KEEP_GA}},
        'best_topos': [
            {
                'actions': t['actions'],
                'disconnections': t['disconnections'],
                'metrics': {'fitness': t['metrics']['fitness'], 'extra_scores': t['metrics']['extra_scores']},
            }
            for t in res['best_topos']
        ],
    }


def descriptors(res: dict) -> list[tuple[str, int]]:
    """``(metric, num_cells)`` of each descriptor of the run."""
    return [(d['metric'], d['num_cells']) for d in res['args']['ga_config']['me_descriptors']]


def elites(res: dict) -> pd.DataFrame:
    """One row per reported elite: its descriptor values, fitness, every observed metric and the actions."""
    rows = []
    for t in res['best_topos']:
        rows.append({**t['metrics']['extra_scores'], 'fitness': t['metrics']['fitness'], 'actions': t['actions']})
    return pd.DataFrame(rows)


def grid(res: dict) -> np.ndarray:
    """Fitness per cell (rows: first descriptor, columns: second), NaN where no elite is reported; clipped indices."""
    desc = descriptors(res)
    dims = tuple(n for _, n in desc)
    table = np.full(dims, np.nan)
    for t in res['best_topos']:
        scores = t['metrics']['extra_scores']
        cell = tuple(min(int(scores[m]), n - 1) for m, n in desc)
        table[cell] = np.nanmax([table[cell], t['metrics']['fitness']])
    return table


def coverage(table: np.ndarray) -> float:
    """Reported (filled) cells over all cells."""
    return float(np.isfinite(table).mean())


def qd_score(table: np.ndarray, initial_fitness: float) -> float:
    """Sum over reported cells of the fitness gain over the unsplit grid."""
    return float(np.nansum(table - initial_fitness))


def heatmap(table: np.ndarray, desc: list[tuple[str, int]], ax=None):
    """Heatmap of the fitness per cell, empty cells left blank (unlike ToOp's plot, which fills them)."""
    ax = ax or plt.gca()
    image = ax.imshow(np.ma.masked_invalid(table), origin='lower', aspect='auto', cmap='viridis')
    ax.set_ylabel(desc[0][0])
    ax.set_xlabel(desc[1][0])
    ax.set_yticks(range(desc[0][1]))
    plt.colorbar(image, ax=ax, label='fitness')
    return image


def cells_with_value(dims: tuple[int, ...], axis: int, value: int) -> list[int]:
    """Flat indices of every cell whose descriptor ``axis`` equals ``value`` (Exercise 2)."""
    return [
        int(np.ravel_multi_index(index, dims))
        for index in np.ndindex(*dims)
        if index[axis] == min(value, dims[axis] - 1)
    ]


def clipped_cells(dims: tuple[int, ...]) -> list[int]:
    """Flat indices of the cells on the last row or column of any axis, which hold "at least" values (Exercise 2)."""
    return [
        int(np.ravel_multi_index(index, dims))
        for index in np.ndindex(*dims)
        if any(i == n - 1 for i, n in zip(index, dims, strict=True))
    ]


def main() -> None:
    """Lab step 3 on the committed run, Exercises 2 and 5; ``--trim IN OUT`` rewrites a full res.json."""
    if len(sys.argv) == 4 and sys.argv[1] == '--trim':
        Path(sys.argv[3]).write_text(json.dumps(trim(load(Path(sys.argv[2]))), indent=1), encoding='utf-8')
        return
    res = load()
    desc = descriptors(res)
    print('descriptors', desc, 'epochs', res['iteration'], 'branch combinations', res['total_branch_combis'])
    print(f'initial fitness {res["initial_fitness"]:.3f}, max fitness {res["max_fitness"]:.3f}')
    print(elites(res).to_string())
    table = grid(res)
    print(
        f'reported cells {int(np.isfinite(table).sum())} of {table.size}: coverage {coverage(table):.4f}, '
        f'QD-score {qd_score(table, res["initial_fitness"]):.3f}'
    )
    dims = (5, 45)
    two = cells_with_value(dims, 0, 2)
    print(f'cells with exactly 2 split substations: {two[0]} to {two[-1]} ({len(two)} cells)')
    clipped = clipped_cells(dims)
    print(
        f'{len(clipped)} clipped cells: last row {clipped_cells(dims)[-45:][:3]}... and last column '
        f'{[c for c in clipped if c % 45 == 44]}'
    )
    three = (5, 45, 3)
    print(f'{three}: {int(np.prod(three))} cells, {int(np.prod(three)) * 8} bytes of float64 fitness')


if __name__ == '__main__':
    main()
