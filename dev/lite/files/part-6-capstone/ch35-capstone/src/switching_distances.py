"""DC against AC switching distance for the validated topologies of a baseline run (Chapter 35, Exercise 4).

The DC value is ``extra_scores.switching_distance`` of ``res.json`` (``get_switching_distance`` in
``dc_solver/jax/aggregate_results.py``: the precomputed ``reassignment_distance`` of each applied branch action,
summed). The AC value is ``extra_scores.switching_distance`` of ``ac_metrics.json`` (``extract_switching_distance``
in ``optimizer/ac/scoring_functions.py``: the number of branch and injection reassignments of the realized
topology). ``python switching_distances.py`` prints the pairs of both baselines and describes the plot.
"""

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BASELINE = Path(__file__).resolve().parent / 'baseline'


def pairs(run_dir):
    """One row per ``topology_*`` folder: actions, DC and AC switching distance, DC and AC overload energy."""
    run_dir = Path(run_dir)
    res = json.loads((run_dir / 'res.json').read_text(encoding='utf-8'))
    by_actions = {tuple(t['actions']): t['metrics']['extra_scores'] for t in res['best_topos']}
    rows = []
    for path in sorted(run_dir.glob('topology_*/ac_metrics.json')):
        ac = json.loads(path.read_text(encoding='utf-8'))
        actions = tuple(ac['dc_info']['actions'])
        dc_scores = by_actions.get(actions, ac['dc_info']['metrics']['extra_scores'])
        rows.append(
            {
                'run': run_dir.name,
                'topology': path.parent.name,
                'actions': list(actions),
                'dc_switching_distance': dc_scores.get('switching_distance'),
                'ac_switching_distance': ac['extra_scores'].get('switching_distance'),
                'dc_overload_energy_n_1': dc_scores['overload_energy_n_1'],
                'ac_overload_energy_n_1': ac['extra_scores']['overload_energy_n_1'],
            }
        )
    return pd.DataFrame(rows)


def all_pairs(root=BASELINE):
    """The pairs of every run under ``baseline/<grid>/run_*``."""
    frames = [pairs(run_dir).assign(grid=run_dir.parent.name) for run_dir in sorted(root.glob('*/run_*'))]
    return pd.concat(frames, ignore_index=True)


def plot(table, ax=None):
    """Scatter of AC against DC switching distance with the diagonal; returns the axes."""
    ax = ax or plt.figure(figsize=(4, 4)).add_subplot()
    for grid, part in table.groupby('grid'):
        ax.scatter(part['dc_switching_distance'], part['ac_switching_distance'], label=grid)
    top = max(table['dc_switching_distance'].max(), table['ac_switching_distance'].max(), 1)
    ax.plot([0, top], [0, top], 'k--', linewidth=0.8, label='equal')
    ax.set_xlabel('DC switching distance (reassignment_distance of the actions)')
    ax.set_ylabel('AC switching distance (reassignments of the realized topology)')
    ax.legend()
    return ax


def main():
    """Print the table and the facts the figure would show."""
    table = all_pairs(Path(sys.argv[1]) if len(sys.argv) > 1 else BASELINE)
    pd.set_option('display.width', 200)
    print(table.to_string(index=False))
    diff = table['ac_switching_distance'] - table['dc_switching_distance']
    print(
        f'\n{len(table)} topologies; AC minus DC switching distance: min {diff.min()}, max {diff.max()}, '
        f'equal in {(diff == 0).sum()} of them'
    )
    plot(table)
    print('figure: one point per topology, DC on x, AC on y, with the diagonal for reference')


if __name__ == '__main__':
    main()
