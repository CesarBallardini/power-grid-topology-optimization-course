"""Read a ToOp ``res.json``: the ``best_topos`` table, the fitness check and the operator's pick (Chapter 35, Ex. 1).

The files under ``baseline/`` were written by ``baseline_toop.py`` with ToOp ``ba1874e`` (node-breaker grid, three
runs with the same seed; the complex example grid, one run); ``python res_table.py [res.json]`` prints the tables.
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASELINE = Path(__file__).resolve().parent / 'baseline'
NODE_BREAKER = BASELINE / 'node_breaker' / 'run_0' / 'res.json'
COMPLEX = BASELINE / 'complex_grid' / 'run_0' / 'res.json'


def load_res(path):
    """The parsed ``res.json``."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def best_topos_table(res):
    """One row per entry of ``best_topos``: actions, disconnections, fitness and every ``extra_scores`` metric."""
    rows = []
    for topo in res['best_topos']:
        row = {
            'actions': topo['actions'],
            'disconnections': topo['disconnections'],
            'fitness': topo['metrics']['fitness'],
            **topo['metrics']['extra_scores'],
            'worst_k_contingency_cases': topo['metrics'].get('worst_k_contingency_cases'),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def target_metrics(res):
    """``{metric: weight}`` of the run, from ``args.ga_config.target_metrics``."""
    return {metric: float(weight) for metric, weight in res['args']['ga_config']['target_metrics']}


def weighted_sum_check(res, atol=1e-9):
    """Recompute ``fitness`` as minus the weighted sum of the target metrics and compare it row by row."""
    table = best_topos_table(res)
    weights = target_metrics(res)
    table['recomputed'] = [-sum(w * row[m] for m, w in weights.items()) for _, row in table.iterrows()]
    table['matches'] = (table['fitness'] - table['recomputed']).abs() < atol
    return table


def for_operator(table, max_splits=1):
    """The best (largest fitness) topology with at most ``max_splits`` split substations, or None."""
    eligible = table[table['split_subs'] <= max_splits]
    return None if eligible.empty else eligible.sort_values('fitness', ascending=False).iloc[0]


def main():
    """Print the table, the check and the operator's pick for the file given (default: both baselines)."""
    paths = [Path(sys.argv[1])] if len(sys.argv) > 1 else [NODE_BREAKER, COMPLEX]
    pd.set_option('display.width', 200)
    for path in paths:
        res = load_res(path)
        table = weighted_sum_check(res)
        print(
            f'{path.parent.parent.name}/{path.parent.name}: target metrics {target_metrics(res)}, '
            f'initial fitness {res["initial_fitness"]:.3f}, {len(table)} best_topos'
        )
        print(table.drop(columns=['worst_k_contingency_cases']).to_string(index=False))
        print('fitness equals the negated weighted sum in every row:', bool(table['matches'].all()))
        pick = for_operator(table, 1)
        print(
            'at most one split substation ->',
            None if pick is None else f'actions {pick["actions"]}, fitness {pick["fitness"]:.3f}',
        )
        print()


if __name__ == '__main__':
    main()
