"""ToOp's AC acceptance rule, ported, and the table of a validation run (Chapter 28, Lab B and Exercise 5).

``evaluate_acceptance`` follows ``optimizer/ac/scoring_functions.py`` (commit ba1874e) check by check;
``test_acceptance_toop.py`` compares it with the original under ToOp's interpreter. ``run_table`` reads the
``ac_metrics.json`` of every ``topology_N/`` folder and the run's ``unsplit_ac_metrics.json``, both written by
``save_ac_metrics_summary`` as ``Metrics.model_dump()`` plus ``dc_info``. Run it to print the Lab B2 traps and
Exercise 5::

    python acceptance.py [run_folder]
"""

import json
import sys
from pathlib import Path

import pandas as pd

THRESHOLDS = {
    'convergence': 1.0,
    'overload-energy': 0.95,
    'critical-branch-count': 1.1,
    'voltage-magnitude': 1.1,
    'voltage-angle': 1.1,
}


# --8<-- [start:rule]
def evaluate_acceptance(
    split: dict, unsplit: dict, critical_voltage: bool = False, thresholds: dict = THRESHOLDS
) -> dict | None:
    """``None`` when the candidate is accepted, else the first failed check with its values and threshold.

    ``split`` and ``unsplit`` are ``extra_scores`` dicts. The checks run in ToOp's order: non-converging loadflows
    (minus the candidate's disconnected branches), N-1 overload energy, critical branch count and, only with
    ``critical_voltage``, voltage jumps and angle differences. Every threshold multiplies the unsplit value, and
    a metric missing from ``split`` or ``unsplit`` takes ToOp's defaults (99999 for the split overload, 999 for
    the unsplit counts, 0 otherwise).
    """
    checks = [
        (
            'convergence',
            split.get('non_converging_loadflows', 0) - split.get('disconnected_branches', 0),
            unsplit.get('non_converging_loadflows', 0),
        ),
        ('overload-energy', split.get('overload_energy_n_1', 99999), unsplit.get('overload_energy_n_1', 0)),
        (
            'critical-branch-count',
            split.get('critical_branch_count_n_1', 0),
            unsplit.get('critical_branch_count_n_1', 999),
        ),
    ]
    if critical_voltage:
        checks += [
            ('voltage-magnitude', split.get('voltage_jump_count_n_1', 0), unsplit.get('voltage_jump_count_n_1', 999)),
            (
                'voltage-angle',
                split.get('critical_va_diff_count_n_1', 0),
                unsplit.get('critical_va_diff_count_n_1', 999),
            ),
        ]
    for criterion, after, before in checks:
        if not after <= before * thresholds[criterion]:
            return {
                'criterion': criterion,
                'value_after': float(after),
                'value_before': float(before),
                'threshold': thresholds[criterion],
            }
    return None


# --8<-- [end:rule]
# --8<-- [start:table]
METRICS = ['non_converging_loadflows', 'disconnected_branches', 'overload_energy_n_1', 'critical_branch_count_n_1']


def run_table(run: Path) -> pd.DataFrame:
    """Lab B1: one row per ``topology_N/ac_metrics.json`` with the metrics, every check by hand and the verdict."""
    unsplit = json.loads((run / 'unsplit_ac_metrics.json').read_text(encoding='utf-8'))['extra_scores']
    rows = []
    for folder in sorted(run.glob('topology_*'), key=lambda p: int(p.name.split('_')[1])):
        res = json.loads((folder / 'ac_metrics.json').read_text(encoding='utf-8'))
        s = res['extra_scores']
        row = {'topology': folder.name, 'ac_fitness': res['fitness'], **{m: s.get(m, 0) for m in METRICS}}
        row['dc_fitness'] = res.get('dc_info', {}).get('metrics', {}).get('fitness')
        row['convergence_ok'] = s.get('non_converging_loadflows', 0) - s.get(
            'disconnected_branches', 0
        ) <= unsplit.get('non_converging_loadflows', 0)
        row['overload_ok'] = s.get('overload_energy_n_1', 99999) <= 0.95 * unsplit.get('overload_energy_n_1', 0)
        row['critical_ok'] = s.get('critical_branch_count_n_1', 0) <= 1.1 * unsplit.get(
            'critical_branch_count_n_1', 999
        )
        verdict = evaluate_acceptance(s, unsplit)
        row['rejected_by'] = None if verdict is None else verdict['criterion']
        rows.append(row)
    table = pd.DataFrame(rows)
    table['dc_rank'] = (
        table['dc_fitness'].rank(ascending=False, method='first') if table['dc_fitness'].notna().any() else None
    )
    return table


# --8<-- [end:table]
def main() -> None:
    """The traps of Lab B2, Exercise 5 and, if a run folder is given, its table."""
    print(
        'zero-baseline trap:',
        evaluate_acceptance(
            {'overload_energy_n_1': 0.01, 'critical_branch_count_n_1': 0},
            {'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 0},
        ),
    )
    print(
        'same candidate, unsplit 1.0 MW:',
        evaluate_acceptance(
            {'overload_energy_n_1': 0.01, 'critical_branch_count_n_1': 0},
            {'overload_energy_n_1': 1.0, 'critical_branch_count_n_1': 0},
        ),
    )
    print(
        'convergence before overload:',
        evaluate_acceptance(
            {'non_converging_loadflows': 0, 'overload_energy_n_1': 5.0, 'critical_branch_count_n_1': 1},
            {'non_converging_loadflows': 1, 'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 0},
        ),
        '(the converging candidate reveals 5 MW the unsplit run never computed)',
    )
    print(
        'more non-converging cases, less overload:',
        evaluate_acceptance(
            {'non_converging_loadflows': 2, 'overload_energy_n_1': 1.0, 'critical_branch_count_n_1': 1},
            {'non_converging_loadflows': 1, 'overload_energy_n_1': 100.0, 'critical_branch_count_n_1': 5},
        ),
    )
    print(
        'Exercise 5:',
        evaluate_acceptance(
            {'non_converging_loadflows': 1, 'overload_energy_n_1': 0.4, 'critical_branch_count_n_1': 1},
            {'non_converging_loadflows': 2, 'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 0},
        ),
    )
    if len(sys.argv) > 1:
        pd.set_option('display.width', 200)
        print(run_table(Path(sys.argv[1])))


if __name__ == '__main__':
    main()
