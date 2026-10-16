"""The operator hand-off of a ToOp run: ORAO summaries as switching instructions (Chapter 28, Lab B4).

``orao_summary.json`` (``changing_switches_to_orao_dict`` in ``optimizer/ac/summary.py``) lists, per validated
topology, the switches whose state changes; ``switches.json`` (written by ``lab_b_toop.py`` from the grid) adds the
kind and voltage level of each. ``instructions`` turns them into lines an operator can read, and ``counts`` compares
the number of operations with the AC and DC ``switching_distance``; ``run_summary`` and ``acceptance.run_table`` give
the numbers of Lab B1. It reads the runs stored in ``lab_b_run/``::

    python handoff.py [run folder]
"""

import json
import sys
from pathlib import Path

import pandas as pd
from acceptance import run_table

RUNS = Path(__file__).resolve().parent / 'lab_b_run'


def load(path: Path) -> dict:
    """A JSON file."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def topology_dirs(run_dir: Path) -> list[Path]:
    """The ``topology_N`` folders of a run, in numeric order."""
    return sorted(Path(run_dir).glob('topology_*'), key=lambda p: int(p.name.split('_')[1]))


def operations(orao_summary: dict) -> list[tuple[str, bool]]:
    """``[(switch id, open?), ...]`` in the order of the summary (only ``SWITCH`` entries exist in ToOp ba1874e)."""
    actions = orao_summary['forced-actions']['preventive-actions-list']['actions']
    return [(a['switchId'], a['open']) for a in actions if a['type'] == 'SWITCH']


def instructions(orao_summary: dict, switches: dict) -> list[str]:
    """One line per operation: verb, kind of switch, its id and voltage level, and what the kind allows."""
    lines = []
    for switch, is_open in operations(orao_summary):
        info = switches[switch]
        kind = info['kind'].lower().replace('_', ' ')
        note = 'can break load current' if info['kind'] == 'BREAKER' else 'operate only without current'
        lines.append(f'{"Open" if is_open else "Close"} {kind} {switch} in {info["voltage_level_id"]} ({note})')
    return lines


def counts(run_dir: Path) -> pd.DataFrame:
    """Per validated topology: operations in the ORAO summary against the AC and DC switching distances."""
    rows = []
    for folder in topology_dirs(run_dir):
        ac = load(folder / 'ac_metrics.json')
        ops = operations(load(folder / 'orao_summary.json'))
        switches = load(folder / 'switches.json')
        rows.append(
            {
                'topology': folder.name,
                'actions': ac['dc_info']['actions'],
                'operations': len(ops),
                'breakers': sum(switches[s]['kind'] == 'BREAKER' for s, _ in ops),
                'disconnectors': sum(switches[s]['kind'] == 'DISCONNECTOR' for s, _ in ops),
                'ac_switching_distance': ac['extra_scores'].get('switching_distance'),
                'dc_switching_distance': ac['dc_info']['metrics']['extra_scores'].get('switching_distance'),
            }
        )
    return pd.DataFrame(rows)


def run_summary(run_dir: Path) -> dict:
    """The DC side of a run (epochs, initial fitness, elites with their PST setpoints) and the unsplit AC metrics."""
    res = load(Path(run_dir) / 'res.json')
    unsplit = load(Path(run_dir) / 'unsplit_ac_metrics.json')['extra_scores']
    keys = ('overload_energy_n_1', 'critical_branch_count_n_1', 'non_converging_loadflows')
    return {
        'epochs': res['iteration'],
        'initial_fitness': res['initial_fitness'],
        'elites': [(t['actions'], t['pst_setpoints'], t['metrics']['fitness']) for t in res['best_topos']],
        'unsplit_ac': {k: unsplit[k] for k in keys},
    }


def main() -> None:
    """Print the Lab B1 table, the instructions and the counts of every stored run (or of the run folder given)."""
    pd.set_option('display.width', 200)
    runs = [Path(sys.argv[1])] if len(sys.argv) > 1 else sorted(p for p in RUNS.iterdir() if p.is_dir())
    for run_dir in runs:
        print(run_dir.name, run_summary(run_dir))
        print(run_table(run_dir).to_string(index=False))
        for folder in topology_dirs(run_dir):
            print(f'  {folder.name}:')
            for line in instructions(load(folder / 'orao_summary.json'), load(folder / 'switches.json')):
                print(f'    {line}')
        print(counts(run_dir).to_string(index=False))


if __name__ == '__main__':
    main()
