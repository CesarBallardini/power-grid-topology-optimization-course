"""A first operator brief from the files a ToOp run leaves behind (Chapter 35, Lab step 4 and Exercise 3).

``situation`` reads ``unsplit_ac_metrics.json``, ``proposals`` the ``topology_*/ac_metrics.json`` files and
``switching_steps`` turns ``orao_summary.json`` into ordered operations in operator language, with a feasibility
note per step (breakers switch under load, disconnectors do not). ``python brief.py [run folder]`` prints the brief
of the node-breaker baseline.
"""

import json
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / 'baseline'
NODE_BREAKER_RUN = BASELINE / 'node_breaker' / 'run_0'


def load(path):
    """A JSON file."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def situation(unsplit_ac_metrics):
    """Base case and worst N-1 figures of the grid before any action, from the unsplit AC metrics."""
    s = unsplit_ac_metrics['extra_scores']
    return {
        'worst N-0 loading': s['max_flow_n_0'],
        'worst N-1 loading': s['max_flow_n_1'],
        'N-1 overload energy (MW)': s['overload_energy_n_1'],
        'critical branches N-1': s['critical_branch_count_n_1'],
        'non-converging contingencies': s['non_converging_loadflows'],
        'DC fitness of the unsplit grid': unsplit_ac_metrics['dc_info']['fitness'],
    }


def proposals(run_dir):
    """One entry per validated topology with the figures the brief quotes."""
    out = []
    for path in sorted(Path(run_dir).glob('topology_*/ac_metrics.json')):
        ac = load(path)
        s = ac['extra_scores']
        out.append(
            {
                'topology': path.parent.name,
                'actions': ac['dc_info']['actions'],
                'disconnections': ac['dc_info']['disconnections'],
                'split substations': s['split_subs'],
                'DC overload energy removed (MW)': None,
                'AC N-1 overload energy (MW)': s['overload_energy_n_1'],
                'AC critical branches N-1': s['critical_branch_count_n_1'],
                'non-converging contingencies': s['non_converging_loadflows'],
                'largest voltage-angle difference N-1 (deg)': s['max_va_diff_n_1'],
                'critical voltage-angle differences': s['critical_va_diff_count_n_1'],
            }
        )
    return out


def switch_kind(switch_id):
    """Breaker or disconnector, read from ToOp's switch ids (``*_BREAKER*``, ``*_DISCONNECTOR_*``)."""
    upper = switch_id.upper()
    if 'BREAKER' in upper:
        return 'breaker'
    if 'DISCONNECTOR' in upper:
        return 'disconnector'
    return 'switch'


def switching_steps(orao_summary):
    """Ordered operations of an ORAO summary as sentences with a feasibility note each."""
    actions = orao_summary['forced-actions']['preventive-actions-list']['actions']
    steps = []
    for n, action in enumerate(actions, 1):
        if action['type'] != 'SWITCH':
            steps.append(f'{n}. {action["type"]} {action.get("id", "")}: not a switch operation, check by hand')
            continue
        verb = 'Open' if action['open'] else 'Close'
        kind = switch_kind(action['switchId'])
        if kind == 'disconnector':
            note = 'disconnector: no load current allowed, the breaker in series must be open first'
        elif verb == 'Close':
            note = 'breaker: closing joins two busbars, synchro-check and a small angle difference required'
        else:
            note = 'breaker: can interrupt load current'
        steps.append(f'{n}. {verb} {kind} {action["switchId"]} ({note})')
    return steps


def brief(run_dir=NODE_BREAKER_RUN):
    """The brief as Markdown text: situation, proposals, switching steps, assumptions."""
    run_dir = Path(run_dir)
    lines = ['# Operator brief (draft)', '', '## Situation (unsplit grid, AC N-1)']
    lines += [f'- {k}: {v}' for k, v in situation(load(run_dir / 'unsplit_ac_metrics.json')).items()]
    unsplit_energy = load(run_dir / 'unsplit_ac_metrics.json')['extra_scores']['overload_energy_n_1']
    for prop in proposals(run_dir):
        lines += [
            '',
            f'## Proposal {prop["topology"]}: actions {prop["actions"]}, disconnections {prop["disconnections"]}',
        ]
        prop['DC overload energy removed (MW)'] = None
        lines += [
            f'- {k}: {v}'
            for k, v in prop.items()
            if k not in ('topology', 'actions', 'disconnections') and v is not None
        ]
        ratio = prop['AC N-1 overload energy (MW)'] / unsplit_energy
        lines.append(f'- AC N-1 overload energy relative to the unsplit grid: {ratio:.2f}')
        lines += ['', '### Switching steps']
        lines += switching_steps(load(run_dir / prop['topology'] / 'orao_summary.json'))
    lines += [
        '',
        '## Assumptions and risks',
        "- DC optimization, AC validation with ToOp's N-1 definition; contingencies outside it are not studied.",
        '- Load flow convergence failures are counted, not explained.',
        '- Diagrams: before/after single-line diagrams of the split station with the switches above highlighted.',
    ]
    return '\n'.join(lines)


def main():
    """Print the brief of the run folder given (default: the node-breaker baseline, run 0)."""
    print(brief(Path(sys.argv[1]) if len(sys.argv) > 1 else NODE_BREAKER_RUN))


if __name__ == '__main__':
    main()
