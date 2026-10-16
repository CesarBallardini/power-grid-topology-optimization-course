"""From a ToOp topology to a switching list: the ORAO summary of the node-breaker grid (Chapter 21, Exercise 5).

Run it with ToOp's interpreter (or inside its container) from this folder::

    python orao_switches_toop.py /tmp/orao

It imports and preprocesses ``data/grid_node_breaker/grid.xiidm`` as the Chapter 22 Lab does, hands every action of
the action set to ``perform_ac_analysis`` in place of the DC optimizer (``run_pipeline``'s DC stage names its
TensorBoard folder after ``datetime.now()``, which Windows rejects), reads the ``orao_summary.json`` files it
writes, joins the switch operations with pypowsybl's switch table and flags the disconnector operations that need
the bay breaker open first.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pypowsybl as pp
from omegaconf import DictConfig
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_interfaces.stored_action_set import ActionSet, load_action_set
from toop_engine_topology_optimizer.ac.scoring_functions import evaluate_acceptance
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    perform_ac_analysis,
    prepare_importer_parameters,
    run_preprocessing,
)
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics

TOOP_GRID = Path(__file__).resolve().parents[5] / 'ToOp' / 'data' / 'grid_node_breaker' / 'grid.xiidm'
AC_CONFIG = {'n_processes': 1, 'critical_voltage_jump_percent': 5.0, 'critical_va_diff_degree': 20.0}


def import_grid(work: Path, source: Path = TOOP_GRID) -> Path:
    """Copy the grid file into ``work``, import and preprocess it (cutoff 10 kV); return the processed grid folder."""
    work.mkdir(parents=True, exist_ok=True)
    grid_file = work / 'grid.xiidm'
    shutil.copy(source, grid_file)
    data_folder = work / 'grid'
    parameters = prepare_importer_parameters(grid_file, data_folder)
    parameters.area_settings.cutoff_voltage = 10
    run_preprocessing(parameters, data_folder, PreprocessParameters(), is_pandapower_net=False)
    return data_folder


def stored_action_set(data_folder: Path) -> ActionSet:
    """The action set written by the preprocessing, with its local actions expanded."""
    return load_action_set(data_folder / 'action_set.json', data_folder / 'action_set_diffs.hdf5')


def ac_validate_every_action(data_folder: Path, run_dir: Path) -> list[Path]:
    """Write a ``res.json`` listing every single action as a candidate and run ToOp's AC validation on all of them."""
    n_actions = len(stored_action_set(data_folder).local_actions)
    candidates = [{'actions': [a], 'disconnections': [], 'metrics': {'fitness': 0.0}} for a in range(n_actions)]
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'res.json').open('w', encoding='utf-8') as f:
        json.dump({'best_topos': candidates, 'initial_fitness': 0.0, 'initial_metrics': {}}, f)
    return perform_ac_analysis(data_folder, run_dir, DictConfig({**AC_CONFIG, 'k_best_topos': n_actions}))


def metrics(path: Path) -> Metrics:
    """The AC metrics stored in a topology folder (or the ``unsplit_ac_metrics.json`` of the run folder)."""
    with path.open(encoding='utf-8') as f:
        payload = json.load(f)
    return Metrics.model_validate({k: v for k, v in payload.items() if k != 'dc_info'})


def acceptance(run_dir: Path, topology_paths: list[Path]) -> pd.DataFrame:
    """Per assessed topology: its action, AC fitness, non-converging cases and ToOp's acceptance verdict."""
    unsplit = metrics(run_dir / 'unsplit_ac_metrics.json')
    rows = []
    for path in topology_paths:
        with (path / 'ac_metrics.json').open(encoding='utf-8') as f:
            action = json.load(f)['dc_info']['actions']
        split = metrics(path / 'ac_metrics.json')
        rejection = evaluate_acceptance(metrics_split=split, metrics_unsplit=unsplit)
        rows.append(
            {
                'topology': path.name,
                'action': action[0] if action else None,
                'fitness': split.fitness,
                'non_converging': split.extra_scores.get('non_converging_loadflows'),
                'verdict': 'accepted' if rejection is None else f'rejected: {rejection.criterion}',
            }
        )
    return pd.DataFrame(rows).set_index('topology')


def switch_table(topology_path: Path, net: pp.network.Network) -> pd.DataFrame:
    """The switch operations of one ORAO summary joined with the grid's switches (kind, voltage level, state now)."""
    with (topology_path / 'orao_summary.json').open(encoding='utf-8') as f:
        actions = json.load(f)['forced-actions']['preventive-actions-list']['actions']
    ops = pd.DataFrame([{'switch_id': a['switchId'], 'open': a['open'], 'orao_id': a['id']} for a in actions])
    if ops.empty:
        return ops
    switches = net.get_switches()[['kind', 'voltage_level_id', 'open']].rename(columns={'open': 'open_now'})
    return ops.join(switches, on='switch_id').set_index('switch_id')


def bay_breakers(action_set: ActionSet) -> dict[str, str]:
    """Disconnector id -> breaker id of the same bay, for every branch, injection and coupler bay of the action set."""
    out = {}
    for group in action_set.starting_bus_groups:
        for connection in group.branch_connections + group.injection_connections:
            bay = connection.asset_bay
            for disconnector in bay.busbar_disconnector_grid_model_id.values():
                out[disconnector] = bay.breaker_grid_model_id
        for coupler in group.couplers:
            bay = coupler.coupler_bay
            breaker = bay.coupler_breaker_ids[0] if bay.coupler_breaker_ids else coupler.grid_model_id
            for disconnector in list(bay.from_busbar_disconnector_ids.values()) + list(
                bay.to_busbar_disconnector_ids.values()
            ):
                out[disconnector] = breaker
    return out


def flag_disconnectors(table: pd.DataFrame, bays: dict[str, str]) -> pd.DataFrame:
    """Add the bay breaker of each disconnector and whether the list opens it: if not, the sequence must add it."""
    table = table.copy()
    table['bay_breaker'] = [bays.get(s) for s in table.index]
    opened = set(table.index[table.open])
    table['needs_breaker'] = [
        kind == 'DISCONNECTOR' and breaker not in opened
        for kind, breaker in zip(table.kind, table.bay_breaker, strict=True)
    ]
    return table


def main() -> None:
    """Import, AC-validate the ten single actions, print the verdicts and the switch table of the best split."""
    work = Path(sys.argv[1] if len(sys.argv) > 1 else 'orao')
    data_folder = import_grid(work)
    run_dir = work / 'run_single_actions'
    paths = ac_validate_every_action(data_folder, run_dir)
    verdicts = acceptance(run_dir, paths)
    unsplit = metrics(run_dir / 'unsplit_ac_metrics.json')
    non_converging = unsplit.extra_scores.get('non_converging_loadflows')
    print(f'unsplit AC fitness {unsplit.fitness:.2f}, non-converging load flows {non_converging}')
    print(verdicts.round(2).to_string())
    best = verdicts.loc[verdicts.action > 0, 'fitness'].idxmin()
    print(f'best split: {best}, action {verdicts.loc[best, "action"]}, fitness {verdicts.loc[best, "fitness"]:.2f}')
    net = pp.network.load(str(data_folder / 'grid.xiidm'))
    table = flag_disconnectors(switch_table(run_dir / best, net), bay_breakers(stored_action_set(data_folder)))
    print(table.to_string())
    print(f'{int(table.needs_breaker.sum())} disconnector operations need their bay breaker opened first')


if __name__ == '__main__':
    main()
