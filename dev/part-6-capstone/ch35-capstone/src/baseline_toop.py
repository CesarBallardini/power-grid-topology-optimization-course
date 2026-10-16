"""The capstone's baseline: preprocess a grid once, run the DC stage three times, validate in AC, tabulate (Lab 1).

Run with ToOp's interpreter from this folder::

    ../../../../../ToOp/.venv/Scripts/python.exe baseline_toop.py [work folder] [grid.xiidm] [runs] [runtime seconds]

Everything is written under the work folder (a temporary folder by default); ToOp's own ``data/`` is only read.
The staged pipeline is the one of ``notebooks/example3_e2e_pipeline.ipynb``: ``run_preprocessing`` once, then
``run_dc_optimization_stage`` and ``perform_ac_analysis`` per run. Two Windows shims from Chapter 34 are applied
(the TensorBoard folder name and the per-epoch clock, see ``tensorboard_log_toop.py`` there); the grid is the
node-breaker example by default, ``data/case57_xiidm/grid.xiidm`` gives a richer repertoire for Exercises 1 and 4.
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd
import toop_engine_topology_optimizer.dc.main as dc_main_module
from omegaconf import DictConfig
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.ac.scoring_functions import evaluate_acceptance
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    perform_ac_analysis,
    prepare_importer_parameters,
    run_dc_optimization_stage,
    run_preprocessing,
)
from toop_engine_topology_optimizer.dc.main import CLIArgs
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef
from toop_engine_topology_optimizer.interfaces.messages.dc_params import BatchedMEParameters, LoadflowSolverParameters
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics

TOOP = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
NODE_BREAKER = TOOP / 'data' / 'grid_node_breaker' / 'grid.xiidm'
CASE57 = TOOP / 'data' / 'case57_xiidm' / 'grid.xiidm'
AC_CONFIG = {
    'n_processes': 1,
    'k_best_topos': 4,
    'critical_voltage_jump_percent': 5.0,
    'critical_va_diff_degree': 20.0,
}


class _DashedDatetime(datetime.datetime):
    def __str__(self):
        return super().__str__().replace(':', '-')


def windows_shim():
    """Colon-free TensorBoard folder names (ToOp ``ba1874e`` builds them from ``str(datetime.now())``)."""
    dc_main_module.datetime = types.SimpleNamespace(datetime=_DashedDatetime)


def preprocess(grid_xiidm, work_dir, action_set_clip=2**10, bb_outages=True):
    """Copy the grid into ``work_dir`` and run ToOp's import and preprocessing once; returns the data folder.

    ``bb_outages`` is ``preprocess_bb_outages``; it must be False for a bus-breaker model such as case57, where
    ``convert_rel_bb_outage_data`` fails on ``rel_bb_outage_zero_flow_br_indices = None``.
    """
    work_dir = Path(work_dir)
    grid_copy = work_dir / Path(grid_xiidm).name
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(grid_xiidm, grid_copy)
    data_folder = work_dir / 'grid'
    run_preprocessing(
        prepare_importer_parameters(grid_copy, data_folder),
        data_folder,
        PreprocessParameters(action_set_clip=action_set_clip, preprocess_bb_outages=bb_outages),
        is_pandapower_net=False,
    )
    return data_folder


def dc_config(
    data_folder, run_dir, seed, runtime_seconds, n_worst_contingencies, max_splits=3, iterations_per_epoch=500
):
    """The DC stage configuration: fixed seed and runtime, a repertoire cell per number of split substations."""
    run_dir = Path(run_dir)
    cfg = DictConfig(
        CLIArgs(
            ga_config=BatchedMEParameters(
                runtime_seconds=runtime_seconds,
                iterations_per_epoch=iterations_per_epoch,
                random_seed=seed,
                n_worst_contingencies=n_worst_contingencies,
                me_descriptors=(DescriptorDef(metric='split_subs', num_cells=max_splits + 1),),
                observed_metrics=('overload_energy_n_1', 'split_subs', 'switching_distance', 'max_flow_n_1'),
            ),
            lf_config=LoadflowSolverParameters(max_num_splits=max_splits, batch_size=8),
            tensorboard_dir=str(run_dir / 'tensorboard'),
            stats_dir=str(run_dir / 'stats'),
        ).model_dump()
    )
    cfg.update(
        {
            'fixed_files': [str(Path(data_folder) / 'static_information.hdf5')],
            'task_name': run_dir.name,
            'output_json': str(run_dir / 'output.json'),
            'num_cuda_devices': 0,
            'omp_num_threads': 4,
            'xla_force_host_platform_device_count': None,
            'double_precision': True,
        }
    )
    return cfg


def run_once(data_folder, run_dir, seed=1, runtime_seconds=10.0, n_worst_contingencies=2, k_best=4):
    """One DC run followed by the AC validation of its best topologies; the configuration is saved next to them."""
    windows_shim()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dc_config(data_folder, run_dir, seed, runtime_seconds, n_worst_contingencies)
    (run_dir / 'dc_config.json').write_text(json.dumps(dict(cfg), indent=2, default=str), encoding='utf-8')
    run_dc_optimization_stage(cfg, run_dir)
    ac_cfg = DictConfig({**AC_CONFIG, 'k_best_topos': k_best})
    (run_dir / 'ac_config.json').write_text(json.dumps(dict(ac_cfg), indent=2), encoding='utf-8')
    perform_ac_analysis(Path(data_folder), run_dir, ac_validation_cfg=ac_cfg, pandapower_runner=False)
    return run_dir


def accepted(run_dir):
    """Per validated topology: DC fitness, AC fitness and ToOp's acceptance verdict against the unsplit AC metrics."""
    run_dir = Path(run_dir)
    unsplit = Metrics.model_validate(json.loads((run_dir / 'unsplit_ac_metrics.json').read_text(encoding='utf-8')))
    rows = []
    for path in sorted(run_dir.glob('topology_*/ac_metrics.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        split = Metrics.model_validate(data)
        reason = evaluate_acceptance(metrics_split=split, metrics_unsplit=unsplit)
        rows.append(
            {
                'topology': path.parent.name,
                'actions': data['dc_info']['actions'],
                'dc_fitness': data['dc_info']['metrics']['fitness'],
                'ac_overload_energy_n_1': split.extra_scores['overload_energy_n_1'],
                'ac_non_converging': split.extra_scores.get('non_converging_loadflows'),
                'accepted': reason is None,
                'rejection': None if reason is None else reason.model_dump(exclude_none=True),
            }
        )
    return rows


def tabulate(work_dir):
    """One row per run: initial and best DC fitness, iterations, topologies validated and accepted."""
    rows = []
    for run_dir in sorted(Path(work_dir).glob('run_*')):
        res = json.loads((run_dir / 'res.json').read_text(encoding='utf-8'))
        verdicts = accepted(run_dir)
        rows.append(
            {
                'run': run_dir.name,
                'initial_fitness': res['initial_fitness'],
                'max_fitness': res['max_fitness'],
                'iterations': res['iteration'],
                'best_topos': len(res['best_topos']),
                'validated': len(verdicts),
                'accepted': sum(v['accepted'] for v in verdicts),
            }
        )
    return pd.DataFrame(rows)


def baseline(
    grid_xiidm=NODE_BREAKER,
    work_dir=None,
    runs=3,
    seed=1,
    runtime_seconds=10.0,
    n_worst_contingencies=2,
    bb_outages=True,
):
    """Preprocess once and run the DC + AC stages ``runs`` times with the same seed and runtime."""
    work_dir = Path(work_dir or tempfile.mkdtemp(prefix='capstone_'))
    data_folder = preprocess(grid_xiidm, work_dir, bb_outages=bb_outages)
    for i in range(runs):
        run_once(data_folder, work_dir / f'run_{i}', seed, runtime_seconds, n_worst_contingencies)
    return work_dir


def main():
    """Run the baseline (by default three times, 10 s each, on the node-breaker grid) and print the Lab 1 table."""
    work_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    grid = Path(sys.argv[2]) if len(sys.argv) > 2 else NODE_BREAKER
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    runtime_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    work_dir = baseline(grid, work_dir, runs=runs, runtime_seconds=runtime_seconds)
    print(f'work folder: {work_dir}')
    print(tabulate(work_dir).to_string(index=False))
    for run_dir in sorted(work_dir.glob('run_*')):
        for row in accepted(run_dir):
            print(run_dir.name, row)


if __name__ == '__main__':
    main()
