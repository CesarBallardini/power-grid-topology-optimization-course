"""Run ToOp's small-grid example (``notebooks/example2_small_grid_toop.ipynb``) with a configurable repertoire.

Run it with ToOp's interpreter (its container or virtual environment) from this folder::

    python small_grid_toop.py OUT_ROOT NAME [--descriptors split_subs:4,switching_distance:20] [--plot]
                              [--weights overload_energy_n_1:1,critical_branch_count_n_1:10] [--seed 42]
                              [--runtime 30]

It copies the example grid ``data/grid_node_breaker/grid.xiidm`` of the ToOp checkout (``$TOOP_REPO``, default
the ``ToOp`` folder next to this repository) into ``OUT_ROOT/NAME``, runs preprocessing and the DC MAP-Elites stage
with the notebook's settings except for the overrides given, and prints where ``res.json``, the per-epoch
``res_<epoch>.json`` files and the repertoire plots are. The AC validation stage is skipped: the lab reads the DC
repertoire only.
"""

import argparse
import datetime
import json
import os
import shutil
import types
from pathlib import Path

import toop_engine_topology_optimizer.dc.main as dc_main
from omegaconf import DictConfig
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    PipelineConfig,
    get_paths,
    prepare_importer_parameters,
    run_pipeline,
    suppress_jax_logs,
)

TOOP_REPO = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[5] / 'ToOp'))
GRID = TOOP_REPO / 'data' / 'grid_node_breaker' / 'grid.xiidm'


class _FileSafeDatetime:
    """``now()`` without colons: ``optimizer/dc/main.py`` names the TensorBoard folder ``str(datetime.now())``."""

    @staticmethod
    def now() -> str:
        return datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')  # noqa: DTZ005


if os.name == 'nt':
    # Windows rejects ':' in folder names, so the unpatched call fails with WinError 123 (ToOp ba1874e).
    dc_main.datetime = types.SimpleNamespace(datetime=_FileSafeDatetime)


def parse_pairs(text: str, cast=int) -> list[tuple[str, float]]:
    """``'a:1,b:20'`` into ``[('a', 1), ('b', 20)]``."""
    return [(name, cast(value)) for name, value in (item.split(':') for item in text.split(',') if item)]


def ga_config(descriptors: str, weights: str, seed: int, runtime: float, plot: bool) -> dict:
    """The notebook's ``ga_config`` with the descriptors, target weights, seed, runtime and plot flag replaced."""
    me_descriptors = [{'metric': m, 'num_cells': c} for m, c in parse_pairs(descriptors)]
    target_metrics = [[m, w] for m, w in parse_pairs(weights, float)]
    observed = ['overload_energy_n_1', 'split_subs']
    for metric in [d['metric'] for d in me_descriptors] + [m for m, _ in target_metrics]:
        if metric not in observed:
            observed.append(metric)
    return {
        'runtime_seconds': runtime,
        'random_seed': seed,
        'me_descriptors': me_descriptors,
        'target_metrics': target_metrics,
        'observed_metrics': observed,
        'n_worst_contingencies': 2,
        'plot': plot,
    }


def run(out_root: Path, name: str, config: dict) -> Path:
    """Copy the grid, run preprocessing and the DC stage; return the folder holding ``res.json``."""
    iteration = out_root / name
    iteration.mkdir(parents=True, exist_ok=True)
    shutil.copy(GRID, iteration / 'grid.xiidm')
    pipeline_cfg = PipelineConfig(root_path=out_root, iteration_name=name, file_name='grid.xiidm', grid_type='powsybl')
    iteration_path, file_path, data_folder, snapshot_dir = get_paths(pipeline_cfg)
    stats_dir = str(iteration_path / 'results' / '{task_name}')
    dc_optimization_cfg = DictConfig(
        {
            'task_name': 'test',
            'fixed_files': [str(data_folder / pipeline_cfg.static_info_relpath)],
            'double_precision': None,
            'tensorboard_dir': stats_dir,
            'stats_dir': stats_dir,
            'summary_frequency': None,
            'checkpoint_frequency': None,
            'stdout': None,
            'double_limits': None,
            'num_cuda_devices': 1,
            'omp_num_threads': 1,
            'xla_force_host_platform_device_count': None,
            'output_json': str(iteration_path / 'results' / 'output.json'),
            'lf_config': {'distributed': False},
            'ga_config': config,
        }
    )
    importer_parameters = prepare_importer_parameters(file_path, data_folder)
    importer_parameters.area_settings.cutoff_voltage = 10
    # The notebook passes enable_bb_outage and bb_outage_as_nminus1 here; Pydantic drops them silently (see the
    # chapter's warning), so they are left out.
    preprocessing_parameters = PreprocessParameters(action_set_clip=2**10)
    run_pipeline(
        pipeline_cfg=pipeline_cfg,
        importer_parameters=importer_parameters,
        preprocessing_parameters=preprocessing_parameters,
        dc_optim_config=dc_optimization_cfg,
        ac_validation_cfg=DictConfig({'n_processes': 1, 'k_best_topos': 5}),
        run_ac_validation_stage=False,
    )
    runs = sorted(snapshot_dir.glob('run_*'), key=lambda p: int(p.name.split('_')[1]))
    return runs[-1]


def summary(run_dir: Path) -> dict:
    """Headline numbers of a run: fitness before and after, epochs, evaluations and number of reported elites."""
    res = json.loads((run_dir / 'res.json').read_text(encoding='utf-8'))
    return {
        'initial_fitness': res['initial_fitness'],
        'max_fitness': res['max_fitness'],
        'epochs': res['iteration'],
        'total_branch_combis': res['total_branch_combis'],
        'reported_elites': len(res['best_topos']),
        'descriptors': [(d['metric'], d['num_cells']) for d in res['args']['ga_config']['me_descriptors']],
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out_root', type=Path)
    parser.add_argument('name')
    parser.add_argument('--descriptors', default='split_subs:4')
    parser.add_argument('--weights', default='overload_energy_n_1:1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runtime', type=float, default=30.0)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    suppress_jax_logs()
    run_dir = run(
        args.out_root, args.name, ga_config(args.descriptors, args.weights, args.seed, args.runtime, args.plot)
    )
    print(json.dumps({'run_dir': str(run_dir), **summary(run_dir)}, indent=2))


if __name__ == '__main__':
    main()
