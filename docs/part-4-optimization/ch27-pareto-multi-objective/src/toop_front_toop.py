"""ToOp DC runs on the complex example grid for Chapter 27 (Labs 1, 3, 4 and 7); needs ToOp's interpreter.

Run it in the course's ToOp container (or ToOp's virtual environment) from this folder::

    python toop_front_toop.py WORK_DIR [--gpu] [EXPERIMENT ...]

``WORK_DIR/grid`` receives the preprocessed ``data/complex_grid/grid.xiidm`` of the ToOp checkout (``$TOOP_REPO``,
default the ``ToOp`` folder next to this repository); the preprocessing runs once and is reused. Each experiment
of ``EXPERIMENTS`` (all by default) is one DC MAP-Elites run whose ``res.json`` lands in ``WORK_DIR/<name>/``:

- ``fine_s1`` … ``fine_s5``: ToOp's default repertoire, ``split_subs`` × ``switching_distance`` with 5 × 45 cells,
  fitness ``-overload_energy_n_1``, seeds 1 to 5 (Labs 1, 4 and 7);
- ``coarse_s1`` … ``coarse_s5``: the same with 5 × 9 cells (Lab 4);
- ``w1``, ``w3``, ``w10``, ``w30``, ``w100``: seed 1, fitness ``-(overload_energy_n_1 + w critical_branch_count_n_1)``
  (Lab 3; the weight 0 is ``fine_s1``).

Every run has the same wall-clock budget, ``RUNTIME`` seconds of search; the AC stage is not run, since the labs
read the DC repertoire only. The runs of ``toop_front/`` are CPU runs (``num_cuda_devices`` 0, ``JAX_PLATFORMS=cpu``);
``--gpu`` makes one CUDA device visible instead (``num_cuda_devices`` 1, with ``JAX_PLATFORMS=cuda,cpu`` and a JAX
that has the CUDA plugin), every other setting unchanged; ``device.txt`` next to ``res.json`` records the device
JAX used. The colon in ToOp's TensorBoard folder name is replaced on Windows, as in Chapter 25.
"""

import datetime
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import jax
import toop_engine_topology_optimizer.dc.main as dc_main
from omegaconf import DictConfig
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    prepare_importer_parameters,
    run_dc_optimization_stage,
    run_preprocessing,
    suppress_jax_logs,
)

TOOP_REPO = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[5] / 'ToOp'))
GRID = TOOP_REPO / 'data' / 'complex_grid' / 'grid.xiidm'
RUNTIME = 60.0
OBSERVED = ('overload_energy_n_1', 'switching_distance', 'split_subs', 'critical_branch_count_n_1', 'max_flow_n_1')
FINE = (('split_subs', 5), ('switching_distance', 45))
COARSE = (('split_subs', 5), ('switching_distance', 9))
EXPERIMENTS = {
    **{f'fine_s{s}': (FINE, 0.0, s) for s in range(1, 6)},
    **{f'coarse_s{s}': (COARSE, 0.0, s) for s in range(1, 6)},
    **{f'w{w}': (FINE, float(w), 1) for w in (1, 3, 10, 30, 100)},
}


class _FileSafeDatetime:
    @staticmethod
    def now() -> str:
        return datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')  # noqa: DTZ005


if os.name == 'nt':
    dc_main.datetime = types.SimpleNamespace(datetime=_FileSafeDatetime)


def preprocess(work_dir: Path, grid_xiidm: Path = GRID) -> Path:
    """Copy the grid into ``work_dir`` and run ToOp's import and preprocessing once; returns the data folder."""
    data_folder = work_dir / 'grid'
    if (data_folder / 'static_information.hdf5').exists():
        return data_folder
    work_dir.mkdir(parents=True, exist_ok=True)
    grid_copy = work_dir / 'grid.xiidm'
    shutil.copy(grid_xiidm, grid_copy)
    run_preprocessing(
        prepare_importer_parameters(grid_copy, data_folder),
        data_folder,
        PreprocessParameters(action_set_clip=2**10, preprocess_bb_outages=True),
        is_pandapower_net=False,
    )
    return data_folder


def dc_config(
    data_folder: Path,
    run_dir: Path,
    descriptors,
    critical_weight: float,
    seed: int,
    runtime: float,
    cuda_devices: int = 0,
):
    """The DC stage configuration of one experiment (ToOp defaults otherwise); ``cuda_devices`` 1 for the GPU."""
    target = [['overload_energy_n_1', 1.0]] + (
        [['critical_branch_count_n_1', critical_weight]] if critical_weight else []
    )
    return DictConfig(
        {
            'task_name': run_dir.name,
            'fixed_files': [str(data_folder / 'static_information.hdf5')],
            'double_precision': None,
            'tensorboard_dir': str(run_dir / 'tensorboard'),
            'stats_dir': str(run_dir / 'stats'),
            'summary_frequency': None,
            'checkpoint_frequency': None,
            'stdout': None,
            'double_limits': None,
            'num_cuda_devices': cuda_devices,
            'omp_num_threads': 4,
            'xla_force_host_platform_device_count': None,
            'output_json': str(run_dir / 'output.json'),
            'lf_config': {'distributed': False},
            'ga_config': {
                'runtime_seconds': runtime,
                'random_seed': seed,
                'n_worst_contingencies': 2,
                'me_descriptors': [{'metric': m, 'num_cells': c} for m, c in descriptors],
                'target_metrics': target,
                'observed_metrics': list(OBSERVED),
            },
        }
    )


def run(work_dir: Path, name: str, runtime: float = RUNTIME, cuda_devices: int = 0) -> Path:
    """Preprocess if needed, run one experiment and return its folder (``res.json``, ``dc_config.json``)."""
    descriptors, critical_weight, seed = EXPERIMENTS[name]
    data_folder = preprocess(work_dir)
    run_dir = work_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dc_config(data_folder, run_dir, descriptors, critical_weight, seed, runtime, cuda_devices)
    (run_dir / 'dc_config.json').write_text(json.dumps(dict(cfg), indent=2, default=str), encoding='utf-8')
    start = time.perf_counter()
    run_dc_optimization_stage(cfg, run_dir)
    wall = time.perf_counter() - start
    (run_dir / 'wall_seconds.txt').write_text(f'{wall:.1f}\n', encoding='utf-8')
    (run_dir / 'device.txt').write_text(f'{jax.devices()[0].device_kind}\n', encoding='utf-8')
    return run_dir


def main() -> None:
    """Run the experiments named on the command line (all by default) and print one summary line each."""
    suppress_jax_logs()
    work_dir, names = Path(sys.argv[1]), [a for a in sys.argv[2:] if a != '--gpu']
    cuda_devices = int('--gpu' in sys.argv[2:])
    for name in names or list(EXPERIMENTS):
        run_dir = run(work_dir, name, cuda_devices=cuda_devices)
        res = json.loads((run_dir / 'res.json').read_text(encoding='utf-8'))
        wall = (run_dir / 'wall_seconds.txt').read_text(encoding='utf-8').strip()
        device = (run_dir / 'device.txt').read_text(encoding='utf-8').strip()
        print(
            name, f'epochs={res["iteration"]}', f'elites={len(res["best_topos"])}', f'wall={wall}s', device, flush=True
        )


if __name__ == '__main__':
    main()
