"""A short DC optimization through ``run_dc_optimization_stage`` and its TensorBoard log, read without TensorBoard.

Lab step 4 of Chapter 34. Run with ToOp's interpreter from this folder::

    ../../../../../ToOp/.venv/Scripts/python.exe tensorboard_log_toop.py [run folder]

ToOp's environment has tensorboardX (the writer) but not TensorBoard (the viewer), so ``read_scalars`` parses the
event file directly: a TFRecord is a little-endian uint64 length, a uint32 CRC, the serialized ``Event`` and a
second CRC. At commit ``ba1874e`` ``dc/main.py`` names the log directory ``str(datetime.datetime.now())``, whose
colons Windows rejects; ``windows_safe_log_dir`` swaps that module's ``datetime`` for one whose ``__str__`` uses
dashes, which is what the course fork's fix does in the source.
"""

import datetime
import os
import struct
import sys
import tempfile
import types
from pathlib import Path

import toop_engine_topology_optimizer.dc.main as dc_main_module
from omegaconf import DictConfig
from tensorboardX.proto import event_pb2
from toop_engine_topology_optimizer.benchmark.benchmark_utils import run_dc_optimization_stage
from toop_engine_topology_optimizer.dc.main import CLIArgs
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef
from toop_engine_topology_optimizer.interfaces.messages.dc_params import BatchedMEParameters, LoadflowSolverParameters

TOOP = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
STATIC_INFORMATION = TOOP / 'data' / 'grid_node_breaker' / 'grid' / 'static_information.hdf5'


class _DashedDatetime(datetime.datetime):
    def __str__(self):
        return super().__str__().replace(':', '-')


def windows_safe_log_dir():
    """Make ``dc/main.py`` build a log directory name without colons (a no-op for the rest of the program)."""
    dc_main_module.datetime = types.SimpleNamespace(datetime=_DashedDatetime)


def stage_config(
    run_dir, static_information=STATIC_INFORMATION, runtime_seconds=12.0, iterations_per_epoch=500, seed=1
):
    """The ``DictConfig`` that ``notebooks/example3_e2e_pipeline.ipynb`` builds, for a short CPU run.

    On this 9-branch grid an epoch of 100 iterations takes less than the 15.6 ms tick of ``time.time()`` on Windows
    under Python 3.11, and ``ga_helpers.update_running_means`` divides by the zero time step; 500 iterations per
    epoch keep every epoch longer than a tick.
    """
    run_dir = Path(run_dir)
    cfg = DictConfig(
        CLIArgs(
            ga_config=BatchedMEParameters(
                runtime_seconds=runtime_seconds,
                iterations_per_epoch=iterations_per_epoch,
                random_seed=seed,
                n_worst_contingencies=2,
                me_descriptors=(DescriptorDef(metric='split_subs', num_cells=4),),
                observed_metrics=('overload_energy_n_1', 'split_subs', 'switching_distance'),
            ),
            lf_config=LoadflowSolverParameters(max_num_splits=3, batch_size=8),
            tensorboard_dir=str(run_dir / 'tensorboard'),
            stats_dir=str(run_dir / 'stats'),
        ).model_dump()
    )
    cfg.update(
        {
            'fixed_files': [str(static_information)],
            'task_name': 'lab',
            'output_json': str(run_dir / 'output.json'),
            'num_cuda_devices': 0,
            'omp_num_threads': 4,
            'xla_force_host_platform_device_count': None,
            'double_precision': True,
        }
    )
    return cfg


def run_stage(run_dir, **kwargs):
    """Run the DC stage into ``run_dir`` (``res.json``, ``stats/``, ``tensorboard/``); returns ``run_dir``."""
    windows_safe_log_dir()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dc_optimization_stage(stage_config(run_dir, **kwargs), run_dir)


def read_events(path):
    """Every ``Event`` of a TFRecord event file."""
    data = Path(path).read_bytes()
    pos = 0
    while pos < len(data):
        (length,) = struct.unpack('<Q', data[pos : pos + 8])
        pos += 12
        event = event_pb2.Event()
        event.ParseFromString(data[pos : pos + length])
        pos += length + 4
        yield event


def read_scalars(run_dir):
    """``{tag: [(step, value), ...]}`` over every event file under ``run_dir``."""
    series = {}
    for path in sorted(Path(run_dir).rglob('events.out.tfevents.*')):
        for event in read_events(path):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    series.setdefault(value.tag, []).append((event.step, value.simple_value))
    return series


def plateau_report(fitness):
    """Epoch of the last improvement, number of epochs and the final value of a ``(step, value)`` fitness series."""
    best, last_improvement = -float('inf'), 0
    for step, value in fitness:
        if value > best + 1e-9:
            best, last_improvement = value, step
    epochs = fitness[-1][0]
    return {
        'epochs': epochs,
        'last_improvement_epoch': last_improvement,
        'final_fitness': fitness[-1][1],
        'flat_tail_fraction': (epochs - last_improvement) / epochs if epochs else 0.0,
    }


def main():
    """Run 15 s of optimization and say whether the fitness had plateaued."""
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='toop_run_'))
    run_stage(run_dir)
    series = read_scalars(run_dir)
    print('tags:', {tag: len(points) for tag, points in series.items()})
    fitness = series['fitness']
    print('fitness per epoch (first 5):', fitness[:5])
    print('fitness per epoch (last 3):', fitness[-3:])
    print('plateau report:', plateau_report(fitness))
    print(
        f'view with: tensorboard --logdir "{run_dir / "tensorboard"}" --host 0.0.0.0  (TensorBoard not installed here)'
    )


if __name__ == '__main__':
    main()
