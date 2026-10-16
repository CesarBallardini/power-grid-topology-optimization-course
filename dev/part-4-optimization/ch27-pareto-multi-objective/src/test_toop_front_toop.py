"""Checks the runner's configuration against ToOp's models and the stored runs (`make test-toop`)."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
import toop_front_toop  # noqa: E402
from toop_engine_topology_optimizer.interfaces.messages.dc_params import BatchedMEParameters  # noqa: E402

RUNS = Path(__file__).resolve().parent / 'toop_front'


@pytest.mark.parametrize('name', list(toop_front_toop.EXPERIMENTS))
def test_config_is_valid_and_matches_the_stored_run(name, tmp_path):
    descriptors, weight, seed = toop_front_toop.EXPERIMENTS[name]
    cfg = toop_front_toop.dc_config(tmp_path / 'grid', tmp_path / name, descriptors, weight, seed, 60.0)
    ga = BatchedMEParameters(**cfg['ga_config'])
    stored = json.loads((RUNS / f'{name}.json').read_text(encoding='utf-8'))['args']['ga_config']
    assert ga.random_seed == stored['random_seed'] == seed
    assert [[m, w] for m, w in ga.target_metrics] == stored['target_metrics']
    assert [(d.metric, d.num_cells) for d in ga.me_descriptors] == [
        (d['metric'], d['num_cells']) for d in stored['me_descriptors']
    ]
    assert ga.runtime_seconds == stored['runtime_seconds'] == toop_front_toop.RUNTIME


def test_gpu_flag_changes_only_the_visible_cuda_devices(tmp_path):
    descriptors, weight, seed = toop_front_toop.EXPERIMENTS['fine_s1']
    cpu = toop_front_toop.dc_config(tmp_path / 'grid', tmp_path / 'a', descriptors, weight, seed, 60.0)
    gpu = toop_front_toop.dc_config(tmp_path / 'grid', tmp_path / 'a', descriptors, weight, seed, 60.0, 1)
    assert cpu['num_cuda_devices'] == 0 and gpu['num_cuda_devices'] == 1
    assert {k: v for k, v in cpu.items() if k != 'num_cuda_devices'} == {
        k: v for k, v in gpu.items() if k != 'num_cuda_devices'
    }


@pytest.mark.parametrize('name', [f'{d}_s{s}' for d in ('gpu', 'cpu') for s in range(1, 6)])
def test_gpu_comparison_runs_use_the_default_config_of_their_seed(name, tmp_path):
    descriptors, weight, seed = toop_front_toop.EXPERIMENTS[f'fine_s{name[-1]}']
    cfg = toop_front_toop.dc_config(tmp_path / 'grid', tmp_path / name, descriptors, weight, seed, 60.0)
    stored = json.loads((RUNS.parent / 'toop_front_gpu' / f'{name}.json').read_text(encoding='utf-8'))['args']
    assert stored['ga_config']['random_seed'] == seed
    assert stored['ga_config']['target_metrics'] == [list(t) for t in cfg['ga_config']['target_metrics']]
    assert [(d['metric'], d['num_cells']) for d in stored['ga_config']['me_descriptors']] == list(descriptors)
