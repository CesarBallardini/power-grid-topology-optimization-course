import json

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import tensorboard_log_toop  # noqa: E402


@pytest.fixture(scope='module')
def run_dir(tmp_path_factory):
    if not tensorboard_log_toop.STATIC_INFORMATION.exists():
        pytest.skip(f'{tensorboard_log_toop.STATIC_INFORMATION} not found')
    path = tmp_path_factory.mktemp('run')
    tensorboard_log_toop.run_stage(path, runtime_seconds=5.0)
    return path


def test_stage_writes_res_json_and_an_event_file(run_dir):
    res = json.loads((run_dir / 'res.json').read_text())
    assert res['initial_fitness'] == pytest.approx(-61.44458010685736)
    assert res['max_fitness'] >= res['initial_fitness']
    assert list(run_dir.rglob('events.out.tfevents.*'))


def test_scalars_contain_fitness_per_epoch(run_dir):
    series = tensorboard_log_toop.read_scalars(run_dir)
    assert 'fitness' in series and 'overload_energy_n_1' in series
    fitness = series['fitness']
    assert fitness[0] == (0, pytest.approx(-61.44458010685736, abs=1e-4))
    report = tensorboard_log_toop.plateau_report(fitness)
    assert report['epochs'] >= 1 and report['final_fitness'] >= fitness[0][1]


def test_plateau_report_on_a_hand_made_series():
    report = tensorboard_log_toop.plateau_report([(0, -60.0), (1, -50.0), (2, -42.0), (3, -42.0), (4, -42.0)])
    assert report == {'epochs': 4, 'last_improvement_epoch': 2, 'final_fitness': -42.0, 'flat_tail_fraction': 0.5}
