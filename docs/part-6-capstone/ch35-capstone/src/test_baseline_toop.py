import json

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import baseline_toop  # noqa: E402


@pytest.fixture(scope='module')
def work_dir(tmp_path_factory):
    if not baseline_toop.NODE_BREAKER.exists():
        pytest.skip(f'{baseline_toop.NODE_BREAKER} not found')
    return baseline_toop.baseline(work_dir=tmp_path_factory.mktemp('capstone'), runs=1, runtime_seconds=5.0)


def test_run_writes_the_files_the_capstone_needs(work_dir):
    run_dir = work_dir / 'run_0'
    for name in ('res.json', 'dc_config.json', 'ac_config.json', 'unsplit_ac_metrics.json'):
        assert (run_dir / name).exists(), name
    for name in ('ac_metrics.json', 'orao_summary.json', 'modified_network.xiidm'):
        assert (run_dir / 'topology_0' / name).exists(), name
    res = json.loads((run_dir / 'res.json').read_text(encoding='utf-8'))
    assert res['initial_fitness'] == pytest.approx(-61.44458010685736)
    assert res['best_topos'][0]['actions'] == [5]


def test_tabulate_and_acceptance(work_dir):
    table = baseline_toop.tabulate(work_dir)
    assert table.columns.tolist() == [
        'run',
        'initial_fitness',
        'max_fitness',
        'iterations',
        'best_topos',
        'validated',
        'accepted',
    ]
    assert table['max_fitness'].iloc[0] == pytest.approx(-42.3453259854399)
    verdicts = baseline_toop.accepted(work_dir / 'run_0')
    assert verdicts[0]['accepted'] is False
    assert verdicts[0]['rejection']['criterion'] == 'overload-energy'
