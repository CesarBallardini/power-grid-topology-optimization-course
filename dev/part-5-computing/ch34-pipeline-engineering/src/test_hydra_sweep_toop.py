import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import hydra_sweep_toop  # noqa: E402


@pytest.fixture(scope='module')
def sweep(tmp_path_factory):
    if not hydra_sweep_toop.GRID_XIIDM.exists():
        pytest.skip(f'{hydra_sweep_toop.GRID_XIIDM} not found')
    work_dir = tmp_path_factory.mktemp('sweep')
    multirun, code = hydra_sweep_toop.run_sweep(work_dir, runtimes=(3,), n_worst=(1, 2))
    return work_dir, multirun, code


def test_two_jobs_with_a_hydra_folder_each(sweep):
    work_dir, multirun, _ = sweep
    rows = hydra_sweep_toop.job_summaries(multirun)
    assert [r['job'] for r in rows] == ['0', '1']
    assert all('ga_config.runtime_seconds=3' in r['overrides'] for r in rows)
    assert {r['overrides'][1] for r in rows} == {
        'ga_config.n_worst_contingencies=1',
        'ga_config.n_worst_contingencies=2',
    }
    assert (work_dir / 'data' / 'gnb' / 'grid' / 'static_information.hdf5').exists()
    assert not any(
        (hydra_sweep_toop.TOOP / 'data' / 'grid_node_breaker' / 'results').glob('multirun/*/*/hydra_sweep*')
    )


def test_assessment_runs_and_reports_only_finished_jobs(sweep):
    work_dir, multirun, code = sweep
    report = hydra_sweep_toop.assess(multirun, work_dir / 'aggregate_report.json')
    summaries = list(multirun.rglob('benchmark_summary.json'))
    if code == 0:
        assert summaries and isinstance(report, dict) and report
    else:  # ba1874e: every job dies in the AC stage on a missing ac_validation key, after its DC stage
        assert not summaries and report is None
