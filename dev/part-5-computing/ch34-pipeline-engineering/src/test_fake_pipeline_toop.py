import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import fake_pipeline_toop  # noqa: E402


@pytest.fixture(scope='module')
def messages():
    if not fake_pipeline_toop.GRID_ROOT.exists():
        pytest.skip(f'{fake_pipeline_toop.GRID_ROOT} not found')
    return fake_pipeline_toop.run_dc_worker(fake_pipeline_toop.start_command(runtime_seconds=4.0))


def test_results_start_with_started_and_end_with_converged(messages):
    table = fake_pipeline_toop.message_table(messages['results'])
    assert table['type'].iloc[0] == 'OptimizationStartedResult'
    assert table['type'].iloc[-1] == 'OptimizationStoppedResult' and table['reason'].iloc[-1] == 'converged'
    assert (table['type'] == 'TopologyPushResult').sum() >= 1
    assert (table['optimizer'] == 'dc').all()
    assert 'heartbeats' in messages


def test_store_drops_the_count_of_the_stopped_optimization(messages):
    session, added, finished = fake_pipeline_toop.store_results(messages['results'])
    assert added == {} and finished == ['lab34']
    topos = fake_pipeline_toop.topology_table(session)
    assert len(topos) >= 2 and topos['unsplit'].sum() == 1
    _, added2, finished2 = fake_pipeline_toop.store_results(messages['results'][:-1])
    assert added2 == {'lab34': len(topos)} and finished2 == []


def test_replay_adds_nothing_and_scrub_empties(messages):
    session, _, _ = fake_pipeline_toop.store_results(messages['results'][:-1])
    _, added, _ = fake_pipeline_toop.store_results(messages['results'][:-1], session)
    assert added == {}
    fake_pipeline_toop.scrub_db(session, max_age_seconds=0)
    assert fake_pipeline_toop.topology_table(session).empty


def test_too_many_worst_contingencies_is_reported_as_an_error_result():
    if not fake_pipeline_toop.GRID_ROOT.exists():
        pytest.skip(f'{fake_pipeline_toop.GRID_ROOT} not found')
    messages = fake_pipeline_toop.run_dc_worker(
        fake_pipeline_toop.start_command(runtime_seconds=3.0, n_worst_contingencies=20)
    )
    table = fake_pipeline_toop.message_table(messages['results'])
    assert table['type'].tolist() == ['OptimizationStoppedResult'] and table['reason'].iloc[0] == 'error'
