"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
import pypowsybl as pp  # noqa: E402
from orao_switches_toop import (  # noqa: E402
    TOOP_GRID,
    ac_validate_every_action,
    acceptance,
    bay_breakers,
    flag_disconnectors,
    import_grid,
    metrics,
    stored_action_set,
    switch_table,
)


@pytest.fixture(scope='module')
def validated(tmp_path_factory):
    if not TOOP_GRID.exists():
        pytest.skip(f'ToOp data file not found: {TOOP_GRID}')
    work = tmp_path_factory.mktemp('orao')
    data_folder = import_grid(work)
    run_dir = work / 'run'
    paths = ac_validate_every_action(data_folder, run_dir)
    return data_folder, run_dir, paths


def test_ten_single_actions_none_accepted(validated):
    data_folder, run_dir, paths = validated
    assert len(paths) == 10 == len(stored_action_set(data_folder).local_actions)
    verdicts = acceptance(run_dir, paths)
    assert (verdicts.verdict != 'accepted').all()
    unsplit = metrics(run_dir / 'unsplit_ac_metrics.json')
    assert unsplit.fitness == pytest.approx(16.34, abs=0.01)
    assert verdicts.loc[verdicts.action > 0, 'fitness'].min() == pytest.approx(25.38, abs=0.01)


def test_orao_summary_lists_switches_and_flags_disconnectors(validated):
    data_folder, run_dir, paths = validated
    net = pp.network.load(str(data_folder / 'grid.xiidm'))
    bays = bay_breakers(stored_action_set(data_folder))
    assert bays['L12_DISCONNECTOR_4_0'] == 'L12_BREAKER'
    verdicts = acceptance(run_dir, paths)
    best = verdicts.loc[verdicts.action > 0, 'fitness'].idxmin()
    table = flag_disconnectors(switch_table(run_dir / best, net), bays)
    assert set(table.kind) == {'BREAKER', 'DISCONNECTOR'}
    assert (table.voltage_level_id == 'VL2').all()
    assert table.loc['VL2_BREAKER', 'open'] and table.loc['VL2_BREAKER', 'kind'] == 'BREAKER'
    assert table.needs_breaker.sum() == (table.kind == 'DISCONNECTOR').sum() == 2
    assert switch_table(run_dir / 'topology_0', net).empty  # the unsplit action moves nothing
