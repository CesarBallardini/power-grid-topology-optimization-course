"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
from grid_pipeline_toop import (  # noqa: E402
    TOOP_GRID,
    actions_per_station,
    funnel,
    import_grid,
    mask_counts,
    read_json,
    stage_that_removes_most,
    variants,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (  # noqa: E402
    PreprocessParameters,
    RelevantStationRules,
)


@pytest.fixture(scope='module')
def node_breaker(tmp_path_factory):
    if not TOOP_GRID.exists():
        pytest.skip(f'ToOp data file not found: {TOOP_GRID}')
    return import_grid(tmp_path_factory.mktemp('ch22') / 'node_breaker')


def test_import_writes_masks_and_relevant_substations(node_breaker):
    counts = mask_counts(node_breaker)
    assert counts['relevant_subs'] == 2
    assert counts['line_for_nminus1'] == 9 and counts['line_for_reward'] == 3  # six lines have no limits
    assert counts['switch_for_nminus1'] == 106
    aux = read_json(node_breaker, 'importer_auxiliary_file_path')['id_lists']
    assert aux['relevant_subs'] == ['VL2_0', 'VL3_0']
    nminus1 = read_json(node_breaker, 'nminus1_definition_file_path')
    assert len(nminus1['monitored_elements']) == 5 and len(nminus1['contingencies']) == 129


def test_funnel_keeps_one_station_with_ten_actions(node_breaker):
    table, info, _ = funnel(node_breaker)
    assert table.loc['compute_bridging_branches', 'n_relevant_subs'] == 2
    assert table.loc['assert_network_data', 'n_relevant_subs'] == 1  # VL3_0 has fewer than 4 non-bridge branches
    assert table.loc['enumerate_station_realizations', 'n_actions'] == 10
    assert table.loc['preprocess_done'].to_dict() == {
        'n_nodes': 7,
        'n_branches': 9,
        'n_relevant_subs': 1,
        'n_actions': 10,
        'n_disc_branches': 8,
    }
    assert stage_that_removes_most(table)['n_relevant_subs'] == 'assert_network_data'
    assert info['n_actions'] == 10 and info['n_monitored_branches'] == 3 and info['n_nminus1_cases'] == 11
    assert info['overload_energy_n0'] == 0.0 and info['overload_energy_n1'] == pytest.approx(61.44, abs=0.01)
    per_station = actions_per_station(node_breaker)
    assert per_station.loc['VL2_a'].to_dict() == {'branches': 5, 'actions': 10, 'bound': 11}


def test_variants_change_the_action_count(node_breaker):
    table = variants(node_breaker).n_actions
    assert table['default'] == table['action_set_filter_bridge_lookup=False'] == 10
    assert table['action_set_filter_bsdf_lodf=False'] == 11
    assert table['max_reassignments_per_sub=1'] == 5
    assert table['action_set_clip=2**3'] < 10  # a random draw of 8 of the 16 patterns, unseeded: 3 or 4 remain


def test_import_rules_do_not_add_stations_after_preprocessing(tmp_path):
    if not TOOP_GRID.exists():
        pytest.skip(f'ToOp data file not found: {TOOP_GRID}')
    folder = import_grid(tmp_path / 'two', rules=RelevantStationRules(min_connected_branches=2))
    assert read_json(folder, 'importer_auxiliary_file_path')['id_lists']['relevant_subs'] == ['VL2_0', 'VL3_0']
    assert funnel(folder, PreprocessParameters())[1]['n_relevant_subs'] == 1
    folder = import_grid(tmp_path / 'three', rules=RelevantStationRules(min_busbars=3))
    assert read_json(folder, 'importer_auxiliary_file_path')['id_lists']['relevant_subs'] == ['VL2_0']
