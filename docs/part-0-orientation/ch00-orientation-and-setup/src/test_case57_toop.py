"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
from case57_toop import (  # noqa: E402
    STAGE_OF,
    branch_type_counts,
    counts,
    entries,
    load,
    node_type_counts,
    timed_loadflows,
    write_folder,
)


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    folder = tmp_path_factory.mktemp('case57') / 'grid'
    before = write_folder(folder)
    si, network_data = load(folder)
    return folder, before, si, network_data


def test_import_stage_writes_only_its_own_files(case):
    folder, before, _, _ = case
    assert before == ['grid.xiidm', 'initial_topology', 'loadflow_parameters.json', 'logs', 'masks']
    assert all(STAGE_OF[name].startswith('import') for name in before)
    after = entries(folder)
    assert set(after) - set(before) == {
        'action_set.json',
        'action_set_diffs.hdf5',
        'nminus1_definition.json',
        'static_information.hdf5',
        'static_information_stats.json',
    }
    assert all(STAGE_OF[name].startswith('DC preprocessing') for name in set(after) - set(before))


def test_write_folder_refuses_a_used_folder(case):
    folder, *_ = case
    with pytest.raises(FileExistsError):
        write_folder(folder)


def test_counts_of_the_57_bus_case(case):
    _, _, si, network_data = case
    assert counts(si) == {'n_nodes': 72, 'n_branches': 82, 'n_timesteps': 1}
    assert node_type_counts(network_data) == {'BUS': 58, 'BUS_B': 12, 'PSTNode': 1, 'REDUCED_NODE': 1}
    assert branch_type_counts(network_data) == {'LINE': 64, 'TWO_WINDINGS_TRANSFORMER': 18}


def test_loadflow_runs_twice_with_the_right_shape(case):
    _, _, si, _ = case
    first, second = timed_loadflows(si)
    assert first > 0 and second > 0
    assert tuple(si.dynamic_information.unsplit_flow.shape) == (1, 82)
