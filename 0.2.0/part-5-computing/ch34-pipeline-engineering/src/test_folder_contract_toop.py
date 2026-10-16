import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import folder_contract_toop  # noqa: E402


@pytest.fixture(scope='module')
def folder():
    if not folder_contract_toop.CASE57.exists():
        pytest.skip(f'{folder_contract_toop.CASE57} not found')
    return folder_contract_toop.CASE57


def test_every_file_of_case57_has_a_known_key(folder):
    table = folder_contract_toop.list_folder(folder)
    assert len(table) >= 15
    assert table['key'].notna().all(), table[table['key'].isna()]
    assert (table['writer'] != '?').all()


def test_ptdf_dataset_shape_and_attributes(folder):
    datasets, attrs = folder_contract_toop.hdf5_summary(folder / 'static_information.hdf5')
    assert datasets['ptdf'] == ((82, 72), 'float64')
    assert attrs['format'] == 'jax' and 'aggregation_metric' in attrs


def test_datasets_and_attributes_map_to_fields(folder):
    mapping, attr_map = folder_contract_toop.field_mapping(folder / 'static_information.hdf5')
    assert mapping['ptdf'] == 'DynamicInformation.ptdf'
    assert mapping['action_set_branch_actions'] == 'DynamicInformation.action_set.branch_actions'
    assert mapping['slack'].startswith('SolverConfig.slack')
    assert attr_map['batch_size_bsdf'] == 'SolverConfig.batch_size_bsdf'
    assert attr_map['version'] == 'file metadata'
