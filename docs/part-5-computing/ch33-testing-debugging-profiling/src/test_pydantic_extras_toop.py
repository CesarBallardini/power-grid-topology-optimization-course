import pydantic_extras
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import pydantic_extras_toop  # noqa: E402


def test_preprocess_parameters_drop_the_busbar_outage_keyword():
    from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters

    params = PreprocessParameters(enable_bb_outage=True)
    assert 'enable_bb_outage' not in params.model_dump()
    assert params.preprocess_bb_outages is False
    with pytest.raises(AssertionError, match='silently drops'):
        pydantic_extras.assert_forbids_extras(PreprocessParameters)


def test_action_set_forbids_extras():
    from toop_engine_interfaces.stored_action_set import ActionSet

    pydantic_extras.assert_forbids_extras(ActionSet)


def test_every_message_model_of_both_packages_ignores_unknown_keywords():
    for package in pydantic_extras_toop.PACKAGES:
        table = pydantic_extras_toop.policies(package)
        assert table, package
        assert set(table.values()) == {'ignore'}, {k: v for k, v in table.items() if v != 'ignore'}
