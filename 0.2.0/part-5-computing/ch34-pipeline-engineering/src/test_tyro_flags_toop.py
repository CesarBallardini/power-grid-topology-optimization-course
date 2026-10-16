import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import tyro_flags_toop  # noqa: E402


@pytest.fixture(scope='module')
def text():
    return tyro_flags_toop.help_text()


def test_every_mapped_flag_is_in_the_help(text):
    table = tyro_flags_toop.flag_table(text)
    missing = [flag for flag, row in table.items() if not row['present']]
    assert not missing, missing
    assert table['--ga-config.runtime-seconds']['default'] == 60
    assert table['--lf-config.batch-size']['default'] == 8


def test_booleans_become_flag_pairs_and_the_nested_model_is_a_group(text):
    assert '--ga-config.enable-nodal-inj-optim, --ga-config.no-enable-nodal-inj-optim' in text
    assert 'ga-config options' in text and 'lf-config options' in text
