import numpy as np
import pytest
from dc_vs_ac import dc_ac_frame, dc_case, own_dc_flows


@pytest.fixture(scope='module')
def case14():
    return dc_case('case14')


def test_own_flows_match_rundcpp_once_taps_are_included(case14):
    _, table, p, slack = case14
    np.testing.assert_allclose(own_dc_flows(table, p, slack), table['p_pandapower'], atol=1e-9)
    assert np.abs(own_dc_flows(table, p, slack, with_taps=False) - table['p_pandapower']).max() > 0.1


def test_injections_are_balanced_and_slack_is_the_ext_grid(case14):
    net, _, p, slack = case14
    assert abs(p.sum()) < 1e-9
    assert slack == net.ext_grid.bus.iloc[0]


def test_frame_has_ac_and_dc_columns_with_losses():
    frame = dc_ac_frame('case14')
    assert len(frame) == 20
    assert frame['losses_ac'].sum() > 0
    assert (frame['abs_error'] >= 0).all()
    assert frame.loc[frame['kind'] == 'trafo', 'tap'].ne(1).sum() == 3
