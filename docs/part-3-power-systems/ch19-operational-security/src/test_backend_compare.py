import numpy as np
import pandapower.networks as pn
import pytest
from backend_compare import branch_ids, compare, powsybl_ids, powsybl_network


@pytest.fixture(scope='module')
def tables():
    return compare()


def test_round_trip_keeps_every_branch():
    net = pn.case14()
    ids = branch_ids(net)
    names = powsybl_ids(powsybl_network(net))
    assert len(ids) == 20 and all(i in names for i in ids)
    assert names['1-2'] == 'LINE-1-2' and names['4-7'] == 'TWT-4-7'


def test_single_branch_n1_agrees_to_a_few_hundredths_of_a_mw(tables):
    pp_table, py_table, diff = tables
    assert pp_table.shape == py_table.shape == (20, 20)
    assert len(diff) == 380  # 20 x 20 minus the outaged branch of each contingency
    assert diff.max() < 0.05
    assert diff.quantile(0.9) < 0.01
    assert np.isnan(np.diag(pp_table.to_numpy())).all() and np.isnan(np.diag(py_table.to_numpy())).all()
