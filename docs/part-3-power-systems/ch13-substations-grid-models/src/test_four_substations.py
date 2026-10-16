import pytest
from four_substations import VOLTAGE_LEVEL, classify_switches, network, switch_counts, voltage_levels


@pytest.fixture(scope='module')
def net():
    return network()


def test_five_node_breaker_voltage_levels(net):
    vl = voltage_levels(net)
    assert len(vl) == 5 and (vl['topology_kind'] == 'NODE_BREAKER').all()
    assert vl.loc[VOLTAGE_LEVEL, 'nominal_v'] == 400.0


def test_switch_counts(net):
    counts = switch_counts(net)
    assert counts.loc[VOLTAGE_LEVEL].to_dict() == {'BREAKER': 11, 'DISCONNECTOR': 22}
    assert counts.to_numpy().sum() == 59


def test_roles_of_s1vl2(net):
    roles = classify_switches(net)['role'].value_counts().to_dict()
    assert roles == {'busbar disconnector': 20, 'bay breaker': 10, 'coupler disconnector': 2, 'coupler': 1}
