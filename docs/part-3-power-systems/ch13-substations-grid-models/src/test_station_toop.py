import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

from reassignment_distance import ASSET_BUSBAR, candidates, reassignment_distance  # noqa: E402
from station_toop import (  # noqa: E402
    CHAIN_COUPLERS,
    chain_station,
    four_substations_bus_groups,
    separation,
    switching_distance,
)


@pytest.fixture(scope='module')
def chain_info():
    return separation(chain_station())


def test_chain_station_has_the_three_splits_of_the_worked_example(chain_info):
    assert chain_info.separation_set.shape == (3, 2, 6)
    assert (chain_info.coupler_states.sum(axis=1) == 1).all()  # exactly one coupler open
    mine = candidates(CHAIN_COUPLERS, ASSET_BUSBAR)
    for row in chain_info.separation_set:
        assert any((row == c).all() or (row[::-1] == c).all() for c in mine)


def test_switching_distance_agrees_with_numpy(chain_info):
    closed = np.zeros(3, dtype=bool)
    for target in ([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 0]):
        target = np.array(target, dtype=bool)
        _best, _invert, reassignment, coupler = switching_distance(chain_info, closed, target)
        assert reassignment == reassignment_distance(candidates(CHAIN_COUPLERS, ASSET_BUSBAR), target).min()
        assert coupler == 1


def test_mirror_target_is_inverted(chain_info):
    closed = np.zeros(3, dtype=bool)
    best, invert, reassignment, _ = switching_distance(chain_info, closed, np.array([1, 1, 1, 0, 0, 0], dtype=bool))
    assert reassignment == 0 and invert is True and chain_info.coupler_states[best].tolist() == [False, True, False]


def test_s1vl2_has_one_split(tmp_path):
    groups = four_substations_bus_groups(tmp_path)
    assert sorted(g.voltage_level_id for g in groups) == ['S1VL1', 'S1VL2', 'S2VL1', 'S3VL1', 'S4VL1']
    s1vl2 = next(g for g in groups if g.voltage_level_id == 'S1VL2')
    assert len(s1vl2.busbars) == 2 and len(s1vl2.couplers) == 1 and not s1vl2.couplers[0].open
    info = separation(s1vl2)
    assert info.separation_set.shape[0] == 1 and info.coupler_states.tolist() == [[True]]
    assert np.asarray(info.coupler_distance).tolist() == [1]
