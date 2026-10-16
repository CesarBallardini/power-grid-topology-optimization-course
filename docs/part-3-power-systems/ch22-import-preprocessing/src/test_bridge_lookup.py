import numpy as np
from bridge_lookup import IS_BRIDGE, describe, funnel, passes_bridge_lookup


def test_counts_six_branches_one_bridge():
    enumerated, after_isolation, after_bridge, removed = funnel()
    assert (enumerated, after_isolation, after_bridge) == (32, 26, 21)
    assert len(removed) == 5


def test_removed_splits_pair_the_bridge_with_one_other_branch():
    removed = funnel()[3]
    # the bridge is branch 0; every removed split puts exactly one other branch on a busbar with it
    assert set(removed) == {'b0+b1', 'b0+b2', 'b0+b3', 'b0+b4', 'b0+b5'}
    for r in removed:
        moved = {int(s[1:]) for s in r.split('+')}
        busbar_with_bridge = moved if 0 in moved else set(range(6)) - moved
        assert len(busbar_with_bridge) == 2


def test_unsplit_and_balanced_splits_pass():
    assert passes_bridge_lookup(np.zeros(6, dtype=bool), IS_BRIDGE)
    assert passes_bridge_lookup(np.array([True, True, True, False, False, False]), IS_BRIDGE)
    assert not passes_bridge_lookup(np.array([True, True, False, False, False, False]), IS_BRIDGE)
    assert describe(np.zeros(6, dtype=bool)) == 'unsplit'


def test_without_bridges_nothing_is_removed():
    assert funnel(6, np.zeros(6, dtype=bool))[2] == 26
