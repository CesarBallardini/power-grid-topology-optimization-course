import numpy as np
from reassignment_distance import ASSET_BUSBAR, asset_table, candidates, reassignment_distance


def test_asset_table_has_one_busbar_per_asset():
    table = asset_table()
    assert table.shape == (4, 6) and (table.sum(axis=0) == 1).all()
    assert table[0].tolist() == [True, True, False, False, False, False]


def test_three_candidates_partition_the_assets():
    cands = candidates()
    assert cands.shape == (3, 2, len(ASSET_BUSBAR))
    assert (cands[:, 0] ^ cands[:, 1]).all()  # every asset on exactly one of the two buses


def test_exact_target_has_distance_zero_only_with_the_right_labels():
    cands = candidates()
    target = np.array([False, False, False, True, True, True])  # the split between busbars 1 and 2
    assert reassignment_distance(cands, target).min() == 0
    assert reassignment_distance(cands, ~target).min() == 0  # the mirror image is the same split
    assert reassignment_distance(cands, ~target, allow_swap=False).min() > 0


def test_distance_counts_two_per_moved_asset():
    cands = candidates()
    target = np.array([False, False, False, True, True, False])  # L5 moves to the first bus: no candidate has it
    best = reassignment_distance(cands, target).min()
    assert best == 2
