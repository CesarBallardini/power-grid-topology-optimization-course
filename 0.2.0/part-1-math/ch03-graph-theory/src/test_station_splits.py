import itertools

import station_splits as ss


def test_ring_has_six_states_and_six_bipartitions():
    assert ss.count_states_and_bipartitions(ss.RING, ss.N_BUSBARS) == (6, 6)
    states = ss.two_busbar_states(ss.RING, ss.N_BUSBARS)
    assert all(sum(state) == 2 for state, _ in states)  # exactly two couplers open


def test_diagonal_adds_states_but_no_bipartition():
    assert ss.count_states_and_bipartitions(ss.RING_WITH_DIAGONAL, ss.N_BUSBARS) == (12, 6)
    ring = {b for _, b in ss.two_busbar_states(ss.RING, ss.N_BUSBARS)}
    diagonal = {b for _, b in ss.two_busbar_states(ss.RING_WITH_DIAGONAL, ss.N_BUSBARS)}
    assert ring == diagonal


def test_bipartition_count_is_two_to_the_n_minus_one_minus_one():
    for n in range(2, 7):
        assert len(ss.all_bipartitions(n)) == 2 ** (n - 1) - 1
        assert len(set(ss.all_bipartitions(n))) == 2 ** (n - 1) - 1


def test_opposite_corners_are_the_unreachable_bipartition():
    expected = [(frozenset({0, 2}), frozenset({1, 3}))]
    assert ss.unreachable_bipartitions(ss.RING, ss.N_BUSBARS) == expected
    assert ss.unreachable_bipartitions(ss.RING_WITH_DIAGONAL, ss.N_BUSBARS) == expected
    assert ss.describe(expected[0]) == '{0, 2} | {1, 3}'


def test_complete_station_reaches_every_bipartition():
    couplers = list(itertools.combinations(range(4), 2))
    assert ss.unreachable_bipartitions(couplers, 4) == []
    assert ss.count_states_and_bipartitions(couplers, 4)[1] == 7
