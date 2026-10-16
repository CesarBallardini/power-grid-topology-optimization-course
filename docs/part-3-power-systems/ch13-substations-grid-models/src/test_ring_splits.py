from ring_splits import closed_form, count_two_component_states, ring_couplers, timed_count


def test_ring_couplers_close_the_ring():
    assert ring_couplers(4) == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_brute_force_matches_n_choose_2():
    for n in range(3, 11):
        assert count_two_component_states(n) == closed_form(n) == n * (n - 1) // 2


def test_timed_count_returns_count_and_duration():
    count, seconds = timed_count(5)
    assert count == 10 and seconds >= 0.0
