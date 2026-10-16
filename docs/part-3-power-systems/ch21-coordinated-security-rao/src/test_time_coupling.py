import itertools

import numpy as np
import pytest
from time_coupling import HOURS, N_TOPO, best_sequence, best_single, break_even, overload_table


def brute_force(table, c):
    best = np.inf
    for path in itertools.product(range(table.shape[1]), repeat=table.shape[0]):
        cost = sum(table[h, t] for h, t in enumerate(path)) + c * sum(a != b for a, b in itertools.pairwise(path))
        best = min(best, cost)
    return best


def test_table_shape_and_sign():
    table = overload_table()
    assert table.shape == (HOURS, N_TOPO)
    assert (table >= 0).all()


@pytest.mark.parametrize('c', [0.0, 3.0, 12.5])
def test_dynamic_program_matches_brute_force_on_a_short_day(c):
    table = overload_table()[:6]
    _, cost, _ = best_sequence(table, c)
    assert cost == pytest.approx(brute_force(table, c))


def test_free_switching_picks_the_hourly_minimum():
    table = overload_table()
    path, cost, _ = best_sequence(table, 0.0)
    assert cost == pytest.approx(table.min(axis=1).sum())
    assert all(table[h, t] == table[h].min() for h, t in enumerate(path))  # hour 13 is a tie between 1 and 2


def test_expensive_switching_reduces_to_the_single_topology():
    table = overload_table()
    c_star = break_even(table)
    path, cost, switches = best_sequence(table, c_star)
    topo, total = best_single(table)
    assert switches == 0
    assert path == [topo] * HOURS
    assert cost == pytest.approx(total)
    assert best_sequence(table, c_star - 0.5)[2] > 0
