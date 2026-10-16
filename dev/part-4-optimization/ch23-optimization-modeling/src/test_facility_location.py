import pytest
from facility_location import solve


def test_both_formulations_share_the_integer_optimum():
    cost_strong, open_strong = solve(strong=True, relax=False)
    cost_big_m, open_big_m = solve(strong=False, relax=False)
    assert cost_strong == pytest.approx(cost_big_m)
    assert all(v in (0.0, 1.0) for v in open_strong.values())
    assert sum(open_strong.values()) >= 1
    assert open_strong == open_big_m


def test_strong_relaxation_is_tighter():
    integer, _ = solve(strong=True, relax=False)
    relaxed_strong, _ = solve(strong=True, relax=True)
    relaxed_big_m, _ = solve(strong=False, relax=True)
    assert relaxed_big_m < relaxed_strong <= integer + 1e-9
