import pytest
from knapsack_bb import branch_and_bound, lp_bound, pulp_check


def test_root_bound_is_ten_and_two_thirds_with_b_fractional():
    bound, fractional, feasible, _ = lp_bound({})
    assert bound == pytest.approx(5 + 3 + 4 * 2 / 3)
    assert fractional == 'b' and feasible


def test_tree_finds_nine_with_a_and_b():
    root, value, x = branch_and_bound()
    assert value == 9 and x == {'a': 1, 'b': 1, 'c': 0}
    assert root.fractional == 'b'
    b0, b1 = root.children
    assert b0.status.startswith('integer, new incumbent 8')
    assert b1.bound == pytest.approx(9.5) and b1.fractional == 'a'
    a0, a1 = b1.children
    assert a0.status.startswith('pruned by bound')
    assert a1.status.startswith('integer, new incumbent 9')


def test_pulp_agrees():
    value, x = pulp_check()
    assert value == pytest.approx(9.0) and x == {'a': 1, 'b': 1, 'c': 0}
