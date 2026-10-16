import pytest
from action_count import n_actions


@pytest.mark.parametrize('d', range(4, 10))
def test_matches_closed_form(d):
    assert n_actions(d) == 2 ** (d - 1) - d


def test_four_branches_give_three_splits_plus_unsplit():
    assert n_actions(4) == 4
