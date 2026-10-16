import pytest
from action_space import assignments, assignments_up_to_symmetry, joint, toop_splits, unitary_topologies


def test_six_elements():
    assert assignments(6) == 64
    assert assignments_up_to_symmetry(6) == 32
    assert toop_splits(6) == 25
    assert unitary_topologies(6) == 26 == toop_splits(6) + 1


@pytest.mark.parametrize('d', [4, 5, 6, 7, 8])
def test_enumeration_matches_the_formula(d):
    assert unitary_topologies(d) == toop_splits(d) + 1


def test_chapter_19_values():
    assert [toop_splits(d) for d in (3, 4, 5, 6)] == [0, 3, 10, 25]


def test_joint_counts():
    assert joint(64) == 2**84
    assert joint(32) == 2**70
    assert joint(26) == 26**14
