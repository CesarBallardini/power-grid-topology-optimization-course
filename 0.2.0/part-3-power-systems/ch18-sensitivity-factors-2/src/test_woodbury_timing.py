import pandapower.networks as pn
import pytest
from ieee14 import from_pandapower
from woodbury_timing import changed, check, timing


@pytest.fixture(scope='module')
def grid118():
    return from_pandapower(pn.case118())


def test_case118_loads(grid118):
    assert grid118.n_bus == 118 and grid118.n_branch == 186


@pytest.mark.parametrize('k', [1, 2, 3])
def test_woodbury_matches_recomputation_on_case118(grid118, k):
    assert check(grid118, k) < 1e-10


def test_changed_scales_only_the_named_branches(grid118):
    g = changed(grid118, [0, 5], [0.5, -0.5])
    assert g.b[0] == pytest.approx(1.5 * grid118.b[0]) and g.b[5] == pytest.approx(0.5 * grid118.b[5])
    assert g.b[1] == grid118.b[1]


def test_timing_returns_one_row_per_k(grid118):
    rows = timing(grid118, (1, 3), number=1)
    assert [k for k, _, _ in rows] == [1, 3]
    assert all(t_w > 0 and t_r > 0 for _, t_w, t_r in rows)
