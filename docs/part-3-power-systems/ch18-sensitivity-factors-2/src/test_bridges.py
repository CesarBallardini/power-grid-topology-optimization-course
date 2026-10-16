import numpy as np
from bridges import TOLERANCES, one_minus_hkk, random_grid, survey


def test_random_grid_has_a_leaf_on_the_last_branch():
    rng = np.random.default_rng(1)
    frm, to, b, k = random_grid(10, 6, rng)
    assert len(frm) == 10 - 1 + 6 + 1 and k == len(frm) - 1
    assert to[k] == 10 and np.sum((frm == 10) | (to == 10)) == 1
    assert b.min() >= 2.0 and b.max() <= 20.0


def test_radial_branch_has_h_kk_one_up_to_round_off():
    values = survey(200, seed=3)
    assert np.abs(values).max() < 1e-12
    assert 0.3 < np.mean(values == 0.0) < 1.0  # often exact, not always


def test_meshed_branch_is_not_flagged():
    rng = np.random.default_rng(2)
    frm, to, b, k = random_grid(8, 6, rng)
    meshed = [j for j in range(len(frm)) if j != k and np.sum((frm == frm[j]) | (to == frm[j])) > 1]
    assert abs(one_minus_hkk(frm, to, b, meshed[0])) > 1e-6


def test_every_tolerance_but_the_exact_one_catches_all_bridges():
    values = np.abs(survey(300, seed=5))
    for name, tol in TOLERANCES.items():
        if tol > 0:
            assert np.all(values <= tol), name
