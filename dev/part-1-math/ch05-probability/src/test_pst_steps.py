import math

import pst_steps as ps
import pytest


def test_support_and_symmetry_for_sigma_1():
    dist = ps.step_distribution(1.0)
    assert list(dist.index) == list(range(-4, 5))
    assert dist.sum() == pytest.approx(1)
    for k in range(1, 5):
        assert dist[k] == pytest.approx(dist[-k])
    assert dist[0] == pytest.approx(0.399, abs=5e-4)
    assert dist[1] == pytest.approx(0.242, abs=5e-4)


def test_no_move_and_fraction_changing():
    assert ps.probability_no_move() == pytest.approx(0.3989, abs=1e-4)
    assert ps.fraction_changing() == pytest.approx(0.1202, abs=1e-4)
    assert ps.fraction_changing() == pytest.approx(0.2 * (1 - ps.probability_no_move()))
    # the discrete P(0) nearly equals the continuous density 1/sqrt(2 pi): the lattice sum is close to the integral
    assert abs(ps.probability_no_move() - 1 / math.sqrt(2 * math.pi)) < 1e-5


def test_support_radius_follows_ceil_of_four_sigma():
    assert list(ps.step_distribution(0.1).index) == [-1, 0, 1]  # max(1, ceil(0.4)) = 1
    assert list(ps.step_distribution(2.5).index) == list(range(-10, 11))
    assert ps.probability_no_move(0.1) > 0.99 and ps.probability_no_move(3.0) < 0.14
