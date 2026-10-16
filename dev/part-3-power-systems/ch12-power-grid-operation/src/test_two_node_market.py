import numpy as np
import pytest
from two_node_market import nodal, sweep, zonal


def test_exercise_1a_matches_the_hint():
    n = nodal()
    assert (n.gen_a, n.gen_b) == (500.0, 300.0)
    assert (n.price_a, n.price_b) == (20.0, 60.0)
    assert n.rent == 20_000.0


def test_exercise_1b_matches_the_hint():
    z = zonal()
    assert (z.gen_a, z.gen_b, z.price) == (800.0, 0.0, 20.0)
    assert (z.redispatch_mw, z.redispatch_cost) == (300.0, 12_000.0)


def test_no_congestion_when_the_limit_exceeds_the_load():
    n = nodal(limit=900.0)
    assert n.price_a == n.price_b == 20.0 and n.rent == 0.0
    assert zonal(limit=900.0).redispatch_cost == 0.0


def test_load_above_both_generators_is_rejected():
    with pytest.raises(ValueError, match='cannot be served'):
        nodal(limit=100.0, load=1500.0)


def test_sweep_has_a_kink_at_the_load():
    limits = np.array([700.0, 800.0, 900.0])
    cost, rent = sweep(limits)
    assert list(cost) == [4000.0, 0.0, 0.0]
    assert list(rent) == [28_000.0, 0.0, 0.0]
