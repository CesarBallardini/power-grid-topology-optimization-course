import numpy as np
import pytest
from powsybl_slack import BALANCE, compare, solve_ieee14


@pytest.fixture(scope='module')
def tables():
    return compare(1.1)


def test_stock_case_is_already_balanced():
    gens, _ = compare(1.0)
    np.testing.assert_allclose(gens['single slack'], gens['proportional to Pmax'], atol=1e-6)
    assert abs(solve_ieee14(BALANCE['single slack'])[2]) < 0.01


def test_single_slack_leaves_the_mismatch_on_the_slack_bus(tables):
    gens, _ = tables
    np.testing.assert_allclose(gens['single slack'], [232.4, 40.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert solve_ieee14(BALANCE['single slack'], 1.1)[2] > 25  # about 29 MW of extra load and losses


def test_proportional_to_p_shares_in_the_ratio_of_the_targets(tables):
    gens, _ = tables
    pick_up = gens['proportional to P'] - gens['single slack']
    assert pick_up[['B3-G', 'B6-G', 'B8-G']].abs().max() < 1e-6
    assert pick_up['B1-G'] / pick_up['B2-G'] == pytest.approx(232.4 / 40.0, rel=1e-3)


def test_proportional_to_pmax_shares_equally_between_the_generating_units(tables):
    gens, _ = tables
    pick_up = gens['proportional to Pmax'] - gens['single slack']
    assert pick_up['B1-G'] == pytest.approx(pick_up['B2-G'], rel=1e-3)  # both have max_p = 9999 MW
    assert pick_up[['B3-G', 'B6-G', 'B8-G']].abs().max() < 1e-6
    assert pick_up['B1-G'] > 10


def test_distributed_slack_leaves_a_small_mismatch():
    for name in ('proportional to P', 'proportional to Pmax'):
        assert abs(solve_ieee14(BALANCE[name], 1.1)[2]) < 1.0
