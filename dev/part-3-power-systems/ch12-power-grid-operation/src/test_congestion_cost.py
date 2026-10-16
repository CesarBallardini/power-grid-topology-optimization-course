import numpy as np
from congestion_cost import COST_BREAKDOWN_MEUR, QUARTERS, changes, cost_per_mwh


def test_unit_cost_per_quarter():
    unit = cost_per_mwh(QUARTERS)
    assert unit['Q1 2025'] == 1000 * 882 / 9035
    assert np.isclose(unit['Q1 2026'], 95.05, atol=0.005)


def test_changes_between_quarters():
    c = changes(QUARTERS)
    assert c.loc['volume_gwh', 'change'] == -787.0
    assert c.loc['cost_meur', 'change_pct'] == 100 * (784 / 882 - 1)
    assert c.loc['eur_per_mwh', 'change'] < 0


def test_breakdown_adds_up_to_the_totals():
    assert list(COST_BREAKDOWN_MEUR.sum()) == list(QUARTERS['cost_meur'])
