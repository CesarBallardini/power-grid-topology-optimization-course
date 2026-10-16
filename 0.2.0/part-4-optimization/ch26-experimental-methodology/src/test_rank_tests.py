import numpy as np
import pytest
from rank_tests import friedman, holm, pairwise_signed_rank, power_rank_sum, power_table, synthetic_instances


def test_synthetic_table_is_seeded_and_c_is_better_on_average():
    table = synthetic_instances()
    assert table.shape == (8, 3) and list(table.columns) == ['A', 'B', 'C']
    assert table.equals(synthetic_instances())
    assert table['C'].mean() < table[['A', 'B']].mean().mean()
    assert table.loc['instance 1', 'A'] == pytest.approx(43.37, abs=0.01)


def test_friedman_and_holm_on_the_default_data():
    table = synthetic_instances()
    statistic, p, ranks = friedman(table)
    assert statistic == pytest.approx(0.75, abs=0.01) and p == pytest.approx(0.6873, abs=1e-3)
    assert ranks.idxmin() == 'C'
    corrected = holm(pairwise_signed_rank(table))
    assert list(corrected.index) == ['A vs C', 'B vs C', 'A vs B']
    assert corrected['threshold'].tolist() == pytest.approx([0.05 / 3, 0.05 / 2, 0.05])
    assert not corrected['reject'].any()
    assert corrected['p_holm'].is_monotonic_increasing


def test_holm_rejects_small_p_values_in_order():
    corrected = holm({'x': 0.001, 'y': 0.03, 'z': 0.04})
    assert corrected['p_holm'].tolist() == pytest.approx([0.003, 0.06, 0.06])
    assert corrected['reject'].tolist() == [True, False, False]


def test_power_grows_with_effect_and_sample_size():
    assert power_rank_sum(0.0, 10, n_rep=500, seed=1) == pytest.approx(0.05, abs=0.03)  # the size of the test
    table = power_table(n_rep=300)
    assert table.shape == (4, 3)
    assert table.loc['4 MW', 30] > table.loc['4 MW', 5] and table.loc['4 MW', 10] > table.loc['1 MW', 10]
    assert np.all(np.diff(table.loc['2 MW'].to_numpy()) > 0)
