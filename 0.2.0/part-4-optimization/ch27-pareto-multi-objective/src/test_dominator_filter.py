import numpy as np
from dominator_filter import OBSERVED, counterexample_a, counterexample_b, dominator_mask, order_example, pareto_mask


def test_non_dominated_row_removed():
    df = counterexample_a()
    np.testing.assert_array_equal(dominator_mask(df, ['split_subs'], OBSERVED), [True, False])
    np.testing.assert_array_equal(pareto_mask(df, OBSERVED), [True, True])


def test_dominated_rows_kept():
    df = counterexample_b()
    np.testing.assert_array_equal(dominator_mask(df, ['split_subs'], OBSERVED), [True, True, True, True])
    np.testing.assert_array_equal(pareto_mask(df, OBSERVED), [True, False, True, False])


def test_order_of_target_metrics_matters():
    df = order_example()
    first = dominator_mask(df, ['split_subs', 'switching_distance'], ['disconnections'])
    second = dominator_mask(df, ['switching_distance', 'split_subs'], ['disconnections'])
    np.testing.assert_array_equal(first, [True, False, True])
    np.testing.assert_array_equal(second, [True, False, False])


def test_pareto_mask_maximizes_fitness():
    df = counterexample_b()
    assert pareto_mask(df, [])[0]  # the best fitness alone is never dominated
    assert pareto_mask(df, []).sum() == 1
