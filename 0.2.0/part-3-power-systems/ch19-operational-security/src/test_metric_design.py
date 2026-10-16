import metrics
import numpy as np
import pytest
from metric_design import (
    acceptance,
    acceptance_with_slack,
    changed,
    dead_zone_example,
    energy_and_count,
    fitness_table,
    penalty_curves,
)


def test_count_cannot_move_without_the_energy_on_the_worked_example():
    assert energy_and_count(metrics.n_1) == (35.0, 3)
    assert energy_and_count(changed(metrics.n_1, 1, 0, 140.0)) == (55.0, 3)  # energy moves, count does not
    assert energy_and_count(changed(metrics.n_1, 0, 0, 115.0)) == (35.0, 3)  # non-worst case: nothing moves
    smaller = metrics.n_1 * np.array([1.0, 1.0, 0.9])
    assert energy_and_count(smaller) == (30.0, 2)
    energy, count = energy_and_count(changed(smaller, 1, 2, 50.001))
    assert count == 3 and energy == pytest.approx(30.001)  # the count jumps for a vanishing energy change


def test_penalty_curves_scale_with_the_rating_except_cumulative_overload():
    loading = np.array([100.0, 110.0, 150.0])
    small, large = penalty_curves(loading, 100.0), penalty_curves(loading, 1000.0)
    np.testing.assert_allclose(small['overload_energy'], [0, 10, 50])
    np.testing.assert_allclose(large['overload_energy'], [0, 100, 500])
    np.testing.assert_allclose(small['power_law'], [0, (1.1**1.5 - 1) * 100, (1.5**1.5 - 1) * 100])
    np.testing.assert_allclose(large['power_law'], 10 * small['power_law'])
    np.testing.assert_allclose(small['cumulative_overload'], large['cumulative_overload'])
    np.testing.assert_allclose(small['cumulative_overload'], [0, 0.1, 0.5])


def test_dead_zone_penalises_a_redistribution_that_overloads_nothing():
    r = dead_zone_example()
    np.testing.assert_allclose(r['limited_limits'], [95.0, 90.0])
    assert r['unsplit_overload_energy_limited_n_0'] == 0.0
    assert r['split_overload_energy_limited_n_0'] == pytest.approx(2.0)
    assert r['split_overload_energy_n_0'] == 0.0


def test_acceptance_rejects_on_the_zero_baseline_and_slack_accepts():
    unsplit = {'non_converging_loadflows': 2, 'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 0}
    split = {
        'non_converging_loadflows': 2,
        'disconnected_branches': 1,
        'overload_energy_n_1': 0.0,
        'critical_branch_count_n_1': 1,
    }
    assert acceptance(split, unsplit) == 'critical-branch-count'
    assert acceptance_with_slack(split, unsplit) is None
    assert acceptance({**split, 'non_converging_loadflows': 4}, unsplit) == 'convergence'
    assert acceptance({**split, 'overload_energy_n_1': 1.0}, unsplit) == 'overload-energy'
    assert acceptance({**split, 'critical_branch_count_n_1': 0}, unsplit) is None
    assert (
        acceptance(
            {'overload_energy_n_1': 94.0, 'critical_branch_count_n_1': 11},
            {'overload_energy_n_1': 100.0, 'critical_branch_count_n_1': 10},
        )
        is None
    )
    assert (
        acceptance(
            {'overload_energy_n_1': 96.0, 'critical_branch_count_n_1': 10},
            {'overload_energy_n_1': 100.0, 'critical_branch_count_n_1': 10},
        )
        == 'overload-energy'
    )


def test_fitness_table_sums_to_713():
    table = fitness_table(
        {'overload_energy_n_1': 1.0, 'critical_branch_count_n_1': 10.0},
        {'overload_energy_n_1': 213.0, 'critical_branch_count_n_1': 50},
    )
    assert table.contribution.sum() == 713.0 and list(table.metric) == [
        'overload_energy_n_1',
        'critical_branch_count_n_1',
    ]
