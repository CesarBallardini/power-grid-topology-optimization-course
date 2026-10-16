import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ga_experiment import (  # noqa: E402
    CONFIGURATIONS,
    analysis,
    anytime,
    bootstrap_median_difference,
    evaluations_within,
    final_values,
    plot_anytime,
    protocol,
    run_all,
    wall_clock_budget,
)


@pytest.fixture(scope='module')
def runs():
    return run_all(seeds=(0, 4, 7), budget=400)


def test_protocol_names_every_element():
    p = protocol()
    assert {
        'instance',
        'configurations',
        'budget',
        'measure',
        'seeds',
        'test',
        'effect size',
        'significance level',
    } <= set(p)
    assert p['configurations'] is CONFIGURATIONS and p['significance level'] == 0.05


def test_runs_are_logged_per_evaluation_and_seeded(runs):
    assert set(runs['configuration']) == set(CONFIGURATIONS) and len(runs) == 2 * 3 * 400
    assert runs.groupby(['configuration', 'seed'])['evaluation'].max().eq(400).all()
    assert runs.groupby(['configuration', 'seed'])['seconds'].is_monotonic_increasing.all()
    finals = final_values(runs)
    assert finals['crossover'].shape == (3,)
    again = final_values(run_all(seeds=(0,), budget=400))
    assert again['crossover'][0] == finals['crossover'][0] and again['no crossover'][0] == finals['no crossover'][0]


def test_anytime_curves_and_plot(runs):
    curve = anytime(runs, 'evaluation', n_points=20)
    assert len(curve) == 40 and (curve['q25'] <= curve['median']).all() and (curve['median'] <= curve['q75']).all()
    assert curve.groupby('configuration')['median'].is_monotonic_decreasing.all()
    fig, ax = plt.subplots()
    plot_anytime(runs, 'seconds', ax)
    assert len(ax.get_lines()) == 2 and len(ax.collections) == 2
    plt.close(fig)


def test_analysis_agrees_with_compare_runs_and_is_seeded():
    a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    b = np.array([12.5, 15.0, 16.0, 17.0, 18.0])
    result = analysis(a, b)
    assert result['U'] == 2.0 and result['A12'] == pytest.approx(0.92)  # a beats b in 23 of 25 pairs
    assert result['p'] == pytest.approx(0.0317, abs=1e-3) and result['median a'] == 12.0 and result['median b'] == 16.0
    low, high = bootstrap_median_difference(a, b)
    assert low < high <= 0.0 and bootstrap_median_difference(a, b) == (low, high)  # seeded
    assert bootstrap_median_difference(a, a)[0] <= 0.0 <= bootstrap_median_difference(a, a)[1]


def test_wall_clock_budget_counts_fewer_evaluations_for_slowed_runs(runs):
    budget = float(runs.groupby(['configuration', 'seed'])['seconds'].max().median() / 2)
    n_eval = evaluations_within(runs, budget, slow_seeds=(0,), slowdown=3.0)
    quiet = evaluations_within(runs, budget)
    for name in CONFIGURATIONS:
        assert n_eval[(name, 0)] < quiet[(name, 0)] and n_eval[(name, 4)] == quiet[(name, 4)]
    values = wall_clock_budget(runs, budget, slow_seeds=(0,))
    assert set(values) == set(CONFIGURATIONS) and all(v.shape == (3,) for v in values.values())
