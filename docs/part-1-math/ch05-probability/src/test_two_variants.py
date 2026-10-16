import matplotlib
import numpy as np
import pytest
import scipy.stats as st
import two_variants as tv

matplotlib.use('Agg')


def test_runs_and_summary_for_seed_0():
    runs = tv.simulate_runs(0)
    assert runs.shape == (30, 2)
    s = tv.summary(runs)
    assert s.loc['A', 'mean'] == pytest.approx(98.79, abs=0.01)
    assert s.loc['B', 'mean'] == pytest.approx(105.76, abs=0.01)
    assert s.loc['A', 'median'] == pytest.approx(98.70, abs=0.01)
    assert s.loc['B', 'q90'] == pytest.approx(117.95, abs=0.01)
    assert (s['q10'] < s['median']).all() and (s['median'] < s['q90']).all()


def test_single_run_ordering_probability():
    exact = tv.probability_single_run_orders_correctly()
    assert exact == pytest.approx(st.norm.cdf(3 / (10 * np.sqrt(2))))
    assert exact == pytest.approx(0.584, abs=5e-4)
    assert tv.single_run_agreement(seed=0) == pytest.approx(exact, abs=0.005)


def test_welch_test_depends_on_the_seed():
    t, p = tv.welch_test(tv.simulate_runs(0))
    assert t < 0 and p == pytest.approx(0.004, abs=5e-4)
    pvalues = [tv.welch_test(tv.simulate_runs(seed))[1] for seed in range(1, 6)]
    assert any(p > 0.05 for p in pvalues)  # thirty runs do not always separate a 3-point difference


def test_many_seeds_recover_the_true_difference():
    diffs = [tv.simulate_runs(seed)['B'].mean() - tv.simulate_runs(seed)['A'].mean() for seed in range(200)]
    assert np.mean(diffs) == pytest.approx(3.0, abs=0.6)  # standard error of the mean of 200 differences is 0.18


def test_runs_needed_for_eighty_percent_power():
    assert tv.runs_needed() == pytest.approx(174, abs=1)
    assert tv.runs_needed(power=0.5) < tv.runs_needed()


def test_ecdf_figure():
    fig = tv.plot_ecdfs(tv.simulate_runs(0))
    assert len(fig.axes[0].lines) == 2
    matplotlib.pyplot.close(fig)
