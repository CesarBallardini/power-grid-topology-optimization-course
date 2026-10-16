import itertools
import math

import clipped_poisson as cp
import matplotlib
import numpy as np
import pytest
import scipy.stats as st

matplotlib.use('Agg')


def test_exact_pmf_for_lambda_2_m_3():
    pmf = cp.clipped_pmf(2.0, 3)
    assert list(pmf.index) == [1, 2, 3]
    assert pmf[1] == pytest.approx(math.exp(-2) * 3)  # 0.406
    assert pmf[2] == pytest.approx(math.exp(-2) * 2)  # 0.271
    assert pmf[3] == pytest.approx(1 - math.exp(-2) * 5)  # P(N >= 3) = 0.323
    assert pmf.to_numpy() == pytest.approx([0.406, 0.271, 0.323], abs=5e-4)
    assert pmf.sum() == pytest.approx(1)
    assert cp.clipped_mean(2.0, 3) == pytest.approx(1.917, abs=5e-4)


@pytest.mark.parametrize('m', range(1, 8))
def test_pmf_sums_to_one_and_matches_scipy(m):
    pmf = cp.clipped_pmf(2.0, m)
    assert pmf.sum() == pytest.approx(1)
    assert len(pmf) == m
    if m > 2:
        for k in range(2, m):
            assert pmf[k] == pytest.approx(st.poisson.pmf(k, 2.0))
    if m > 1:
        assert pmf[m] == pytest.approx(st.poisson.sf(m - 1, 2.0))
    else:
        assert pmf[1] == 1


def test_lower_clip_adds_exp_minus_lambda_and_upper_clip_lowers_the_mean():
    lam = 2.0
    assert cp.mean_of_max_with_one(lam) == pytest.approx(lam + math.exp(-lam))
    big_m = cp.clipped_mean(lam, 60)  # upper clip far in the tail: only the lower clip acts
    assert big_m == pytest.approx(cp.mean_of_max_with_one(lam), abs=1e-12)
    means = [cp.clipped_mean(lam, m) for m in range(1, 8)]
    assert all(a < b for a, b in itertools.pairwise(means))
    assert cp.clipped_mean(lam, 3) < lam < cp.clipped_mean(lam, 4)


def test_simulation_matches_the_exact_pmf():
    n, k = cp.simulate(seed=0)
    assert len(n) == 10_000 and k.min() == 1 and k.max() == 3
    freq = cp.frequencies(k)
    exact = cp.clipped_pmf(2.0, 3)
    assert np.abs(freq.to_numpy() - exact.to_numpy()).max() < 0.01  # 3 sigma of a 10 000-draw frequency is ~0.015
    assert cp.frequencies(n)[0] == pytest.approx(math.exp(-2), abs=0.01)
    table = cp.sample_mean_vs_m(seed=0)
    assert np.abs(table['sample_mean'] - table['exact_mean']).max() < 0.01


def test_invalid_m_is_rejected():
    with pytest.raises(ValueError, match='at least 1'):
        cp.clipped_pmf(2.0, 0)


def test_figures():
    n, k = cp.simulate(seed=1)
    fig = cp.plot_frequencies(n, k, 2.0, 3)
    assert len(fig.axes) == 2 and len(fig.axes[1].patches) == 3
    fig2 = cp.plot_mean_vs_m(cp.sample_mean_vs_m(), 2.0)
    assert len(fig2.axes[0].lines) == 4
    matplotlib.pyplot.close('all')
