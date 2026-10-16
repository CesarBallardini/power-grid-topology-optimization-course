import matplotlib
import near_bridge as nb
import numpy as np
import pytest

matplotlib.use('Agg')


def test_weak_branch_turns_the_bridge_into_a_near_bridge():
    fr_, to_, g_ = nb.with_weak_branch(1e-2)
    assert len(g_) == 8 and (fr_[-1], to_[-1], g_[-1]) == (5, 0, 1e-2)
    den, _ = nb.removal_in_dtype(1e-2, np.float64)
    assert 0 < den < 0.01  # of the order of g_w / 2


def test_voltage_after_removal_is_one_over_the_weak_conductance():
    for g_w in nb.WEAK_CONDUCTANCES:
        assert nb.resolved_float64(g_w) == pytest.approx(1 / g_w, rel=1e-9)


def test_float32_error_grows_while_float64_stays_small():
    table = nb.precision_table()
    err32, err64 = table['rel_err_float32'].to_numpy(), table['rel_err_float64'].to_numpy()
    assert err32[0] < 1e-4 and err32[-1] > 0.01  # about 1e-5 to a few percent
    assert np.all(np.diff(err32) > 0)
    assert (err64 < 1e-9).all()
    assert table['denominator'].to_numpy() == pytest.approx([0.004965, 5e-05, 5e-07], rel=1e-2)


def test_condition_number_is_flat_before_and_slope_minus_one_after_removal():
    weak = np.logspace(-6, -2, 9)
    before, after = nb.condition_numbers(weak), nb.condition_numbers(weak, removed=True)
    assert before == pytest.approx(26, abs=0.3)
    assert nb.loglog_slope(weak, after) == pytest.approx(-1, abs=0.01)
    assert after[0] > 1e7 and after[-1] < 3100
    err32 = nb.precision_table(tuple(weak))['rel_err_float32'].to_numpy()
    assert -1.2 < nb.loglog_slope(weak, err32) < -0.6
    fig = nb.plot_condition(weak, before, after, err32)
    assert len(fig.axes[0].lines) == 3
    matplotlib.pyplot.close(fig)
