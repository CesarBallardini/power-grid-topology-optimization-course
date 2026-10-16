import matplotlib
import numpy as np
import pytest
import rank_one_lab as rl
from sherman_morrison import fr, g, n, reduced_incidence, remove_branch, to

matplotlib.use('Agg')


@pytest.fixture(scope='module')
def network():
    ar = reduced_incidence(fr, to, n)
    return ar, np.linalg.inv(ar.T @ np.diag(g) @ ar)


def test_denominator_table_flags_only_the_bridge(network):
    ar, minv = network
    table = rl.denominator_table(ar, g, minv, fr, to)
    assert len(table) == 7
    assert table.loc[6, 'denominator'] == pytest.approx(0, abs=1e-12)
    assert (table.loc[:5, 'denominator'] > 0.19).all()
    assert (table['g_reff'] <= 1 + 1e-12).all()
    assert table.loc[3, 'denominator'] == pytest.approx(0.298246, abs=1e-6)


def test_three_removal_methods_agree():
    ways = rl.remove_three_ways(3)
    np.testing.assert_allclose(ways['inverse_update'], ways['rebuild'], atol=1e-14)
    np.testing.assert_allclose(ways['solution_update'], ways['rebuild'], atol=1e-14)
    assert ways['rebuild'][4] > ways['before'][4]  # node 5 gets farther from ground
    assert ways['rebuild'][2] == pytest.approx(ways['rebuild'][3])  # node 3 now hangs off node 4 without current


def test_update_solution_is_the_inverse_update_applied_to_the_injection(network):
    ar, minv = network
    i_r = rl.injection()
    v = minv @ i_r
    for k in range(6):
        np.testing.assert_allclose(rl.update_solution(v, minv, ar, g, k), remove_branch(minv, ar, g, k) @ i_r)


def test_sparse_laplacian_matches_dense():
    fr_, to_, g_, n_ = rl.grid_network(5, seed=2)
    dense = reduced_incidence(fr_, to_, n_).T @ np.diag(g_) @ reduced_incidence(fr_, to_, n_)
    np.testing.assert_allclose(rl.sparse_reduced_laplacian(fr_, to_, g_, n_).toarray(), dense)
    assert n_ == 25 and len(g_) == 40


def test_timing_table_methods_agree_on_a_small_grid():
    table = rl.time_removals(*rl.grid_network(8, seed=0), n_removals=20)
    assert len(table) == 20
    assert table['max_diff'].max() < 1e-9
    assert (table['rebuild_s'] > 0).all() and (table['update_s'] > 0).all()
    assert table.attrs['factor_s'] > 0
    fig = rl.plot_timings(table)
    assert len(fig.axes[0].lines) == 2
    matplotlib.pyplot.close(fig)
