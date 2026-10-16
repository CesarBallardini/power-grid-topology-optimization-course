import matplotlib
import switching_distances

matplotlib.use('Agg')


def test_node_breaker_pairs():
    table = switching_distances.pairs(switching_distances.BASELINE / 'node_breaker' / 'run_0')
    assert len(table) == 1
    row = table.iloc[0]
    assert row['actions'] == [5]
    assert row['dc_switching_distance'] == 2.0 and row['ac_switching_distance'] == 3.0
    assert row['ac_overload_energy_n_1'] < row['dc_overload_energy_n_1']


def test_all_pairs_cover_every_run_and_plot_draws():
    table = switching_distances.all_pairs()
    assert set(table['grid']) >= {'node_breaker'}
    assert len(table) >= 3
    ax = switching_distances.plot(table)
    assert ax.get_xlabel().startswith('DC switching distance')
    assert len(ax.collections) == table['grid'].nunique()
