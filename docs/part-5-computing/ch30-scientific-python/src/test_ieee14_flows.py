import matplotlib

matplotlib.use('Agg')

from ieee14_flows import bar_chart, line_flows  # noqa: E402


def test_line_flows_sorted_and_close():
    df = line_flows()
    assert len(df) == 17 and list(df.columns) == ['p1_ac', 'p1_dc']
    mags = df['p1_ac'].abs().to_numpy()
    assert (mags[:-1] >= mags[1:]).all()
    assert (df['p1_ac'] - df['p1_dc']).abs().max() < 10  # the lossless DC model is in the same ballpark


def test_bar_chart_has_two_series():
    fig = bar_chart(line_flows())
    assert len(fig.axes[0].containers) == 2
    matplotlib.pyplot.close(fig)
