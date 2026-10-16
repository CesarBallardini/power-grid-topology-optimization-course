import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402
from case57_ac_dc import histogram, side_by_side, to_polars  # noqa: E402


def test_side_by_side_has_one_row_per_line():
    df = side_by_side()
    assert df.shape == (63, 3)  # case57 has 63 lines and 17 transformers
    assert (df['difference_mw'] == df['ac_p_from_mw'] - df['dc_p_from_mw']).all()
    assert df['difference_mw'].abs().max() < 25


def test_histogram_counts_every_line():
    df = side_by_side()
    fig = histogram(df, bins=10)
    assert sum(patch.get_height() for patch in fig.axes[0].patches) == len(df)
    matplotlib.pyplot.close(fig)


def test_polars_frame_matches_pandas():
    pytest.importorskip('polars')
    df = side_by_side()
    pf = to_polars(df)
    assert pf.shape == (63, 4) and pf['difference_mw'].sum() == pytest.approx(df['difference_mw'].sum())
