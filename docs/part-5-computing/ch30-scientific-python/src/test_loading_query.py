import pytest
from loading_query import pandas_query, polars_query, res_line_case57


def test_pandas_query_keeps_half_the_lines():
    df = res_line_case57()
    assert len(df) == 63  # lines only; the 17 transformers are in net.res_trafo
    result = pandas_query(df)
    above = df[df['loading_percent'] > df['loading_percent'].median()]
    assert len(above) == 31 and set(result.index) == set(above['from_bus'])
    assert (result > df['loading_percent'].median()).all()


def test_polars_query_agrees_with_pandas():
    pytest.importorskip('polars')
    df = res_line_case57()
    expected = pandas_query(df)
    got = polars_query(df)
    assert got['from_bus'].to_list() == sorted(expected.index)
    assert got['loading_percent'].to_list() == pytest.approx(expected.sort_index().to_list())
