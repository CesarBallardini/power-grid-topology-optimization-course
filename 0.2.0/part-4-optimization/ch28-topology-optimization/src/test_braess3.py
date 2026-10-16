import numpy as np
import pytest
from braess3 import LINES, RATING, X, big_m_cycle, dc_opf, gap_when_open


def test_opening_the_direct_line_lowers_the_cost():
    closed, open_ = dc_opf(), dc_opf((2,))
    assert closed['cost'] == pytest.approx(3000.0)
    assert open_['cost'] == pytest.approx(1800.0)
    np.testing.assert_allclose(closed['p'], [150, 30])
    np.testing.assert_allclose(open_['p'], [180, 0])
    assert abs(closed['f'][2]) == pytest.approx(RATING[2])  # the direct line is the bottleneck


def test_lmps_explain_it():
    closed, open_ = dc_opf(), dc_opf((2,))
    np.testing.assert_allclose(closed['lmp'], [10, 30, 50])
    np.testing.assert_allclose(open_['lmp'], [10, 10, 10])


def test_flows_follow_the_angles():
    r = dc_opf()
    for k, (i, j) in enumerate(LINES):
        assert r['f'][k] == pytest.approx(100 * (r['theta'][i] - r['theta'][j]) / X[k])


def test_big_m_is_valid_and_tight_enough():
    assert big_m_cycle(2) == pytest.approx(400.0)
    assert gap_when_open(2) == pytest.approx(360.0)
    assert gap_when_open(2) <= big_m_cycle(2)
