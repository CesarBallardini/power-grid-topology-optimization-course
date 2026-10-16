import numpy as np
import pytest
from loadflow_limits import F_N1, breakpoints, new_limit


def test_breakpoints_for_the_default_factors():
    low, high = breakpoints()
    assert low == pytest.approx(250.0)
    assert high == pytest.approx(1000.0 / 1.2)


def test_three_branches_of_the_rule():
    assert new_limit(0.0) == pytest.approx(50.0)  # unloaded line: m * I_old
    assert new_limit(100.0) == pytest.approx(150.0)  # lower clip: I_lf + m I_old
    assert new_limit(500.0) == pytest.approx(600.0)  # f * I_lf
    assert new_limit(900.0) == pytest.approx(1000.0)  # upper clip: the old limit
    assert new_limit(1200.0) == pytest.approx(1000.0)


def test_vectorised_and_monotone():
    i = np.linspace(0.0, 1500.0, 301)
    out = new_limit(i)
    assert np.all(np.diff(out) >= -1e-9)
    assert np.all(out <= 1000.0 + 1e-9)
    assert new_limit(500.0, f=F_N1) == pytest.approx(700.0)
