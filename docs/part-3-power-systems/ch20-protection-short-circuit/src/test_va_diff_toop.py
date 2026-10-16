"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import sys

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_contingency_analysis')
from va_diff_toop import CH19_SRC, above, worst  # noqa: E402


def test_helpers_on_a_small_table():
    va = pd.DataFrame({'contingency': ['A', 'B', 'C'], 'element': ['A', 'B', 'C'], 'va_diff': [25.0, -12.0, np.nan]})
    assert above(va, 20.0).element.tolist() == ['A']
    assert above(va, 10.0).element.tolist() == ['A', 'B']
    assert above(va, 30.0).empty
    assert worst(va).element == 'A'


def test_chapter_19_grid_has_no_angle_above_10_degrees():
    sys.path.insert(0, str(CH19_SRC))
    import ac_n1_toop
    import pypowsybl

    path = ac_n1_toop.default_grid_path()
    if not path.exists():
        pytest.skip(f'{path} not found: set TOOP_REPO')
    net = pypowsybl.network.load(str(path))
    va = ac_n1_toop.frames(ac_n1_toop.run(net, ac_n1_toop.n1_definition(net)))['va_diff']
    assert len(va) == 9 and va.va_diff.isna().sum() == 1
    assert above(va, 20.0).empty and above(va, 10.0).empty
    w = worst(va)
    assert w.contingency in ('L1', 'L2') and abs(w.va_diff) == pytest.approx(0.485, abs=0.001)
