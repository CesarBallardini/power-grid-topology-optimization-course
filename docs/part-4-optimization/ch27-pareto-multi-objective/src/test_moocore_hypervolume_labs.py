import numpy as np
import pytest

pytestmark = pytest.mark.labs
pytest.importorskip('moocore')

import pareto  # noqa: E402
from moocore_hypervolume_labs import random_fronts, reference_sweep  # noqa: E402


def test_exercise_4_values():
    sweep = reference_sweep(pareto.POINTS)
    np.testing.assert_allclose([s[1] for s in sweep], [37.0, 52.21, 262.0])
    np.testing.assert_allclose([s[2] for s in sweep], [37.0, 52.21, 262.0])


def test_random_fronts_agree():
    assert random_fronts(n_fronts=50) < 1e-12
