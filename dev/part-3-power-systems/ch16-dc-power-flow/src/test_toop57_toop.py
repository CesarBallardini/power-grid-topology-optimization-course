"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
from toop57 import compare  # noqa: E402


@pytest.fixture(scope='module')
def case57(tmp_path_factory):
    return compare(tmp_path_factory.mktemp('case57'))


def test_unsplit_flow_has_a_time_axis(case57):
    frame, si = case57
    di = si.dynamic_information
    assert np.asarray(di.unsplit_flow).shape == (di.n_timesteps, di.n_branches) == (1, len(frame))
    assert np.asarray(di.nodal_injections).shape == (di.n_timesteps, di.n_nodes)


def test_unsplit_flow_is_the_backend_dc_flow_with_opposite_sign_to_powsybl(case57):
    frame, _ = case57
    assert frame['unsplit - basecase_dc'].abs().max() < 1e-6
    assert frame['unsplit + p1'].abs().max() < 0.5
