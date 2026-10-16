"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
import case7  # noqa: E402
from case7_toop import describe, load, matrices, single_splits, unsplit, write_case  # noqa: E402


@pytest.fixture(scope='module')
def opf_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp('case7')
    write_case(folder, case7.dispatch().x)
    return load(folder)


def test_unsplit_n1_matches_scipy(opf_case):
    si, network_data, metrics = opf_case
    assert metrics[1] == pytest.approx(20.0, abs=1e-6)
    (n0,), (n1,), _ = matrices(si, unsplit(si))
    lim = np.asarray(si.dynamic_information.branch_limits.max_mw_flow)
    n_out = int(np.asarray(network_data.outaged_branch_mask).sum())
    assert n_out == 11
    assert (100 * np.abs(n1[:n_out]) / lim).max() == pytest.approx(250 / 240 * 100, abs=0.01)
    np.testing.assert_allclose(np.abs(n0), np.abs(case7.flows(case7.dispatch().x)), atol=1e-6)


def test_twelve_single_splits_none_of_which_helps(opf_case):
    si, network_data, _ = opf_case
    per_sub = np.asarray(si.dynamic_information.action_set.n_actions_per_sub)
    np.testing.assert_array_equal(per_sub, [4, 4, 4])
    _, n1_all, _ = matrices(si, single_splits(si))
    lim = np.asarray(si.dynamic_information.branch_limits.max_mw_flow)
    worst = np.nan_to_num((np.abs(n1_all[:, :11]) / lim).max(axis=(1, 2)), nan=np.inf) * 100
    assert worst.min() == pytest.approx(250 / 240 * 100, abs=0.01)
    assert describe(si, network_data, 0).endswith('unsplit')
    assert 'split' in describe(si, network_data, 1)
