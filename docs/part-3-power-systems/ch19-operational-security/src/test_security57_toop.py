"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
import metrics  # noqa: E402
from security57_toop import (  # noqa: E402
    METRICS,
    limits,
    load,
    matrices,
    numpy_metrics,
    regimes,
    toop_metric,
    with_actions,
)
from toop_engine_dc_solver.jax import default_topology  # noqa: E402


@pytest.fixture(scope='module')
def case57(tmp_path_factory):
    stats, si, unsplit = load(tmp_path_factory.mktemp('case57'))
    return stats, si, unsplit


def test_unsplit_energies_match_run_initial_loadflow(case57):
    stats, si, unsplit = case57
    assert stats.n_branches == 82 and stats.n_nminus1_cases == 84 and stats.has_double_limits
    n0, n1, ok = matrices(si, default_topology(si.solver_config, batch_size=4))
    assert n0.shape == (4, 1, 82) and n1.shape == (4, 1, 84, 82) and ok.all()
    mine = numpy_metrics(n0[0], n1[0], limits(si))
    assert mine['overload_energy_n_0'] == pytest.approx(unsplit[0], abs=1e-6)
    assert mine['overload_energy_n_1'] == pytest.approx(unsplit[1], abs=1e-6)
    assert unsplit[1] == pytest.approx(3249.796, abs=1e-3)


def test_double_limits_match_branch_limits(case57):
    _, si, _ = case57
    lim = limits(si)
    n0, n1, _ = matrices(si, default_topology(si.solver_config, batch_size=1))
    np.testing.assert_allclose(
        metrics.double_limits(n0[0][:, None, :], lim['max_mw_flow']), lim['max_mw_flow_limited']
    )
    np.testing.assert_allclose(metrics.double_limits(n1[0], lim['max_mw_flow_n_1']), lim['max_mw_flow_n_1_limited'])
    counts = regimes(n1[0], lim['max_mw_flow_n_1'], lim['max_mw_flow_n_1_limited'])
    assert sum(counts.values()) == 82


@pytest.mark.parametrize('metric', METRICS)
def test_every_metric_matches_aggregate_to_metric(case57, metric):
    _, si, _ = case57
    topologies = with_actions(si, {1: 1, 2: 5})
    n0, n1, _ = matrices(si, topologies)
    toop = toop_metric(si, topologies, metric)
    mine = [numpy_metrics(n0[t], n1[t], limits(si))[metric] for t in range(4)]
    np.testing.assert_allclose(toop, mine, atol=1e-6)
    assert toop[0] == toop[3]  # topologies 0 and 3 stay unsplit
