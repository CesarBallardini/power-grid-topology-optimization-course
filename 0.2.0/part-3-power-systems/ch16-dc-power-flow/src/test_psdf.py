import numpy as np
import pytest
from dc_power_flow import dc_power_flow
from psdf import flows_with_shift, psdf_matrix, ptdf_from_solves

FRM, TO = np.array([0, 0, 1, 2]), np.array([1, 2, 3, 3])
X = np.array([0.1, 0.2, 0.1, 0.2])
P = np.array([0.8, 0.2, -0.5, -0.5])


def test_single_ring_psdf_is_one_over_the_loop_reactance():
    psdf = psdf_matrix(ptdf_from_solves(FRM, TO, X, 4, 0), FRM, TO, X)
    np.testing.assert_allclose(np.abs(psdf), 1 / X.sum(), atol=1e-12)


@pytest.mark.parametrize('k', range(4))
def test_psdf_column_predicts_the_shifted_power_flow(k):
    h = ptdf_from_solves(FRM, TO, X, 4, 0)
    psdf = psdf_matrix(h, FRM, TO, X)
    phi = np.zeros(4)
    phi[k] = np.deg2rad(7.0)
    predicted = dc_power_flow(FRM, TO, X, P, 0) + psdf @ phi
    np.testing.assert_allclose(predicted, flows_with_shift(FRM, TO, X, P, 0, phi), atol=1e-12)


def test_ptdf_from_solves_does_not_depend_on_the_slack_for_balanced_injections():
    h0, h3 = ptdf_from_solves(FRM, TO, X, 4, 0), ptdf_from_solves(FRM, TO, X, 4, 3)
    np.testing.assert_allclose(h0 @ P, h3 @ P, atol=1e-12)
    assert np.abs(h0 - h3).max() > 0.1  # the columns themselves differ
