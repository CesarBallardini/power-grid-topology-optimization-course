import numpy as np
from node_reduction import BRANCHES, N_BUS, P_REF, SIGNIFICANT, ptdf_matrix, reduce_ptdf


def test_reduced_ptdf_is_exact_at_the_reference():
    h = ptdf_matrix(BRANCHES, N_BUS)
    h_red, p_red = reduce_ptdf(h, P_REF, SIGNIFICANT)
    assert h_red.shape == (len(BRANCHES), len(SIGNIFICANT) + 1)
    np.testing.assert_allclose(h_red @ p_red, h @ P_REF, atol=1e-9)


def test_change_at_a_kept_bus_is_exact_and_at_a_collapsed_bus_is_not():
    h = ptdf_matrix(BRANCHES, N_BUS)
    h_red, _ = reduce_ptdf(h, P_REF, SIGNIFICANT)
    p = P_REF.copy()
    p[2] += 20.0
    p[0] -= 20.0
    np.testing.assert_allclose(h_red @ np.append(p[SIGNIFICANT], 1.0), h @ p, atol=1e-9)
    p = P_REF.copy()
    p[3] -= 20.0
    p[0] += 20.0
    assert np.abs(h_red @ np.append(p[SIGNIFICANT], 1.0) - h @ p).max() > 1.0


def test_injections_balance_and_slack_column_is_zero():
    assert P_REF.sum() == 0.0
    h = ptdf_matrix(BRANCHES, N_BUS)
    np.testing.assert_allclose(h[:, 0], 0.0)
