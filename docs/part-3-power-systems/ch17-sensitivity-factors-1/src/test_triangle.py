import numpy as np
from triangle import FRM, TO, lodf, ptdf


def test_ptdf_by_hand():
    h = ptdf()
    np.testing.assert_allclose(h[:, 2], 0)  # slack column
    np.testing.assert_allclose(h[:, 0], [1 / 3, 2 / 3, 1 / 3])  # inject at bus 1: detour 1-2-3 takes a third
    np.testing.assert_allclose(h[:, 1], [-1 / 3, 1 / 3, 2 / 3])  # inject at bus 2: detour 2-1-3 takes a third


def test_lodf_by_hand_and_current_divider():
    expected = np.array([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    np.testing.assert_allclose(lodf(), expected, atol=1e-12)
    h = ptdf()
    np.testing.assert_allclose(np.diag(h[:, FRM] - h[:, TO]), 2 / 3)  # h_kk = x_other / (x_k + x_other) = 2/3
