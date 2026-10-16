import numpy as np
from dc_nodal import incidence, nodal_solve


def test_chapter_8_worked_example():
    theta, flows = nodal_solve([1, 1, 2], [0, 2, 0], [1.0, 2.0, 4.0], [0.0, 3.0, 0.0], 3)
    np.testing.assert_allclose(theta, [0.0, 18 / 14, 6 / 14])
    np.testing.assert_allclose(flows, [18 / 14, 2 * 12 / 14, 4 * 6 / 14])


def test_kcl_holds_at_every_node():
    fr, to, b = [0, 0, 1, 1, 2, 3, 4], [1, 2, 2, 3, 4, 4, 5], [10.0, 5.0, 8.0, 4.0, 6.0, 3.0, 2.0]
    inj = np.array([0.0, 0.0, 0.5, 0.0, 0.0, -1.0])
    theta, flows = nodal_solve(fr, to, b, inj, 6)
    a = incidence(np.array(fr), np.array(to), 6)
    net = a.T @ flows
    np.testing.assert_allclose(net[1:], inj[1:], atol=1e-12)
    np.testing.assert_allclose(net[0], 0.5, atol=1e-12)  # the ground supplies the balance
    assert theta[0] == 0
