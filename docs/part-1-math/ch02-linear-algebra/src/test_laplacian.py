import numpy as np
from laplacian import fr, g, incidence, laplacian, nodal_voltages, to


def test_rows_of_incidence_sum_to_zero():
    a = incidence(fr, to, 4)
    assert a.shape == (4, 4)
    np.testing.assert_array_equal(a.sum(axis=1), 0)


def test_laplacian_is_symmetric_with_one_zero_eigenvalue():
    gmat = laplacian(incidence(fr, to, 4), g)
    np.testing.assert_allclose(gmat, gmat.T)
    w = np.linalg.eigvalsh(gmat)
    np.testing.assert_allclose(w, [0.0, 0.6106, 3.091, 5.2984], atol=5e-5)
    np.testing.assert_allclose(gmat @ np.ones(4), 0, atol=1e-12)


def test_nodal_solve_satisfies_kirchhoff():
    a = incidence(fr, to, 4)
    gmat = laplacian(a, g)
    i = np.array([1.0, 0.0, 0.0, -1.0])
    v = nodal_voltages(gmat, i)
    assert v[0] == 0
    np.testing.assert_allclose(gmat @ v, i, atol=1e-12)  # KCL at every node, ground included
    np.testing.assert_allclose(a.T @ (g * (a @ v)), i, atol=1e-12)  # branch currents add up to the injections
