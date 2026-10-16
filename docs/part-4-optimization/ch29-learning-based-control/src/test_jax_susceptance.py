import jax.numpy as jnp
import numpy as np
import pytest
from jax_ptdf import B, flows
from jax_susceptance import P, dflows_db, finite_difference_column, flows_b


def test_same_flows_as_the_worked_example():
    np.testing.assert_allclose(flows_b(P, B), flows(P), atol=1e-12)


@pytest.mark.parametrize('k', [0, 1, 2])
def test_jacobian_matches_finite_differences(k):
    j = np.asarray(dflows_db())
    np.testing.assert_allclose(j[:, k], finite_difference_column(k), atol=1e-7)


def test_homogeneity():
    # flows are invariant to a common scaling of all susceptances, so J b = 0 (Euler's theorem, degree zero)
    np.testing.assert_allclose(np.asarray(dflows_db()) @ np.asarray(B), 0, atol=1e-12)
    np.testing.assert_allclose(flows_b(P, 3 * B), flows_b(P, B), atol=1e-12)


def test_kirchhoff_is_preserved_by_every_derivative():
    # a change of one susceptance moves flow around the loop only: the derivative of every nodal balance is zero
    from jax_ptdf import A  # noqa: PLC0415

    j = np.asarray(dflows_db())
    np.testing.assert_allclose(np.asarray(A).T @ j, 0, atol=1e-12)
    assert jnp.asarray(j).shape == (3, 3)
