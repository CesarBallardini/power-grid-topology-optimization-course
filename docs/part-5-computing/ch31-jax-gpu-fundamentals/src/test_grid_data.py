import jax.numpy as jnp
import numpy as np
import pytest
from grid_data import branch_data, numpy_ptdf, numpy_screen
from n1_screen import ptdf_matrix, screen


@pytest.mark.parametrize(('case', 'n_bus', 'n_branch'), [('case14', 14, 20), ('case118', 118, 186)])
def test_branch_data_sizes(case, n_bus, n_branch):
    from_bus, to_bus, x, p = branch_data(case)
    assert len(from_bus) == len(to_bus) == len(x) == n_branch and len(p) == n_bus
    assert (x != 0).all() and from_bus.max() < n_bus and to_bus.max() < n_bus


@pytest.mark.parametrize('case', ['case14', 'case118'])
def test_jax_screening_matches_numpy_to_1e9(case):
    from_bus, to_bus, x, p = branch_data(case)
    ptdf_np = numpy_ptdf(from_bus, to_bus, x, len(p))
    flows_np, ok_np = numpy_screen(ptdf_np, from_bus, to_bus, p)
    ptdf = ptdf_matrix(jnp.array(from_bus), jnp.array(to_bus), jnp.array(x), len(p))
    flows, ok = screen(ptdf, jnp.array(from_bus), jnp.array(to_bus), jnp.array(p), jnp.arange(len(x)))
    np.testing.assert_allclose(np.asarray(ptdf), ptdf_np, atol=1e-9)
    np.testing.assert_array_equal(np.asarray(ok), ok_np)
    np.testing.assert_allclose(np.asarray(flows)[ok_np], flows_np[ok_np], atol=1e-9)
