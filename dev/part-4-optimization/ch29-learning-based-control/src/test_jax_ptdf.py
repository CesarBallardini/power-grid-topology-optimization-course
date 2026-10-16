import jax.numpy as jnp
import numpy as np
from jax_ptdf import A, flows, ptdf


def test_flows_and_ptdf_values():
    p = jnp.array([-1.0, 0.5])
    np.testing.assert_allclose(flows(p), [0.6176, -0.3824, -0.1176], atol=5e-5)
    h = ptdf()
    np.testing.assert_allclose(h, [[-0.7647, -0.2941], [0.2353, -0.2941], [-0.2353, -0.7059]], atol=5e-5)
    np.testing.assert_allclose(h @ p, flows(p), atol=1e-12)  # flows are linear in the injections


def test_kirchhoff_holds():
    p = jnp.array([-1.0, 0.5])
    f = flows(p)
    np.testing.assert_allclose((A.T @ f)[1:], p, atol=1e-12)
