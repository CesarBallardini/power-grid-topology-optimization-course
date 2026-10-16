import jax.numpy as jnp
import numpy as np
from n1_screen import ptdf_matrix, screen

FROM, TO = jnp.array([0, 0, 1, 1, 2]), jnp.array([1, 2, 2, 3, 3])
X = jnp.array([0.1, 0.2, 0.1, 0.2, 0.1])
P = jnp.array([100.0, 0.0, 0.0, -100.0])


def test_screen_matches_resolving_each_outage():
    ptdf = ptdf_matrix(FROM, TO, X, 4)
    flows, ok = screen(ptdf, FROM, TO, P, jnp.arange(5))
    assert flows.shape == (5, 5) and bool(ok.all())
    for k in range(5):
        keep = np.arange(5) != k
        ref = np.zeros(5)
        ref[keep] = np.asarray(ptdf_matrix(FROM[keep], TO[keep], X[keep], 4) @ P)
        np.testing.assert_allclose(flows[k], ref, atol=1e-9)


def test_bridge_is_flagged_not_raised():
    from_bus, to_bus, x = jnp.array([0, 1]), jnp.array([1, 2]), jnp.array([0.1, 0.1])
    ptdf = ptdf_matrix(from_bus, to_bus, x, 3)
    _, ok = screen(ptdf, from_bus, to_bus, jnp.array([1.0, 0.0, -1.0]), jnp.arange(2))
    assert not bool(ok.any())
