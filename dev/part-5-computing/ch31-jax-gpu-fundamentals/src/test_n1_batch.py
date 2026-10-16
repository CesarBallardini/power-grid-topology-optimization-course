import jax
import jax.numpy as jnp
import numpy as np
from grid_data import branch_data
from n1_batch import random_injections, screen_injections
from n1_screen import ptdf_matrix, screen


def test_batch_matches_one_injection_at_a_time():
    from_bus, to_bus, x, p = (jnp.array(v) for v in branch_data('case14'))
    ptdf = ptdf_matrix(from_bus, to_bus, x, len(p))
    injections = random_injections(jax.random.key(1), p, 5)
    outages = jnp.arange(len(x))
    flows, ok = screen_injections(ptdf, from_bus, to_bus, injections, outages)
    assert flows.shape == (5, 20, 20) and ok.shape == (5, 20)
    for k in range(5):
        single, ok_single = screen(ptdf, from_bus, to_bus, injections[k], outages)
        np.testing.assert_allclose(flows[k], single, atol=1e-9)
        assert bool(jnp.array_equal(ok[k], ok_single))


def test_random_injections_are_reproducible_and_distinct():
    p = jnp.array([100.0, -50.0, -50.0])
    a = random_injections(jax.random.key(0), p, 4)
    b = random_injections(jax.random.key(0), p, 4)
    assert a.shape == (4, 3) and bool(jnp.array_equal(a, b))
    assert len({tuple(np.asarray(row).round(6)) for row in a}) == 4
