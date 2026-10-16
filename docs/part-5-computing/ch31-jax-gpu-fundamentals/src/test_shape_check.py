import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('jaxtyping')
pytest.importorskip('beartype')

from n1_screen import n1_flows, ptdf_matrix  # noqa: E402
from shape_check import call_with_mismatch, n1_flows_checked  # noqa: E402

FROM, TO = jnp.array([0, 0, 1, 1, 2]), jnp.array([1, 2, 2, 3, 3])
PTDF = ptdf_matrix(FROM, TO, jnp.array([0.1, 0.2, 0.1, 0.2, 0.1]), 4)
P = jnp.array([100.0, 0.0, 0.0, -100.0])


def test_checked_version_matches_the_plain_one():
    for k in range(5):
        flows, ok = n1_flows_checked(PTDF, FROM, TO, P, jnp.array(k))
        plain, ok_plain = n1_flows(PTDF, FROM, TO, P, jnp.array(k))
        np.testing.assert_allclose(flows, plain, atol=1e-12)
        assert bool(ok) == bool(ok_plain)


def test_mismatch_is_caught_eagerly_and_while_tracing():
    assert call_with_mismatch(jit=False) == 'TypeCheckError'
    assert call_with_mismatch(jit=True) == 'TypeCheckError'
