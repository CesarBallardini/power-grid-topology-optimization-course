import jax.numpy as jnp
import numpy as np
from grid_data import branch_data, numpy_ptdf, numpy_screen
from lodf_precision import lodf_matrix, precision_study


def test_lodf_matrix_reproduces_the_numpy_screening_on_case14():
    from_bus, to_bus, x, p = branch_data('case14')
    ptdf = numpy_ptdf(from_bus, to_bus, x, len(p))
    lodf, denom = lodf_matrix(jnp.array(ptdf), jnp.array(from_bus), jnp.array(to_bus))
    flows_np, ok = numpy_screen(ptdf, from_bus, to_bus, p)
    f0 = ptdf @ p
    flows = (f0[:, None] + np.asarray(lodf) * f0[None, :]).T  # row k: outage k
    np.testing.assert_allclose(flows[ok], flows_np[ok], atol=1e-9)
    assert (np.abs(np.asarray(denom)) > 1e-11).tolist() == ok.tolist()
    np.testing.assert_array_equal(np.diagonal(np.asarray(lodf)), -1.0)


def test_float32_error_grows_as_the_denominator_shrinks():
    study = precision_study('case118')
    ok, denom, error = study['ok'], study['abs_denom'], study['error']
    assert len(ok) == 186 and 0 < (~ok).sum() < 20
    assert (error[ok] > 0).all() and error[ok].max() < 1e-2
    small, large = error[ok][denom[ok] < 0.2], error[ok][denom[ok] >= 0.5]
    assert np.median(small) > np.median(large)
