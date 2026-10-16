import jax
import jax.numpy as jnp
import numpy as np
from control_flow import (
    clip_negative_cond,
    clip_negative_python,
    clip_negative_where,
    error_under_jit,
    gather_fill,
    pad_indices,
    safe_sqrt_cond,
    safe_sqrt_where,
)

X = jnp.array([-1.0, 2.0, -3.0, 5.0])


def test_python_if_works_eagerly_and_fails_under_jit():
    np.testing.assert_array_equal(clip_negative_python(X), [0.0, 2.0, 0.0, 5.0])
    assert error_under_jit(X) == 'TracerBoolConversionError'


def test_where_and_cond_agree_under_jit():
    for x in (X, -X):
        expected = clip_negative_python(x)
        np.testing.assert_array_equal(jax.jit(clip_negative_where)(x), expected)
        np.testing.assert_array_equal(jax.jit(clip_negative_cond)(x), expected)


def test_where_evaluates_the_untaken_branch():
    assert jnp.isnan(jax.grad(safe_sqrt_where)(-1.0))
    assert jax.grad(safe_sqrt_cond)(-1.0) == 0.0
    assert jax.grad(safe_sqrt_where)(4.0) == jax.grad(safe_sqrt_cond)(4.0) == 0.25


def test_sentinel_equal_to_length_is_dropped_but_minus_one_wraps():
    values = jnp.array([10.0, 20.0, 30.0, 40.0])
    groups = [[0, 1, 2], [3], [1, 3]]
    good = gather_fill(values, pad_indices(groups, sentinel=4))
    bad = gather_fill(values, pad_indices(groups, sentinel=-1))
    np.testing.assert_array_equal(good.sum(axis=1), [60.0, 40.0, 60.0])
    np.testing.assert_array_equal(bad.sum(axis=1), [60.0, 120.0, 100.0])  # every pad slot read values[-1]
    assert pad_indices(groups, sentinel=4).shape == (3, 3)
