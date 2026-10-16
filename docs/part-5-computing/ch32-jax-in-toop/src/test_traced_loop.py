import jax
import jax.numpy as jnp
import pytest
from traced_loop import MAX_STEPS, _apply_fori, _apply_python, apply_fori, apply_python, draw_steps, run_both

TABLE = jnp.arange(1.0, 2 * MAX_STEPS + 1.0)


def test_draws_stay_in_range():
    draws = [int(draw_steps(k)) for k in jax.random.split(jax.random.key(0), 50)]
    assert min(draws) >= 1 and max(draws) <= MAX_STEPS and len(set(draws)) > 1


def test_fori_traces_once_and_python_once_per_distinct_bound():
    draws = [3, 1, 3, 5, 1, 2]
    fori, python = run_both(TABLE, draws)
    assert fori <= 1 and python == len(set(draws))
    for n in draws:
        assert float(apply_fori(TABLE, jnp.asarray(n), 0)) == float(apply_python(TABLE, n, 0)) == sum(range(1, n + 1))


def test_jaxprs_differ_in_kind():
    fori = str(jax.make_jaxpr(_apply_fori)(TABLE, jnp.asarray(3), 0))
    python = str(jax.make_jaxpr(_apply_python, static_argnums=1)(TABLE, 3, 0))
    assert 'while' in fori and 'while' not in python
    assert python.count('dynamic_slice') == 3 or python.count('gather') == 3  # three unrolled table reads


def test_python_loop_rejects_a_traced_bound():
    with pytest.raises(jax.errors.TracerIntegerConversionError):  # range() asks the tracer for an int
        jax.jit(_apply_python)(TABLE, jnp.asarray(3), 0)
