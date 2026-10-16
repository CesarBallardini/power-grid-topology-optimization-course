import jax
import jax.numpy as jnp
import pytest
from compile_count import (
    counting,
    exercise_1_traces,
    screen_static,
    screen_then,
    static_traces,
    ten_outage_sets,
    then_traces,
    worst_flow,
)
from n1_screen import ptdf_matrix, screen

FROM, TO = jnp.array([0, 0, 1, 1, 2]), jnp.array([1, 2, 2, 3, 3])
PTDF = ptdf_matrix(FROM, TO, jnp.array([0.1, 0.2, 0.1, 0.2, 0.1]), 4)
P = jnp.array([100.0, 0.0, 0.0, -100.0])


def test_static_tuple_compiles_once_per_distinct_tuple():
    before = static_traces[0]
    sets = ten_outage_sets(5, 2)
    for outages in sets + sets:  # the second round hits the cache
        screen_static(PTDF, FROM, TO, P, outages)
    assert static_traces[0] - before == len(set(sets))


def test_fresh_lambda_recompiles_and_module_function_does_not():
    before = then_traces[0]
    for _ in range(3):
        screen_then(PTDF, FROM, TO, P, jnp.arange(5), post=lambda flows: flows.sum(axis=1))  # noqa: B023
    assert then_traces[0] - before == 3
    before = then_traces[0]
    for _ in range(3):
        screen_then(PTDF, FROM, TO, P, jnp.arange(5), post=worst_flow)
    assert then_traces[0] - before == 1


def test_no_tracing_proves_the_fix_and_catches_a_retrace():
    pair, other_pair, triple = jnp.arange(2), jnp.array([1, 3]), jnp.arange(3)  # built outside the guarded block
    screen(PTDF, FROM, TO, P, pair)
    with jax.no_tracing():
        screen(PTDF, FROM, TO, P, other_pair)  # same shape and dtype: cached
    with pytest.raises(RuntimeError), jax.no_tracing():
        screen(PTDF, FROM, TO, P, triple)  # new shape: would trace


def test_exercise_1_four_traces():
    assert exercise_1_traces() == [1, 1, 2, 3, 4]


def test_counting_wrapper_counts_python_calls():
    wrapped, traces = counting(lambda x: x + 1)
    jitted = jax.jit(wrapped)
    for _ in range(5):
        jitted(jnp.ones(2))
    assert traces[0] == 1
