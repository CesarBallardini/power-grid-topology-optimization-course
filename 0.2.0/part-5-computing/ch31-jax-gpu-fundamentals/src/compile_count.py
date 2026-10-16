"""Counting compilations: static tuples, fresh lambdas, and the fixes (Chapter 31, Lab 4 and Exercise 1).

A Python counter incremented inside a jitted function only runs while JAX traces the function, so it counts
compilations, not calls. ``jax.no_tracing()`` turns an unexpected trace into an error.
"""

import jax
import jax.numpy as jnp
from n1_screen import n1_flows, ptdf_matrix, screen


def counting(fn):
    """Wrap ``fn`` so that every Python-level execution (every trace, under jit) is counted.

    Returns:
        ``(wrapped, traces)`` where ``traces`` is a one-element list holding the count.
    """
    traces = [0]

    def wrapped(*args, **kwargs):
        traces[0] += 1
        return fn(*args, **kwargs)

    return wrapped, traces


# --8<-- [start:static]
def _screen_static(ptdf, from_bus, to_bus, p, outages: tuple):
    """The outage set as a Python tuple marked static: one compilation per distinct tuple."""
    return jax.vmap(n1_flows, in_axes=(None, None, None, None, 0))(ptdf, from_bus, to_bus, p, jnp.array(outages))


screen_static_counted, static_traces = counting(_screen_static)
screen_static = jax.jit(screen_static_counted, static_argnums=4)
# --8<-- [end:static]


# --8<-- [start:lambda]
def _screen_then(ptdf, from_bus, to_bus, p, outages, post):
    """Screen, then apply a post-processing function passed as a static argument."""
    flows, ok = screen(ptdf, from_bus, to_bus, p, outages)
    return post(flows), ok


screen_then_counted, then_traces = counting(_screen_then)
screen_then = jax.jit(screen_then_counted, static_argnames='post')


def worst_flow(flows):
    """Largest absolute flow per outage. Defined once at module level: one function object, one compilation."""
    return jnp.abs(flows).max(axis=1)


# --8<-- [end:lambda]


def ten_outage_sets(n_branch: int, size: int = 4) -> list[tuple[int, ...]]:
    """Ten different outage tuples of the same length."""
    return [tuple(range(k, k + size)) for k in range(10) if k + size <= n_branch] or [tuple(range(size))] * 10


def exercise_1_traces() -> list[int]:
    """Trace count after each of the five calls of Exercise 1: g(x3, 2), g(y3, 2), g(x4, 2), g(x3, 3), g(x3f32, 2)."""

    def f(x, n):
        return (x * n).sum()

    f_counted, traces = counting(f)
    g = jax.jit(f_counted, static_argnums=1)
    x3, y3, x4 = jnp.arange(3.0), jnp.ones(3), jnp.arange(4.0)
    history = []
    for args in ((x3, 2), (y3, 2), (x4, 2), (x3, 3), (x3.astype(jnp.float32), 2)):
        g(*args)
        history.append(traces[0])
    return history


def main() -> None:
    """Count compilations on the four-bus grid, then show the fixed versions trace once."""
    from_bus, to_bus = jnp.array([0, 0, 1, 1, 2]), jnp.array([1, 2, 2, 3, 3])
    ptdf = ptdf_matrix(from_bus, to_bus, jnp.array([0.1, 0.2, 0.1, 0.2, 0.1]), 4)
    p = jnp.array([100.0, 0.0, 0.0, -100.0])
    sets = ten_outage_sets(5, 2)
    for outages in sets:
        screen_static(ptdf, from_bus, to_bus, p, outages)
    print(f'static tuple: {len(sets)} calls with {len(set(sets))} distinct tuples -> {static_traces[0]} traces')
    for _ in range(10):
        screen_then(ptdf, from_bus, to_bus, p, jnp.arange(5), post=lambda flows: jnp.abs(flows).max(axis=1))  # noqa: B023
    print(f'fresh lambda: 10 calls -> {then_traces[0]} traces')
    then_traces[0] = 0
    for _ in range(10):
        screen_then(ptdf, from_bus, to_bus, p, jnp.arange(5), post=worst_flow)
    print(f'module-level function: 10 calls -> {then_traces[0]} trace(s)')
    arrays = [jnp.array(outages) for outages in sets]  # creating arrays is itself a (cached) dispatch: do it outside
    screen(ptdf, from_bus, to_bus, p, arrays[0])  # compiles once for outage arrays of this length
    with jax.no_tracing():  # would raise if any of these calls traced
        for outages in arrays[1:]:
            screen(ptdf, from_bus, to_bus, p, outages)
        screen_then(ptdf, from_bus, to_bus, p, jnp.arange(5), post=worst_flow)
    print('fixed versions: no trace under jax.no_tracing()')
    print('exercise 1, traces after each call:', exercise_1_traces())


if __name__ == '__main__':
    main()
