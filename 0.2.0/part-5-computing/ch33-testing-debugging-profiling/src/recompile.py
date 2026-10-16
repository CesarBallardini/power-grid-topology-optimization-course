"""Planted bug (d): a fresh ``lambda`` on every call makes ``jax.jit`` trace and compile again every time (Chapter 33).

ToOp's ``run_solver_symmetric`` takes ``aggregate_output_fn`` as a static argument of ``jax.jit``, so the cache key
includes the function object: a lambda built inside a loop is a new object each time, and the compiled program
is never reused. A module-level function keeps one entry in the cache.
"""

import time

import jax
import jax.numpy as jnp

TRACES = {'count': 0}


def aggregate(flows, limits):
    """Overload energy of a batch of flows; counts how often JAX traces it."""
    TRACES['count'] += 1
    return jnp.maximum(jnp.abs(flows) - limits, 0.0).sum(axis=-1)


def solve(flows, limits, aggregate_fn):
    """Stand-in for ``run_solver_symmetric``: the aggregation function is static, so it is part of the cache key."""
    return jax.jit(aggregate_fn)(flows, limits)


def run(n_calls, fresh_lambda, batch=256, n_branches=2000):
    """Call ``solve`` ``n_calls`` times; returns the number of traces and the wall-clock seconds per call."""
    key = jax.random.PRNGKey(0)
    flows = jax.random.normal(key, (batch, n_branches))
    limits = jnp.ones(n_branches)
    TRACES['count'] = 0
    seconds = []
    for _ in range(n_calls):
        fn = (lambda f, lim: aggregate(f, lim)) if fresh_lambda else aggregate  # noqa: E731
        start = time.perf_counter()
        jax.block_until_ready(solve(flows, limits, fn))
        seconds.append(time.perf_counter() - start)
    return TRACES['count'], seconds


def main():
    """Print traces and timings for both ways of passing the aggregation function (CPU numbers of this machine)."""
    for fresh in (True, False):
        traces, seconds = run(5, fresh)
        label = 'new lambda per call ' if fresh else 'module-level function'
        print(f'{label}: {traces} traces in 5 calls, ms per call {[round(1000 * s, 1) for s in seconds]}')


if __name__ == '__main__':
    main()
