"""A data-dependent loop count without recompiling: fori_loop with a traced bound against a Python for (Exercise 3).

ToOp's ``mutate.py`` draws the number of substation mutations from a Poisson distribution, clips it to
``[1, max_num_splits]`` and uses it as the upper bound of ``jax.lax.fori_loop``; the toy here does the same with a
step that adds one entry of a table per iteration.
"""

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_enable_x64', True)

TRACES = {'fori': 0, 'python': 0}
MAX_STEPS = 8


def draw_steps(key: jax.Array, lam: float = 2.0) -> jax.Array:
    """Poisson draw clipped to [1, MAX_STEPS], as mutate.py clips n_subs_mutated."""
    return jnp.clip(jax.random.poisson(key, lam, shape=()), 1, MAX_STEPS)


# --8<-- [start:loops]
def _apply_fori(table, n_steps, start):
    """n_steps may be a tracer: fori_loop with a traced upper bound lowers to while_loop, compiled once."""
    TRACES['fori'] += 1

    def body(i, acc):
        return acc + table[start + i]

    return lax.fori_loop(0, n_steps, body, 0.0)


def _apply_python(table, n_steps, start):
    """n_steps must be a Python int: the loop is unrolled while tracing, once per distinct value."""
    TRACES['python'] += 1
    acc = 0.0
    for i in range(n_steps):
        acc = acc + table[start + i]
    return acc


apply_fori = jax.jit(_apply_fori)
apply_python = jax.jit(_apply_python, static_argnums=1)
# --8<-- [end:loops]


def run_both(table, draws: list[int]) -> tuple[int, int]:
    """Call both versions once per draw; return how many times each one traced."""
    before = TRACES['fori'], TRACES['python']
    for n in draws:
        apply_fori(table, jnp.asarray(n), 0)
        apply_python(table, int(n), 0)
    return TRACES['fori'] - before[0], TRACES['python'] - before[1]


def main() -> None:
    """Twenty Poisson draws: one trace for fori_loop, one per distinct value for the Python loop."""
    table = jnp.arange(1.0, 2 * MAX_STEPS + 1.0)
    draws = [int(draw_steps(k)) for k in jax.random.split(jax.random.key(0), 20)]
    print('draws:', draws)
    fori, python = run_both(table, draws)
    print(f'traces: fori_loop {fori}, Python for {python} (distinct draws: {len(set(draws))})')
    print(
        'results agree:',
        all(float(apply_fori(table, jnp.asarray(n), 0)) == float(apply_python(table, n, 0)) for n in draws),
    )
    print('--- jaxpr, fori_loop with a traced bound:')
    print(jax.make_jaxpr(_apply_fori)(table, jnp.asarray(3), 0))
    print('--- jaxpr, Python for with n_steps = 3:')
    print(jax.make_jaxpr(_apply_python, static_argnums=1)(table, 3, 0))
    try:
        jax.jit(_apply_python)(table, jnp.asarray(3), 0)
    except jax.errors.TracerIntegerConversionError as exc:
        print('Python for with a traced bound:', type(exc).__name__)


if __name__ == '__main__':
    main()
