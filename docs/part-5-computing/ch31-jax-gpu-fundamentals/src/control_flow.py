"""Control flow under jit and gathers with a padding sentinel (Chapter 31, Exercises 2 and 3)."""

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_enable_x64', True)


# --8<-- [start:clip]
def clip_negative_python(x):
    """Python ``if`` on a traced value: fine eagerly, TracerBoolConversionError under jit."""
    if x.sum() > 0:
        return jnp.maximum(x, 0.0)
    return x


def clip_negative_where(x):
    """Both branches are computed on every call; ``where`` picks per element."""
    return jnp.where(x.sum() > 0, jnp.maximum(x, 0.0), x)


def clip_negative_cond(x):
    """One branch is executed at run time (outside vmap); the predicate is a scalar."""
    return lax.cond(x.sum() > 0, lambda v: jnp.maximum(v, 0.0), lambda v: v, x)


# --8<-- [end:clip]


def error_under_jit(x) -> str:
    """Name of the exception that ``jit(clip_negative_python)`` raises, or ``'no error'``."""
    try:
        jax.jit(clip_negative_python)(x)
    except jax.errors.TracerBoolConversionError as exc:
        return type(exc).__name__
    return 'no error'


# --8<-- [start:nan]
def safe_sqrt_where(x):
    """Square root where positive, else 0: the gradient at a negative x is NaN, since sqrt'(x) is evaluated there."""
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)


def safe_sqrt_cond(x):
    """The same with cond: the untaken branch is neither evaluated nor differentiated."""
    return lax.cond(x > 0, jnp.sqrt, lambda v: 0.0 * v, x)


# --8<-- [end:nan]


# --8<-- [start:pad]
def pad_indices(groups: list[list[int]], sentinel: int) -> jnp.ndarray:
    """Variable-length index lists as one rectangular int array, unused slots filled with ``sentinel``."""
    width = max(len(g) for g in groups)
    return jnp.array([list(g) + [sentinel] * (width - len(g)) for g in groups])


def gather_fill(x, idx):
    """Gather with out-of-bounds slots replaced by 0.0; valid sentinels are >= len(x), never negative."""
    return x.at[idx].get(mode='fill', fill_value=0.0)


# --8<-- [end:pad]


def main() -> None:
    """Run the three clip versions, the NaN gradient, and the two sentinels."""
    x = jnp.array([-1.0, 2.0, -3.0, 5.0])
    print('eager Python if   :', clip_negative_python(x))
    print('under jit         :', error_under_jit(x))
    print('where under jit   :', jax.jit(clip_negative_where)(x))
    print('cond under jit    :', jax.jit(clip_negative_cond)(x))
    print('grad where at -1  :', jax.grad(safe_sqrt_where)(-1.0))
    print('grad cond at -1   :', jax.grad(safe_sqrt_cond)(-1.0))
    print(
        'vmap of cond jaxpr uses select_n:', 'select_n' in str(jax.make_jaxpr(jax.vmap(clip_negative_cond))(x[None]))
    )
    values = jnp.array([10.0, 20.0, 30.0, 40.0])
    groups = [[0, 1, 2], [3], [1, 3]]
    idx_ok = pad_indices(groups, sentinel=len(values))
    idx_bad = pad_indices(groups, sentinel=-1)
    print('sentinel = len(x):', idx_ok.tolist(), '->', gather_fill(values, idx_ok).tolist())
    print('sentinel = -1    :', idx_bad.tolist(), '->', gather_fill(values, idx_bad).tolist())
    print('row sums, len(x) sentinel:', gather_fill(values, idx_ok).sum(axis=1).tolist())
    print('row sums, -1 sentinel    :', gather_fill(values, idx_bad).sum(axis=1).tolist())


if __name__ == '__main__':
    main()
