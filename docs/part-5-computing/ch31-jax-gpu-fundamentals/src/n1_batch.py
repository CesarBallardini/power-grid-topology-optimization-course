"""A second batch axis: N-1 screening of many injection vectors in one compiled call (Chapter 31, Lab 2)."""

import jax
import jax.numpy as jnp
from grid_data import branch_data
from n1_screen import ptdf_matrix, screen

# vmap of the already jitted, already vmapped screening: axis 3 (the injection) gets the batch, the rest is shared
screen_injections = jax.jit(jax.vmap(screen, in_axes=(None, None, None, 0, None)))


def random_injections(key: jax.Array, p: jax.Array, n: int, spread: float = 0.1) -> jax.Array:
    """``n`` perturbed copies of the injection vector, one explicit PRNG key per copy.

    Args:
        key: A ``jax.random`` key; it is split into ``n`` subkeys, so the same key gives the same batch.
        p: The base injection vector (MW).
        n: Number of injection vectors.
        spread: Relative standard deviation of the multiplicative noise.

    Returns:
        Array of shape ``(n, n_bus)``.
    """
    keys = jax.random.split(key, n)
    noise = jax.vmap(lambda k: jax.random.normal(k, p.shape))(keys)
    return p * (1.0 + spread * noise)


def main() -> None:
    """IEEE 118: 100 injection vectors, one call, an array of shape (100, n_branch, n_branch)."""
    from_bus, to_bus, x, p = (jnp.array(v) for v in branch_data('case118'))
    ptdf = ptdf_matrix(from_bus, to_bus, x, len(p))
    injections = random_injections(jax.random.key(0), p, 100)
    flows, ok = screen_injections(ptdf, from_bus, to_bus, injections, jnp.arange(len(x)))
    print('flows', flows.shape, flows.dtype, ' ok', ok.shape)
    print('bridges flagged per injection (the same for all):', int(jnp.sum(~ok[0])), 'to', int(jnp.sum(~ok).max()))
    worst = jnp.abs(jnp.where(ok[:, :, None], flows, 0.0)).max(axis=(1, 2))
    print(f'largest post-contingency flow over the batch: min {worst.min():.1f}, max {worst.max():.1f} MW')
    again = random_injections(jax.random.key(0), p, 100)
    print('same key, same batch:', bool(jnp.array_equal(injections, again)))


if __name__ == '__main__':
    main()
