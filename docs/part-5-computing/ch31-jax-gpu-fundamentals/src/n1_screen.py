"""N-1 screening of every branch outage at once with jit and vmap, with no exceptions inside (Chapter 31)."""

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)  # before creating any array


def ptdf_matrix(from_bus: jnp.ndarray, to_bus: jnp.ndarray, x: jnp.ndarray, n_bus: int, slack: int = 0):
    """PTDF (branches x buses) with the slack column at zero."""
    m = len(from_bus)
    a = jnp.zeros((m, n_bus)).at[jnp.arange(m), from_bus].set(1.0).at[jnp.arange(m), to_bus].set(-1.0)
    bf = a / x[:, None]
    bbus = a.T @ bf
    keep = jnp.arange(n_bus) != slack
    sol = jnp.linalg.solve(bbus[1:, 1:], bf[:, 1:].T).T  # slack 0 is the first bus
    return jnp.where(keep[None, :], jnp.concatenate([jnp.zeros((m, 1)), sol], axis=1), 0.0)


def n1_flows(ptdf, from_bus, to_bus, p, outage):
    """Flows after the outage of one branch, plus a flag that is False when the outage islands the grid."""
    f0 = ptdf @ p
    col = ptdf[:, from_bus[outage]] - ptdf[:, to_bus[outage]]
    denom = 1.0 - col[outage]  # zero exactly for a bridge
    lodf = jnp.where(jnp.arange(ptdf.shape[0]) == outage, -1.0, col / denom)
    return f0 + lodf * f0[outage], jnp.abs(denom) > 1e-11  # flows and a success flag, no exception


screen = jax.jit(jax.vmap(n1_flows, in_axes=(None, None, None, None, 0)))


def main() -> None:
    """Screen a four-bus grid: the N-1 flow matrix has one row per outage."""
    from_bus, to_bus = jnp.array([0, 0, 1, 1, 2]), jnp.array([1, 2, 2, 3, 3])
    x = jnp.array([0.1, 0.2, 0.1, 0.2, 0.1])
    p = jnp.array([100.0, 0.0, 0.0, -100.0])
    ptdf = ptdf_matrix(from_bus, to_bus, x, 4)
    flows, ok = screen(ptdf, from_bus, to_bus, p, jnp.arange(len(from_bus)))  # shape (n_branch, n_branch)
    print(jnp.round(flows, 1))
    print(ok)


if __name__ == '__main__':
    main()
