"""DC flows of a three-bus ring in JAX, and the PTDF as its automatic derivative (Chapter 29)."""

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

LINES = jnp.array([[0, 1], [1, 2], [0, 2]])  # 3-bus ring, bus 0 is the slack
B = jnp.array([10.0, 5.0, 8.0])  # line susceptances
A = jnp.zeros((3, 3)).at[jnp.arange(3), LINES[:, 0]].set(1.0).at[jnp.arange(3), LINES[:, 1]].set(-1.0)
B_F = B[:, None] * A  # branch flows = B_F @ theta
B_BUS = A.T @ B_F  # nodal susceptance matrix


def flows(p_noslack: jnp.ndarray) -> jnp.ndarray:
    """Branch flows for the injections at buses 1 and 2 (the slack balances them)."""
    theta = jnp.concatenate([jnp.zeros(1), jnp.linalg.solve(B_BUS[1:, 1:], p_noslack)])
    return B_F @ theta


def ptdf() -> jnp.ndarray:
    """Columns of the PTDF for buses 1 and 2: the Jacobian of ``flows``, by forward-mode differentiation."""
    return jax.jacfwd(flows)(jnp.zeros(2))


def main() -> None:
    """Flows for one injection pattern and the PTDF."""
    p = jnp.array([-1.0, 0.5])  # injections at buses 1 and 2
    print(flows(p))  # [ 0.6176 -0.3824 -0.1176]
    print(ptdf())  # [[-0.7647 -0.2941] [0.2353 -0.2941] [-0.2353 -0.7059]]


if __name__ == '__main__':
    main()
