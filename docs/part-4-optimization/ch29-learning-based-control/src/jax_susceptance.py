"""Derivative of the DC flows with respect to the susceptances, by JAX and by finite differences (Chapter 29, Ex. 3).

Extends ``jax_ptdf.py`` (same three-bus ring): the flows become a function of the susceptance vector too, and
``jax.jacfwd`` with ``argnums=1`` gives the (line, susceptance) Jacobian. Run it with::

    python jax_susceptance.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_ptdf import A, B

jax.config.update('jax_enable_x64', True)

P = jnp.array([-1.0, 0.5])  # injections at buses 1 and 2, as in the worked example


# --8<-- [start:grad]
def flows_b(p_noslack: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Branch flows for the injections ``p_noslack`` when the lines have susceptances ``b``."""
    b_f = b[:, None] * A
    b_bus = A.T @ b_f
    theta = jnp.concatenate([jnp.zeros(1), jnp.linalg.solve(b_bus[1:, 1:], p_noslack)])
    return b_f @ theta


def dflows_db(p_noslack: jnp.ndarray = P, b: jnp.ndarray = B) -> jnp.ndarray:
    """Jacobian of the flows with respect to the susceptances, shape (line, line), forward mode."""
    return jax.jacfwd(flows_b, argnums=1)(p_noslack, b)


def finite_difference_column(k: int, h: float = 1e-6, p_noslack: jnp.ndarray = P, b: jnp.ndarray = B) -> np.ndarray:
    """Central difference of the flows with respect to ``b[k]``: one column of the Jacobian."""
    e = jnp.zeros(len(b)).at[k].set(h)
    return np.asarray((flows_b(p_noslack, b + e) - flows_b(p_noslack, b - e)) / (2 * h))


# --8<-- [end:grad]
def main() -> None:
    """The Jacobian, its finite-difference check and the operator reading."""
    j = dflows_db()
    print('flows:', np.round(np.asarray(flows_b(P, B)), 4))
    print('d flows / d b (rows: lines 0-1, 1-2, 0-2; columns: b of the same lines):')
    print(np.round(np.asarray(j), 4))
    k = 2
    fd = finite_difference_column(k)
    gap = np.abs(fd - np.asarray(j[:, k])).max()
    print(f'column {k} by finite differences: {np.round(fd, 4)}, max |difference| {gap:.2e}')
    print(
        'row sums (a uniform scaling of all susceptances leaves the flows unchanged):', np.round(np.asarray(j @ B), 10)
    )


if __name__ == '__main__':
    main()
