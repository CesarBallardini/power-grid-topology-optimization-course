"""Shape annotations checked at run time with jaxtyping and beartype (Chapter 31, Exercise 6).

jaxtyping and beartype are in ToOp's environment, not in the course one: run this file with ToOp's interpreter.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from n1_screen import ptdf_matrix

jax.config.update('jax_enable_x64', True)


# --8<-- [start:annotated]
# ruff reads the shape strings as forward references and reports F722; ToOp ignores that rule repository-wide
@jaxtyped(typechecker=beartype)
def n1_flows_checked(
    ptdf: Float[Array, ' n_branch n_bus'],  # noqa: F722
    from_bus: Int[Array, ' n_branch'],  # noqa: F722
    to_bus: Int[Array, ' n_branch'],  # noqa: F722
    p: Float[Array, ' n_bus'],  # noqa: F722
    outage: Int[Array, ''],  # noqa: F722
) -> tuple[Float[Array, ' n_branch'], Bool[Array, '']]:  # noqa: F722
    """The worked example's ``n1_flows`` with every shape named; the same name must mean the same size."""
    f0 = ptdf @ p
    col = ptdf[:, from_bus[outage]] - ptdf[:, to_bus[outage]]
    denom = 1.0 - col[outage]
    lodf = jnp.where(jnp.arange(ptdf.shape[0]) == outage, -1.0, col / denom)
    return f0 + lodf * f0[outage], jnp.abs(denom) > 1e-11


# --8<-- [end:annotated]


def call_with_mismatch(jit: bool) -> str:
    """Call the checked function with a PTDF of 4 columns and an injection vector of 5: the error's name."""
    from_bus, to_bus = jnp.array([0, 0, 1, 1, 2]), jnp.array([1, 2, 2, 3, 3])
    ptdf = ptdf_matrix(from_bus, to_bus, jnp.array([0.1, 0.2, 0.1, 0.2, 0.1]), 4)
    p_wrong = jnp.array([100.0, 0.0, 0.0, -50.0, -50.0])
    fn = jax.jit(n1_flows_checked) if jit else n1_flows_checked
    try:
        fn(ptdf, from_bus, to_bus, p_wrong, jnp.array(0))
    except Exception as exc:  # noqa: BLE001 - the exercise asks which exception appears
        return type(exc).__name__
    return 'no error'


def main() -> None:
    """A valid call, then the mismatch eagerly and under jit."""
    from_bus, to_bus = jnp.array([0, 0, 1, 1, 2]), jnp.array([1, 2, 2, 3, 3])
    ptdf = ptdf_matrix(from_bus, to_bus, jnp.array([0.1, 0.2, 0.1, 0.2, 0.1]), 4)
    flows, ok = n1_flows_checked(ptdf, from_bus, to_bus, jnp.array([100.0, 0.0, 0.0, -100.0]), jnp.array(0))
    print('valid call:', jnp.round(flows, 1), bool(ok))
    print('mismatch, eager   :', call_with_mismatch(jit=False))
    print('mismatch, under jit:', call_with_mismatch(jit=True))


if __name__ == '__main__':
    main()
