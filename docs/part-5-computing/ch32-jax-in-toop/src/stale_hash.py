"""The stale-result trap of identity-hashed static arguments, reproduced without ToOp (Chapter 32, Exercise 2).

``IdentityConfig`` hashes and compares by ``id``, as ToOp's ``SolverConfig`` does; ``FrozenConfig`` is the safer
design, a frozen dataclass whose hash follows its values.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

TRACES = {'identity': 0, 'frozen': 0}


# --8<-- [start:configs]
@dataclass
class IdentityConfig:
    """Static configuration hashed by object identity: the compile cache cannot see a mutated field."""

    n_worst: int

    def __hash__(self) -> int:
        """Identity, like ToOp's SolverConfig.__hash__."""
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Identity, like ToOp's SolverConfig.__eq__."""
        return self is other


@dataclass(frozen=True)
class FrozenConfig:
    """Static configuration hashed by value: a new value is a new cache key, an equal value hits the cache."""

    n_worst: int


# --8<-- [end:configs]


def _worst(flows, config):
    """The ``config.n_worst`` largest |flows|, a stand-in for ToOp's number_most_affected."""
    kind = 'identity' if isinstance(config, IdentityConfig) else 'frozen'
    TRACES[kind] += 1  # only runs while tracing
    return jnp.sort(jnp.abs(flows))[::-1][: config.n_worst]


worst = jax.jit(_worst, static_argnums=1)


def trap(flows) -> tuple[tuple, tuple, tuple]:
    """Mutate an IdentityConfig in place and call again, then use ``replace``: the shapes of the three results."""
    config = IdentityConfig(n_worst=3)
    first = worst(flows, config)
    config.n_worst = 5  # in place: same id, same hash, same compiled program
    mutated = worst(flows, config)
    replaced = worst(flows, replace(config, n_worst=5))  # new object: new key, recompiles
    return first.shape, mutated.shape, replaced.shape


def safe(flows) -> tuple[tuple, tuple, int]:
    """The frozen design: new values recompile, equal values share the compiled program."""
    before = TRACES['frozen']
    a = worst(flows, FrozenConfig(n_worst=3))
    b = worst(flows, FrozenConfig(n_worst=5))
    worst(flows, FrozenConfig(n_worst=3))  # equal by value: cache hit
    return a.shape, b.shape, TRACES['frozen'] - before


def main() -> None:
    """Show the trap and the fix on a vector of eight flows."""
    flows = jnp.array([10.0, -80.0, 30.0, -5.0, 60.0, 20.0, -45.0, 70.0])
    first, mutated, replaced = trap(flows)
    print(f'IdentityConfig: n_worst=3 -> shape {first}; mutated to 5 in place -> {mutated}; replace -> {replaced}')
    print(f'  traces of the identity version: {TRACES["identity"]} (the in-place call did not trace)')
    a, b, traces = safe(flows)
    print(f'FrozenConfig: 3 -> {a}, 5 -> {b}, 3 again -> cache hit; traces: {traces}')


if __name__ == '__main__':
    main()
