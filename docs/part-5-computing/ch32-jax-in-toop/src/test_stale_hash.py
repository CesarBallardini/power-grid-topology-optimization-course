import jax.numpy as jnp
import pytest
from stale_hash import TRACES, FrozenConfig, IdentityConfig, safe, trap, worst

FLOWS = jnp.array([10.0, -80.0, 30.0, -5.0, 60.0, 20.0, -45.0, 70.0])


def test_in_place_mutation_returns_the_stale_result_and_replace_recompiles():
    first, mutated, replaced = trap(FLOWS)
    assert first == (3,) and mutated == (3,) and replaced == (5,)


def test_the_check_that_would_have_caught_it():
    """A result must follow the configuration it was computed with; the identity design fails this check."""
    config = IdentityConfig(n_worst=2)
    worst(FLOWS, config)
    config.n_worst = 4
    with pytest.raises(AssertionError):
        assert worst(FLOWS, config).shape == (config.n_worst,)
    assert worst(FLOWS, FrozenConfig(n_worst=4)).shape == (4,)  # the frozen design passes the same check


def test_frozen_config_recompiles_on_new_values_only():
    a, b, traces = safe(FLOWS)
    assert a == (3,) and b == (5,)
    assert traces <= 2  # 3 and 5 at most once each; a repeated value never traces
    before = TRACES['frozen']
    worst(FLOWS, FrozenConfig(n_worst=5))
    assert TRACES['frozen'] == before
