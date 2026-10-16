import bridge_debug
import jax
import numpy as np


def test_plain_jit_returns_nan_only_for_the_bridge_outage():
    flows = bridge_debug.run_plain()
    assert flows.shape == (6, 6)
    assert np.flatnonzero(np.isnan(flows).any(axis=1)).tolist() == [5]
    assert np.isfinite(flows[:5]).all()


def test_debug_nans_raises_floating_point_error():
    message = bridge_debug.run_debug_nans()
    assert 'nan' in message
    assert jax.config.jax_debug_nans is False  # restored


def test_debug_print_shows_a_zero_denominator(capfd):
    bridge_debug.run_with_print()
    out = capfd.readouterr().out
    assert 'LODF denominators' in out and '0.' in out


def test_flagged_version_marks_the_bridge():
    flows, ok = bridge_debug.run_flagged()
    assert ok.tolist() == [True] * 5 + [False]
    assert np.isfinite(flows[ok]).all()
