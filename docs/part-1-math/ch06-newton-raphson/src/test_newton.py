import numpy as np
import pytest
from newton import jacobian, newton, residual


def test_converges_quadratically_from_minus_one():
    history = newton(residual, jacobian, [1.0, -1.0])
    x, r = history[-1]
    assert r < 1e-12 and len(history) <= 6
    np.testing.assert_allclose(x, [1.0042, -1.7296], atol=5e-5)
    norms = [r for _, r in history[:4]]
    assert np.allclose(norms, [8.6e-1, 4.0e-2, 1.0e-4, 8.4e-10], rtol=0.1)


def test_other_start_finds_other_root():
    x, r = newton(residual, jacobian, [-1.0, 1.0])[-1]
    assert r < 1e-12
    np.testing.assert_allclose(x, [-1.8163, 0.8374], atol=5e-5)


def test_singular_jacobian_at_origin():
    with pytest.raises(np.linalg.LinAlgError):
        newton(residual, jacobian, [0.0, 0.0])
