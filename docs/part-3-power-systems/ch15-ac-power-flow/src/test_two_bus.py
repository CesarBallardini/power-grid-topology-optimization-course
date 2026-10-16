import numpy as np
import pytest
from two_bus import exact_solution, injections_2bus, jacobian_2bus, newton_steps, transfer

X, P, Q = 0.1, 1.0, 0.5


def test_first_step_by_hand():
    first = newton_steps(P, Q, X, 1)[0]
    np.testing.assert_allclose(first['mismatch'], [-1.0, -0.5])
    np.testing.assert_allclose(first['jacobian'], [[10.0, 0.0], [0.0, 10.0]])
    np.testing.assert_allclose(first['state'], [-0.1, 0.95])


def test_jacobian_matches_finite_differences():
    d, v, h = -0.2, 0.93, 1e-6
    jac = jacobian_2bus(d, v, X)
    col_d = (injections_2bus(d + h, v, X) - injections_2bus(d - h, v, X)) / (2 * h)
    col_v = (injections_2bus(d, v + h, X) - injections_2bus(d, v - h, X)) / (2 * h)
    np.testing.assert_allclose(jac[:, 0], col_d, atol=1e-7)
    np.testing.assert_allclose(jac[:, 1], col_v, atol=1e-7)


def test_iteration_converges_to_the_exact_root():
    d, v = exact_solution(P, Q, X)
    np.testing.assert_allclose(injections_2bus(d, v, X), [-P, -Q], atol=1e-12)
    np.testing.assert_allclose(newton_steps(P, Q, X, 6)[-1]['state'], [d, v], atol=1e-10)


def test_maximum_transfer_at_ninety_degrees():
    delta = np.linspace(0, np.pi, 1801)
    p, _ = transfer(1.0, 1.0, X, delta)
    assert p.max() == pytest.approx(1 / X)
    assert delta[p.argmax()] == pytest.approx(np.pi / 2)
