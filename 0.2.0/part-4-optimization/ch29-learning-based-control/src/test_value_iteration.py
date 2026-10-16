import numpy as np
from value_iteration import GAMMA, P, R, bellman_update, policy_value, value_iteration


def test_transition_matrices_are_stochastic():
    np.testing.assert_allclose(P.sum(axis=2), 1.0)


def test_first_iterates_by_hand():
    v1, _ = bellman_update(np.zeros(2))
    np.testing.assert_allclose(v1, [0.0, -0.1])  # max over actions of the immediate rewards
    v2, _ = bellman_update(v1)
    np.testing.assert_allclose(v2, [0.9 * (0.9 * 0 + 0.1 * -0.1), -0.1 + 0.9 * (0.8 * 0 + 0.2 * -0.1)])


def test_fixed_point_and_policy():
    v, policy, history = value_iteration()
    np.testing.assert_allclose(v, [-0.0989, -0.2088], atol=5e-5)
    assert policy.tolist() == [0, 1]
    assert len(history) - 1 == 177
    np.testing.assert_allclose(bellman_update(v)[0], v, atol=1e-9)  # it solves the Bellman optimality equation
    np.testing.assert_allclose(policy_value(policy), v, atol=1e-9)


def test_greedy_policy_is_optimal_among_the_four():
    v, policy, _ = value_iteration()
    for other in ([0, 0], [1, 1], [1, 0], [0, 1]):
        assert (policy_value(np.array(other)) <= v + 1e-9).all()


def test_contraction_rate():
    _, _, history = value_iteration()
    gaps = [np.abs(history[k + 1] - history[k]).max() for k in range(5, 60)]
    ratios = np.array(gaps[1:]) / np.array(gaps[:-1])
    assert ratios.max() <= GAMMA + 1e-9
    assert R.shape == (2, 2)
