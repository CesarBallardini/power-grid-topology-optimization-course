"""Value iteration on a two-state, two-action MDP (Chapter 29, Exercise 1).

State 0 is "secure", state 1 is "overloaded". Action 0 waits, action 1 switches (a topology action). Rewards are
per step after the transition. Run it to print the iterates quoted in the solution::

    python value_iteration.py
"""

import numpy as np

# P[a, s, s']: probability of landing in s' from s under action a
P = np.array(
    [
        [[0.9, 0.1], [0.3, 0.7]],  # wait: a secure grid mostly stays secure, an overloaded one mostly stays overloaded
        [[0.8, 0.2], [0.8, 0.2]],  # switch: from either state the action resolves the overload with probability 0.8
    ]
)
# R[a, s]: expected immediate reward of taking a in s
R = np.array(
    [
        [0.0, -1.0],  # waiting costs nothing when secure and one unit per overloaded step
        [-0.1, -0.1],  # switching has a small cost wherever it is done
    ]
)
GAMMA = 0.9


# --8<-- [start:vi]
def bellman_update(v: np.ndarray, gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray]:
    """One Bellman optimality step: Q(s, a) = R(a, s) + gamma sum_s' P(a, s, s') V(s'); returns max and argmax."""
    q = R + gamma * P @ v  # shape (action, state)
    return q.max(axis=0), q.argmax(axis=0)


def value_iteration(
    gamma: float = GAMMA, tol: float = 1e-10, max_iter: int = 10_000
) -> tuple[np.ndarray, np.ndarray, list]:
    """Iterate from V = 0 until the largest change is below ``tol``; returns V*, the greedy policy and the history."""
    v = np.zeros(len(R[0]))
    history = [v.copy()]
    for _ in range(max_iter):
        v_new, policy = bellman_update(v, gamma)
        history.append(v_new.copy())
        if np.abs(v_new - v).max() < tol:
            return v_new, policy, history
        v = v_new
    raise RuntimeError('value iteration did not converge')


def policy_value(policy: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Exact value of a deterministic policy: solve (I - gamma P_pi) V = R_pi."""
    p_pi = np.array([P[a, s] for s, a in enumerate(policy)])
    r_pi = np.array([R[a, s] for s, a in enumerate(policy)])
    return np.linalg.solve(np.eye(len(policy)) - gamma * p_pi, r_pi)


# --8<-- [end:vi]
def main() -> None:
    """The first iterates by hand, the fixed point and its check."""
    v, policy, history = value_iteration()
    for k in (1, 2, 3):
        print(f'V_{k} = {np.round(history[k], 4)}')
    print(
        f'converged after {len(history) - 1} iterations: V* = {np.round(v, 4)}, policy = {policy} (0 wait, 1 switch)'
    )
    print(f'exact value of that policy: {np.round(policy_value(policy), 4)}')
    for other in ([0, 0], [1, 1], [1, 0]):
        print(f'  policy {other}: {np.round(policy_value(np.array(other)), 4)}')


if __name__ == '__main__':
    main()
