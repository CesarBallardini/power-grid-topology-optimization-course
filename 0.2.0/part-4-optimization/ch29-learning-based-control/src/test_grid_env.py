import numpy as np
import pytest
from grid_env import (
    LINES,
    N_STATES,
    N_STEPS,
    TOPOLOGIES,
    GridEnv,
    compare,
    do_nothing,
    frequent_actions,
    greedy_agent,
    load_profile,
    q_learning,
    q_policy,
    rho,
    run_episode,
    state_index,
)


def test_topologies_exclude_bridges():
    assert len(TOPOLOGIES) == 7  # every line of the 5-bus mesh can be opened without islanding
    assert all(m.sum() in (5, 6) for m in TOPOLOGIES)


def test_peak_overloads_the_meshed_grid_and_one_switch_relieves_it():
    peak = load_profile(np.arange(N_STEPS)).max()
    assert peak == pytest.approx(150.0, abs=0.5)
    meshed = rho(0, peak)
    assert meshed.max() == pytest.approx(1.14, abs=0.01) and int(meshed.argmax()) == 2  # line 2-3
    best = min(rho(t, peak).max() for t in range(1, len(TOPOLOGIES)))
    assert best == pytest.approx(0.70, abs=0.01)
    assert rho(0, 0.45 * peak).max() < 0.7


def test_state_index_is_a_bijection_on_its_range():
    seen = set()
    for bin_rho in (0.5, 0.7, 0.85, 0.95, 1.05, 1.2):
        for line in range(len(LINES)):
            loading = np.zeros(len(LINES))
            loading[line] = bin_rho
            for topology in range(len(TOPOLOGIES)):
                seen.add(state_index(loading, topology))
    assert len(seen) == N_STATES and max(seen) == N_STATES - 1


def test_episode_mechanics():
    env = GridEnv(0)
    loading, topology = env.reset()
    assert topology == 0 and loading.shape == (len(LINES),)
    assert env.simulate(4).max() < 1  # topology 3 is safe at the start
    loading, reward, done = env.step(4)
    assert env.topology == 3 and reward == pytest.approx(-0.02) and not done
    loading, reward, done = env.step(0)
    assert env.topology == 3 and reward == 0.0


def test_do_nothing_trips_at_the_peak_and_greedy_survives():
    nothing = run_episode(GridEnv(0), do_nothing)
    greedy = run_episode(GridEnv(0), greedy_agent)
    assert nothing['survival'] < 230 and nothing['reward'] < -5
    assert greedy['survival'] == N_STEPS - 1 and greedy['reward'] > -0.1


def test_q_learning_matches_the_greedy_agent():
    actions = frequent_actions()
    assert actions == [0, 4]  # do nothing, or open line 2-3
    q = q_learning(actions)
    results = compare(range(3), q=q, actions=actions)
    for run in results['Q-learning']:
        assert run['survival'] == N_STEPS - 1
    assert all(r['survival'] < N_STEPS - 1 for r in results['do-nothing'])
    policy = q_policy(q, actions)
    env = GridEnv(1)
    loading, _ = env.reset()
    assert policy(env, loading) == 0  # nothing to do in the morning
