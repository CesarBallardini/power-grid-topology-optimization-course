from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.labs
pytest.importorskip('grid2op')
pytest.importorskip('lightsim2grid')

import grid2op_agents_labs as lab  # noqa: E402


@pytest.fixture(scope='module')
def env():
    return lab.make_env()


@pytest.fixture(scope='module')
def unitary(env):
    return env.action_space.get_all_unitary_topologies_set(env.action_space)


def test_describe(env):
    info = lab.describe(env)
    assert (info['n_sub'], info['n_line'], info['n_gen'], info['n_load']) == (14, 20, 6, 11)
    assert (info['rho'], info['topo_vect'], info['line_status']) == (20, 57, 20)
    assert info['unitary_topologies'] == 178 and sum(info['per_substation']) == 178


def test_shaped_reward():
    obs = SimpleNamespace(rho=np.array([0.5, 1.2, 1.1]))
    assert lab.shaped_reward(obs, changed=False, done=False, step=10) == pytest.approx(-0.3)
    assert lab.shaped_reward(obs, changed=True, done=False, step=10) == pytest.approx(-0.32)
    assert lab.shaped_reward(obs, changed=False, done=True, step=10) == pytest.approx(-5.3)
    assert lab.shaped_reward(obs, changed=False, done=True, step=lab.N_STEPS) == pytest.approx(-0.3)


def test_state_index_range(env):
    obs = env.reset()
    for in_reference in (True, False):
        assert 0 <= lab.state_index(obs, in_reference) < lab.n_states(env)


def test_hardest_chronics():
    scan = [(0, 288, 0.8), (1, 3, 1.05), (2, 288, 0.95)]
    assert lab.hardest_chronics(scan, 2) == [1, 2]


def test_do_nothing_day(env, unitary):
    actions = [env.action_space({}), *unitary]
    agents = lab.Agents(env, actions, seed=0)
    stats = lab.run_episode(env, agents.do_nothing, 0, 0, actions)
    assert 1 <= stats['survival'] <= lab.N_STEPS and 0 < stats['mean_max_rho'] < 2
    assert agents.greedy(SimpleNamespace(rho=np.array([0.5]))) == 0  # below the threshold: no simulation
    assert agents.simulations == 0


def test_q_learning_short(env, unitary):
    actions = [env.action_space({}), *unitary[:3]]
    q = lab.q_learning(env, actions, chronics=(0,), episodes=1)
    assert q.shape == (lab.n_states(env), 4) and np.isfinite(q).all()
    runs = lab.evaluate_q(env, q, actions, chronics=(0,))
    assert runs[0]['survival'] >= 1
