import numpy as np
import pytest

pytestmark = pytest.mark.labs
pytest.importorskip('grid2op')
pytest.importorskip('stable_baselines3')

import dqn_labs  # noqa: E402
import grid2op_agents_labs as lab  # noqa: E402
from stable_baselines3.common.env_checker import check_env  # noqa: E402


@pytest.fixture(scope='module')
def setup():
    env = lab.make_env()
    unitary = env.action_space.get_all_unitary_topologies_set(env.action_space)
    actions = [env.action_space({}), *unitary[:4]]
    return env, actions


def test_gym_wrapper(setup):
    env, actions = setup
    gym_env = dqn_labs.Case14Gym(env, actions, chronics=(0,))
    check_env(gym_env, warn=False)
    obs, _ = gym_env.reset(options={'chronic': 0})
    assert obs.shape == (env.n_line + 1,) and obs[-1] == 1.0
    obs, reward, terminated, truncated, _ = gym_env.step(1)
    assert obs[-1] in (0.0, 1.0) and np.isfinite(reward) and not truncated


def test_short_training_and_policy(setup):
    env, actions = setup
    gym_env = dqn_labs.Case14Gym(env, actions, chronics=(0,))
    model = dqn_labs.train(gym_env, steps=300)
    runs = dqn_labs.evaluate(env, model, actions, chronics=(0,))
    assert 1 <= runs[0]['survival'] <= lab.N_STEPS
