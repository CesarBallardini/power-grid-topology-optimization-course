"""Chapter 29, Lab step 4: a small DQN (Stable-Baselines3) on l2rpn_case14_sandbox through a Gymnasium wrapper.

Runs in the labs container (``make test-labs``). The action space is the reduced one of Lab step 3 (do-nothing plus
the unitary topologies the greedy agent played most often on the training days), the observation is the vector of
line loadings plus the reference-topology flag, and the reward is ``grid2op_agents_labs.shaped_reward``. Training
cycles through the training days of Lab step 3, evaluation uses the evaluation days of the baselines. The training
budget is deliberately short (``TRAIN_STEPS`` environment steps, about 6 minutes on one CPU core); the numbers in the
solutions are for that budget and change with a longer one. Run it to print them::

    python dqn_labs.py
"""

import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import grid2op_agents_labs as lab  # noqa: E402
import gymnasium as gym  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

TRAIN_STEPS = 20_000  # environment steps of training; about 70 days of five-minute steps
SEED = 0


# --8<-- [start:dqn]
class Case14Gym(gym.Env):
    """Gymnasium view of the Grid2Op day: discrete reduced actions, loadings as observation, the lab's reward."""

    metadata = {'render_modes': []}

    def __init__(self, env, actions: list, chronics=lab.TRAIN_CHRONICS, seed: int = 0):
        """``actions[0]`` is do-nothing; each ``reset`` takes the next training chronic in turn."""
        super().__init__()
        self.env, self.actions, self.chronics = env, actions, list(chronics)
        self.base_seed, self.episode = seed, 0
        self.observation_space = gym.spaces.Box(0.0, 10.0, (env.n_line + 1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(actions))
        self.in_reference, self.step_count = True, 0

    def _observe(self, obs) -> np.ndarray:
        return np.append(np.minimum(obs.rho, 10.0), float(self.in_reference)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """Start the next chronic (or ``options['chronic']``) at its first step, one day long."""
        super().reset(seed=seed)
        chronic = (
            options['chronic']
            if options and 'chronic' in options
            else self.chronics[self.episode % len(self.chronics)]
        )
        self.episode += 1
        self.in_reference, self.step_count = True, 0
        obs = lab.reset_day(self.env, int(chronic), self.base_seed + self.episode)
        self.last_obs = obs
        return self._observe(obs), {}

    def step(self, action: int):
        """Grid2Op step with the lab's reward; a game over before the end of the day terminates the episode."""
        obs, _, done, info = self.env.step(self.actions[int(action)])
        self.step_count += 1
        changed = int(action) != 0 and not info['is_illegal']
        self.in_reference = self.in_reference and not changed
        reward = lab.shaped_reward(obs, changed, done, self.step_count)
        self.last_obs = obs
        terminated = done and self.step_count < lab.N_STEPS
        truncated = done and not terminated
        return self._observe(obs), reward, terminated, truncated, info


def train(gym_env: Case14Gym, steps: int = TRAIN_STEPS, seed: int = SEED) -> DQN:
    """DQN with a small MLP, exploration decaying over the first half of the budget, learning from step 1000."""
    model = DQN(
        'MlpPolicy',
        gym_env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.5,
        exploration_final_eps=0.02,
        policy_kwargs={'net_arch': [64, 64]},
        seed=seed,
        verbose=0,
        device='cpu',
    )
    model.learn(total_timesteps=steps)
    return model


class DQNPolicy:
    """The trained network as an agent in the signature of ``grid2op_agents_labs`` (deterministic action)."""

    def __init__(self, model: DQN):
        """A fresh policy starts in the reference topology."""
        self.model = model
        self.in_reference = True

    def __call__(self, obs) -> int:
        """The network's greedy action for the observed loadings."""
        x = np.append(np.minimum(obs.rho, 10.0), float(self.in_reference)).astype(np.float32)
        i = int(self.model.predict(x, deterministic=True)[0])
        if i != 0:
            self.in_reference = False
        return i


def evaluate(env, model: DQN, actions: list, chronics=lab.CHRONICS) -> list[dict]:
    """The DQN's greedy policy on the baseline chronics, same statistics as the other agents."""
    return [lab.run_episode(env, DQNPolicy(model), chronic, chronic, actions) for chronic in chronics]


# --8<-- [end:dqn]
def main() -> None:
    """Lab step 4: train, evaluate on the baselines' days, and compare the training cost with the greedy agent's."""
    env = lab.make_env()
    unitary = env.action_space.get_all_unitary_topologies_set(env.action_space)
    actions = [env.action_space({}), *unitary]
    top, _ = lab.frequent_actions(env, unitary)
    reduced = [actions[0], *[unitary[i] for i in top]]
    print(f'reduced action space: {len(reduced)} actions (do-nothing + {len(top)} unitary topologies)')
    gym_env = Case14Gym(env, reduced)
    t0 = time.perf_counter()
    model = train(gym_env)
    dt = time.perf_counter() - t0
    print(
        f'DQN: {TRAIN_STEPS} training steps in {dt:.0f} s ({1e3 * dt / TRAIN_STEPS:.1f} ms per step), '
        f'{gym_env.episode} training episodes'
    )
    results = lab.compare(env, actions)
    results['DQN'] = evaluate(env, model, reduced)
    print(f'evaluation on the {len(lab.CHRONICS)} evaluation days:')
    print(lab.summary_table(results))
    print('survival per evaluation day:')
    for name, runs in results.items():
        print(f'  {name:18s} {[r["survival"] for r in runs]}')
    simulations = sum(r['simulations'] for r in results['greedy (simulate)'])
    print(
        f'greedy agent: {simulations} simulations over {len(lab.CHRONICS)} days '
        f'({simulations / len(lab.CHRONICS):.0f} per day) against {TRAIN_STEPS} training steps once'
    )


if __name__ == '__main__':
    main()
