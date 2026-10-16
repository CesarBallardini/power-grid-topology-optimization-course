"""Chapter 29, Lab steps 1-3 on the real thing: Grid2Op's ``l2rpn_case14_sandbox`` with the LightSim2Grid backend.

Runs in the labs container (``make test-labs``, see ``docker/labs``), not in the course ``.venv``. The environment,
the three baseline agents (do-nothing, random, greedy by simulation), tabular Q-learning over a reduced action set
and the comparison over the same chronics mirror ``grid_env.py``, the NumPy stand-in of the fallback track, so the
two tracks print the same table. The reward is the one of ``grid_env.py``, computed from the observation: minus the
summed overload in units of rating, minus 0.02 per topology change, minus 5 when a game over ends the day early.
The agents are evaluated on the ten hardest first days of chronics 0-299 (highest do-nothing peak loading) and the
learned ones trained on the twenty hardest of chronics 300-1003, so they never see an evaluation day. Run it to
print every number quoted
in the solutions (several minutes on one CPU core, see the solutions for the time of each step)::

    python grid2op_agents_labs.py
"""

import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')  # Grid2Op warns about the dataset and about the Gymnasium shims
import grid2op  # noqa: E402
from lightsim2grid import LightSimBackend  # noqa: E402

ENV_NAME = 'l2rpn_case14_sandbox'
DATA_DIR = Path.home() / 'data_grid2op' / ENV_NAME  # the named volume of docker/labs; test data otherwise
N_STEPS = 288  # one day of five-minute steps
# First days ranked by the do-nothing agent's peak loading (hardest_chronics, checked by main): the ten hardest of
# chronics 0-299 are the evaluation days of every agent, the twenty hardest of 300-1003 the training days
CHRONICS = (248, 72, 3, 57, 154, 194, 106, 181, 220, 240)
TRAIN_CHRONICS = (963, 912, 305, 341, 452, 428, 930, 352, 998, 903, 753, 441, 860, 944, 751, 582, 422, 999, 317, 682)
RHO_BINS = np.array([0.6, 0.8, 0.9, 1.0, 1.1])  # state discretization of the maximum loading, as in grid_env.py
THRESHOLD = 0.9  # the random and greedy agents act above this loading


# --8<-- [start:env]
def make_env(seed: int = 0):
    """The case 14 sandbox on LightSim2Grid: the full dataset when the data volume holds it, else the test chronics."""
    env = grid2op.make(ENV_NAME, test=not DATA_DIR.exists(), backend=LightSimBackend())
    env.seed(seed)
    return env


def describe(env) -> dict:
    """Lab step 1: the sizes of the grid and of the observation, and the unitary topology actions."""
    unitary = env.action_space.get_all_unitary_topologies_set(env.action_space)
    obs = env.reset()
    return {
        'n_sub': env.n_sub,
        'n_line': env.n_line,
        'n_gen': env.n_gen,
        'n_load': env.n_load,
        'rho': obs.rho.shape[0],
        'topo_vect': obs.topo_vect.shape[0],
        'line_status': obs.line_status.shape[0],
        'unitary_topologies': len(unitary),
        'per_substation': np.bincount(
            [int(np.flatnonzero(a.get_topological_impact()[1])[0]) for a in unitary], minlength=env.n_sub
        ).tolist(),
        'chronics': len(env.chronics_handler.real_data.subpaths)
        if hasattr(env.chronics_handler.real_data, 'subpaths')
        else None,
    }


def reset_day(env, chronic: int, seed: int):
    """Start ``chronic`` at its first step with a horizon of one day, seeded (only the random agent uses the seed)."""
    env.seed(seed)
    return env.reset(seed=seed, options={'time serie id': chronic, 'max step': N_STEPS})


def shaped_reward(obs, changed: bool, done: bool, step: int) -> float:
    """The reward of ``grid_env.py``: minus the overload, minus 0.02 per change, minus 5 for an early game over."""
    reward = -float(np.maximum(0.0, obs.rho - 1.0).sum()) - 0.02 * changed
    if done and step < N_STEPS:
        reward -= 5.0
    return reward


# --8<-- [end:env]
# --8<-- [start:agents]
class Agents:
    """The lab's agents over the action list ``actions`` (index 0 is do-nothing, the rest unitary topologies)."""

    def __init__(self, env, actions: list, seed: int = 0):
        """``actions[0]`` must be the do-nothing action; ``seed`` drives the random agent."""
        self.env = env
        self.actions = actions
        self.rng = np.random.default_rng(seed)
        self.simulations = 0  # power flows spent by the greedy agent, to compare with the learned agents

    def do_nothing(self, obs) -> int:
        """Never acts."""
        return 0

    def random(self, obs) -> int:
        """Plays a random topology action when a line passes the threshold."""
        return int(self.rng.integers(1, len(self.actions))) if obs.rho.max() > THRESHOLD else 0

    def greedy(self, obs) -> int:
        """Simulates every action when a line passes the threshold and plays the one with the lowest maximum loading.

        A simulation that ends the game (``done``) or is illegal counts as infinitely bad; do-nothing is candidate 0,
        so the agent acts only when some topology beats staying put.
        """
        if obs.rho.max() <= THRESHOLD:
            return 0
        worst = np.full(len(self.actions), np.inf)
        for i, action in enumerate(self.actions):
            sim_obs, _, sim_done, sim_info = obs.simulate(action)
            self.simulations += 1
            if not sim_done and not sim_info['is_illegal'] and not sim_info['is_ambiguous']:
                worst[i] = sim_obs.rho.max()
        return int(np.argmin(worst))


def run_episode(env, agent, chronic: int, seed: int, actions: list, trace: list | None = None) -> dict:
    """One day of ``chronic`` played by ``agent(obs) -> action index``: survival, mean max loading, reward, ms/step."""
    obs = reset_day(env, chronic, seed)
    total, maxima, done, step = 0.0, [], False, 0
    t0 = time.perf_counter()
    while not done:
        index = agent(obs)
        obs, _, done, info = env.step(actions[index])
        step += 1
        changed = index != 0 and not info['is_illegal']
        total += shaped_reward(obs, changed, done, step)
        maxima.append(float(obs.rho.max()))
        if trace is not None:
            trace.append(index)
    return {
        'survival': step,
        'mean_max_rho': float(np.mean(maxima)),
        'reward': total,
        'ms_per_step': 1e3 * (time.perf_counter() - t0) / step,
    }


def state_index(obs, in_reference: bool) -> int:
    """Discrete state: bin of the maximum loading, most loaded line, whether the grid is in its reference topology."""
    bin_index = int(np.searchsorted(RHO_BINS, obs.rho.max()))
    return (bin_index * len(obs.rho) + int(obs.rho.argmax())) * 2 + int(in_reference)


def n_states(env) -> int:
    """Size of the Q table's first axis."""
    return (len(RHO_BINS) + 1) * env.n_line * 2


def frequent_actions(env, unitary: list, chronics=TRAIN_CHRONICS, n_top: int = 20) -> tuple[list[int], dict]:
    """Lab step 3: the ``n_top`` unitary actions the greedy agent plays most often on the training chronics.

    Returns the indices into ``unitary`` (most frequent first), the play counts and the greedy agent's episode
    statistics on those days (with the number of simulations it spent).
    """
    actions = [env.action_space({}), *unitary]
    counts = np.zeros(len(actions), dtype=int)
    stats = []
    for chronic in chronics:
        agents = Agents(env, actions, seed=chronic)
        trace: list[int] = []
        stats.append(run_episode(env, agents.greedy, chronic, chronic, actions, trace))
        stats[-1]['simulations'] = agents.simulations
        counts += np.bincount(trace, minlength=len(actions))
    counts[0] = 0
    top = [int(i) - 1 for i in np.argsort(-counts, kind='stable')[:n_top] if counts[i] > 0]
    return top, {'counts': counts[1:], 'stats': stats}


def q_learning(
    env,
    actions: list,
    chronics=TRAIN_CHRONICS,
    episodes: int = 200,
    seed: int = 100,
    alpha: float = 0.2,
    gamma: float = 0.95,
    eps: float = 0.3,
) -> np.ndarray:
    """Tabular Q-learning over ``actions`` on the days ``chronics`` in turn; epsilon decays linearly to 0.01."""
    rng = np.random.default_rng(seed)
    q = np.zeros((n_states(env), len(actions)))
    for episode in range(episodes):
        epsilon = max(0.01, eps * (1 - episode / episodes))
        chronic = int(chronics[episode % len(chronics)])
        obs = reset_day(env, chronic, seed + episode)
        in_reference, done, step = True, False, 0
        s = state_index(obs, in_reference)
        while not done:
            i = int(rng.integers(len(actions))) if rng.random() < epsilon else int(np.argmax(q[s]))
            obs, _, done, info = env.step(actions[i])
            step += 1
            changed = i != 0 and not info['is_illegal']
            in_reference = in_reference and not changed
            reward = shaped_reward(obs, changed, done, step)
            s_next = state_index(obs, in_reference)
            target = reward if (done and step < N_STEPS) else reward + gamma * q[s_next].max()
            q[s, i] += alpha * (target - q[s, i])
            s = s_next
    return q


class QPolicy:
    """The greedy policy of a Q table in the signature of the other agents (tracks the reference-topology flag)."""

    def __init__(self, q: np.ndarray):
        """A fresh policy starts in the reference topology."""
        self.q = q
        self.in_reference = True

    def __call__(self, obs) -> int:
        """The action with the highest Q value in the observed state."""
        i = int(np.argmax(self.q[state_index(obs, self.in_reference)]))
        if i != 0:
            self.in_reference = False
        return i


def compare(env, actions: list, chronics=CHRONICS) -> dict:
    """Do-nothing, random and greedy agents over ``actions`` on the same chronics and seeds."""
    out: dict[str, list[dict]] = {'do-nothing': [], 'random': [], 'greedy (simulate)': []}
    for chronic in chronics:
        agents = Agents(env, actions, seed=chronic)
        out['do-nothing'].append(run_episode(env, agents.do_nothing, chronic, chronic, actions))
        out['random'].append(run_episode(env, agents.random, chronic, chronic, actions))
        out['greedy (simulate)'].append(run_episode(env, agents.greedy, chronic, chronic, actions))
        out['greedy (simulate)'][-1]['simulations'] = agents.simulations
    return out


def evaluate_q(env, q: np.ndarray, actions: list, chronics=CHRONICS) -> list[dict]:
    """The greedy policy of the Q table on the evaluation chronics."""
    return [run_episode(env, QPolicy(q), chronic, chronic, actions) for chronic in chronics]


def summary_table(results: dict) -> str:
    """The table of the solutions: survival (median, min), mean maximum loading, reward and ms per step."""
    lines = [f'{"agent":18s} {"survival (median, min)":>24s} {"mean max rho":>13s} {"reward":>9s} {"ms/step":>8s}']
    for name, runs in results.items():
        surv = np.array([r['survival'] for r in runs])
        lines.append(
            f'{name:18s} {np.median(surv):10.0f} {surv.min():12d} '
            f'{np.mean([r["mean_max_rho"] for r in runs]):13.3f} '
            f'{np.mean([r["reward"] for r in runs]):9.2f} '
            f'{np.mean([r["ms_per_step"] for r in runs]):8.2f}'
        )
    return '\n'.join(lines)


# --8<-- [end:agents]
def chronic_scan(env, chronics=range(300)) -> list[tuple[int, int, float]]:
    """Do-nothing survival and peak loading of the first day of each chronic, to pick the hard ones."""
    out = []
    for chronic in chronics:
        obs = reset_day(env, chronic, 0)
        done, step, peak = False, 0, float(obs.rho.max())
        while not done:
            obs, _, done, _ = env.step(env.action_space({}))
            step += 1
            peak = max(peak, float(obs.rho.max()))
        out.append((chronic, step, peak))
    return out


def hardest_chronics(scan: list[tuple[int, int, float]], n: int) -> list[int]:
    """The ``n`` chronics of ``scan`` with the highest do-nothing peak loading, hardest first."""
    return [c for c, _, _ in sorted(scan, key=lambda r: -r[2])[:n]]


def main() -> None:
    """Lab steps 1 to 3 on l2rpn_case14_sandbox."""
    env = make_env()
    info = describe(env)
    print(f'{ENV_NAME} ({"full dataset" if DATA_DIR.exists() else "test chronics"}, grid2op {grid2op.__version__}):')
    print(
        f'  {info["n_sub"]} substations, {info["n_line"]} lines, {info["n_gen"]} generators, {info["n_load"]} loads, '
        f'{info["chronics"]} chronics'
    )
    print(f'  obs.rho {info["rho"]}, obs.topo_vect {info["topo_vect"]}, obs.line_status {info["line_status"]}')
    print(f'  unitary topology actions: {info["unitary_topologies"]}, per substation {info["per_substation"]}')
    for pool, chosen, n in ((range(300), CHRONICS, 10), (range(300, info['chronics']), TRAIN_CHRONICS, 20)):
        t0 = time.perf_counter()
        scan = chronic_scan(env, pool)
        failed = [(c, s) for c, s, _ in scan if s < N_STEPS]
        print(
            f'do-nothing, first day of chronics {pool.start}-{pool.stop - 1} ({time.perf_counter() - t0:.0f} s): '
            f'{len(failed)} game overs (chronic, step) {failed}'
        )
        print(f'  the {n} highest peak loadings: {hardest_chronics(scan, n)}')
        if tuple(hardest_chronics(scan, n)) != chosen:
            print(f'  differs from the days in this module: {chosen}')
    unitary = env.action_space.get_all_unitary_topologies_set(env.action_space)
    actions = [env.action_space({}), *unitary]
    t0 = time.perf_counter()
    top, greedy = frequent_actions(env, unitary)
    print(
        f'greedy agent on the {len(TRAIN_CHRONICS)} training days: {time.perf_counter() - t0:.0f} s, '
        f'{sum(s["simulations"] for s in greedy["stats"])} simulations, {int((greedy["counts"] > 0).sum())} distinct '
        f'actions played; the {len(top)} most frequent (index, count): {[(i, int(greedy["counts"][i])) for i in top]}'
    )
    reduced = [actions[0], *[unitary[i] for i in top]]
    t0 = time.perf_counter()
    q = q_learning(env, reduced)
    print(
        f'Q-learning over {len(reduced)} actions: 200 training days in {time.perf_counter() - t0:.0f} s, '
        f'{int(np.count_nonzero(q.any(axis=1)))} of {q.shape[0]} states visited'
    )
    t0 = time.perf_counter()
    results = compare(env, actions)
    results['Q-learning'] = evaluate_q(env, q, reduced)
    print(f'evaluation on the {len(CHRONICS)} evaluation days ({time.perf_counter() - t0:.0f} s):')
    print(summary_table(results))
    greedy_simulations = sum(r['simulations'] for r in results['greedy (simulate)'])
    print(f'greedy agent: {greedy_simulations} simulations on the evaluation days')
    print('survival per evaluation day:')
    for name, runs in results.items():
        print(f'  {name:18s} {[r["survival"] for r in runs]}')


if __name__ == '__main__':
    main()
