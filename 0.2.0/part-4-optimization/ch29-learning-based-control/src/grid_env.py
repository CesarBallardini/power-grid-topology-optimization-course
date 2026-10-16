"""A tiny topology-control environment in NumPy, four agents and tabular Q-learning (Chapter 29, Lab steps 1-3).

Grid2Op is not part of the course environment, so this module plays its role on a five-bus DC grid: one day of
288 five-minute steps with a load profile that peaks above what the meshed topology can carry, actions that open one
line or close every line, line loadings ``rho`` from the PTDF, and game over when a line exceeds the cascade limit
or the grid is islanded. The agents are do-nothing, random, greedy-by-simulation (the exhaustive one-step search
of the lab) and the greedy policy of a learned Q table. Run it to print the comparison over ten seeds::

    python grid_env.py
"""

import time

import numpy as np
from scipy.sparse.csgraph import connected_components

LINES = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]])
X = np.array([0.20, 0.20, 0.10, 0.30, 0.30, 0.20])  # p.u.
RATING = np.array([120.0, 130.0, 15.0, 100.0, 55.0, 15.0])  # MW; 1-2 and 3-4 are weak tie lines
LOAD_SHARE = np.array([0.0, 0.0, 0.35, 0.40, 0.25])  # bus 0 is the slack generator, loads at buses 2, 3, 4
N_STEPS = 288  # one day of five-minute steps
PEAK_MW = 150.0
CASCADE_RHO = 1.5  # a line above 150 % trips at once; above 100 % it trips after OVERFLOW_STEPS consecutive steps
OVERFLOW_STEPS = 2  # Grid2Op's default soft-overflow allowance
RHO_BINS = np.array([0.6, 0.8, 0.9, 1.0, 1.1])  # state discretization of the maximum loading


# --8<-- [start:env]
def load_profile(step: np.ndarray) -> np.ndarray:
    """Total load in MW: a morning ramp and an evening peak on a day of ``N_STEPS`` steps."""
    t = step / N_STEPS
    return PEAK_MW * (0.45 + 0.25 * np.exp(-(((t - 0.35) / 0.08) ** 2)) + 0.55 * np.exp(-(((t - 0.78) / 0.07) ** 2)))


def topologies() -> list[np.ndarray]:
    """Action set: every line in service, then one line open at a time (bridges excluded, they island the grid)."""
    out = [np.ones(len(LINES), dtype=bool)]
    for k in range(len(LINES)):
        mask = np.ones(len(LINES), dtype=bool)
        mask[k] = False
        if ptdf(mask) is not None:
            out.append(mask)
    return out


def ptdf(in_service: np.ndarray) -> np.ndarray | None:
    """PTDF of the lines in service with bus 0 as slack, or ``None`` when the grid is islanded."""
    lines, n_bus = LINES[in_service], len(LOAD_SHARE)
    a = np.zeros((len(lines), n_bus))
    a[np.arange(len(lines)), lines[:, 0]] = 1
    a[np.arange(len(lines)), lines[:, 1]] = -1
    adjacency = np.zeros((n_bus, n_bus))
    adjacency[lines[:, 0], lines[:, 1]] = 1
    if connected_components(adjacency, directed=False)[0] > 1:
        return None
    b = 1 / X[in_service]
    bbus = a.T @ (b[:, None] * a)
    h = np.zeros((len(lines), n_bus))
    h[:, 1:] = (b[:, None] * a[:, 1:]) @ np.linalg.inv(bbus[1:, 1:])
    return h


TOPOLOGIES = topologies()
PTDFS = [ptdf(mask) for mask in TOPOLOGIES]


def rho(topology: int, load_mw: float) -> np.ndarray:
    """Loading of every line (0 for an open line) when the total load is ``load_mw``, split by ``LOAD_SHARE``."""
    mask = TOPOLOGIES[topology]
    p = -LOAD_SHARE * load_mw
    p[0] = -p[1:].sum()
    out = np.zeros(len(LINES))
    out[mask] = np.abs(PTDFS[topology] @ p) / RATING[mask]
    return out


class GridEnv:
    """One episode: the state is (step, topology); ``reset`` draws the day's load noise from the seed."""

    n_actions = len(TOPOLOGIES) + 1  # 0 = do nothing, k = switch to topology k - 1

    def __init__(self, seed: int, noise: float = 0.04):
        """``seed`` drives the day's load noise (relative standard deviation ``noise``) and the random agent."""
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def reset(self) -> tuple[np.ndarray, int]:
        """Start a day; returns the observation (rho) and the topology index."""
        self.step_index = 0
        self.topology = 0
        self.overflow = np.zeros(len(LINES), dtype=int)  # consecutive steps above 100 % per line
        self.load = load_profile(np.arange(N_STEPS)) * (1 + self.noise * self.rng.standard_normal(N_STEPS))
        return rho(0, self.load[0]), 0

    def simulate(self, action: int) -> np.ndarray:
        """What ``obs.simulate`` does in Grid2Op: the loading after ``action`` at the next step, no state change."""
        topology = self.topology if action == 0 else action - 1
        return rho(topology, self.load[min(self.step_index + 1, N_STEPS - 1)])

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Apply the action and advance one step: reward = -overload - switching cost; done at a trip or day end."""
        changed = action != 0 and action - 1 != self.topology
        if action != 0:
            self.topology = action - 1
        self.step_index += 1
        done = self.step_index >= N_STEPS - 1
        loading = rho(self.topology, self.load[self.step_index])
        overload = np.maximum(0, loading - 1).sum()
        reward = -overload - 0.02 * changed
        self.overflow = np.where(loading > 1, self.overflow + 1, 0)
        if loading.max() > CASCADE_RHO or self.overflow.max() > OVERFLOW_STEPS:
            reward -= 5.0  # a line trips: game over, as in Grid2Op
            done = True
        return loading, reward, done


# --8<-- [end:env]
# --8<-- [start:agents]
def do_nothing(env: GridEnv, loading: np.ndarray) -> int:
    """Never acts."""
    return 0


def random_agent(env: GridEnv, loading: np.ndarray) -> int:
    """Acts at random when a line is above 90 %, as a baseline for the value of information."""
    return int(env.rng.integers(env.n_actions)) if loading.max() > 0.9 else 0


def greedy_agent(env: GridEnv, loading: np.ndarray, threshold: float = 0.9) -> int:
    """Simulates every action when the loading passes ``threshold`` and plays the one with the lowest maximum."""
    if loading.max() <= threshold:
        return 0
    worst = [env.simulate(a).max() for a in range(env.n_actions)]
    return int(np.argmin(worst))


def state_index(loading: np.ndarray, topology: int) -> int:
    """Discrete state: bin of the maximum loading, index of the most loaded line, current topology."""
    bin_index = int(np.searchsorted(RHO_BINS, loading.max()))
    return (bin_index * len(LINES) + int(loading.argmax())) * len(TOPOLOGIES) + topology


N_STATES = (len(RHO_BINS) + 1) * len(LINES) * len(TOPOLOGIES)


def run_episode(env: GridEnv, policy) -> dict:
    """Survival steps, mean maximum loading, total reward and wall-clock seconds per step of one episode."""
    loading, _ = env.reset()
    total, maxima, done = 0.0, [], False
    t0 = time.perf_counter()
    while not done:
        loading, reward, done = env.step(policy(env, loading))
        total += reward
        maxima.append(loading.max())
    return {
        'survival': env.step_index,
        'mean_max_rho': float(np.mean(maxima)),
        'reward': total,
        'ms_per_step': 1e3 * (time.perf_counter() - t0) / len(maxima),
    }


def frequent_actions(n_top: int = 2, seeds: range = range(100, 110)) -> list[int]:
    """Lab step 3: do-nothing plus the ``n_top`` actions the greedy agent played most often on training days."""
    counts = np.zeros(GridEnv.n_actions, dtype=int)
    for seed in seeds:
        env = GridEnv(seed)
        loading, _ = env.reset()
        done = False
        while not done:
            a = greedy_agent(env, loading)
            counts[a] += 1
            loading, _, done = env.step(a)
    counts[0] = 0
    return [0, *[int(a) for a in np.argsort(-counts)[:n_top] if counts[a] > 0]]


def q_learning(
    actions: list[int], episodes: int = 500, seed: int = 100, alpha: float = 0.2, gamma: float = 0.95, eps: float = 0.3
) -> np.ndarray:
    """Tabular Q-learning over the reduced action list on days drawn from ``seed``; returns the Q table.

    Exploration is epsilon-greedy with epsilon decaying linearly from ``eps`` to 0.01: with a constant epsilon the
    agent switches somewhere during the morning on almost every training day and never sees the meshed grid at
    the peak, so the states that matter are never learned.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, len(actions)))
    for episode in range(episodes):
        epsilon = max(0.01, eps * (1 - episode / episodes))
        env = GridEnv(int(rng.integers(1 << 31)))
        loading, topology = env.reset()
        s, done = state_index(loading, topology), False
        while not done:
            i = int(rng.integers(len(actions))) if rng.random() < epsilon else int(np.argmax(q[s]))
            loading, reward, done = env.step(actions[i])
            s_next = state_index(loading, env.topology)
            tripped = done and env.step_index < N_STEPS - 1
            target = reward if tripped else reward + gamma * q[s_next].max()
            q[s, i] += alpha * (target - q[s, i])
            s = s_next
    return q


def q_policy(q: np.ndarray, actions: list[int]):
    """The greedy policy of a Q table over the reduced action list, in the signature of the other agents."""

    def policy(env: GridEnv, loading: np.ndarray) -> int:
        return actions[int(np.argmax(q[state_index(loading, env.topology)]))]

    return policy


def compare(
    seeds: range = range(10), q: np.ndarray | None = None, actions: list[int] | None = None
) -> dict[str, list[dict]]:
    """Every agent on the same seeds; the Q policy is included when a table and its action list are given."""
    agents = {'do-nothing': do_nothing, 'random': random_agent, 'greedy (simulate)': greedy_agent}
    if q is not None and actions is not None:
        agents['Q-learning'] = q_policy(q, actions)
    return {name: [run_episode(GridEnv(seed), policy) for seed in seeds] for name, policy in agents.items()}


# --8<-- [end:agents]
def main() -> None:
    """Lab steps 1 to 3 on the five-bus grid."""
    print(f'{len(LOAD_SHARE)} buses, {len(LINES)} lines, {len(TOPOLOGIES)} topologies', end=', ')
    print(f'{GridEnv.n_actions} actions, {N_STATES} states')
    peak = load_profile(np.arange(N_STEPS)).max()
    print(f'peak load {peak:.1f} MW; loading at the peak, all lines closed: {np.round(rho(0, peak), 2)}')
    for t in range(1, len(TOPOLOGIES)):
        print(
            f'  topology {t} (line {LINES[~TOPOLOGIES[t]][0] + 1} open): max rho at the peak {rho(t, peak).max():.2f}'
        )
    actions = frequent_actions()
    print(
        f'most frequent greedy actions on 10 training days: {actions[1:]} (topologies {[a - 1 for a in actions[1:]]})'
    )
    t0 = time.perf_counter()
    q = q_learning(actions)
    print(f'Q-learning over {len(actions)} actions: 500 episodes in {time.perf_counter() - t0:.1f} s')
    results = compare(q=q, actions=actions)
    print(f'{"agent":18s} {"survival (median, min)":>24s} {"mean max rho":>13s} {"reward":>9s} {"ms/step":>8s}')
    for name, runs in results.items():
        surv = np.array([r['survival'] for r in runs])
        print(
            f'{name:18s} {np.median(surv):10.0f} {surv.min():12d} '
            f'{np.mean([r["mean_max_rho"] for r in runs]):13.3f} '
            f'{np.mean([r["reward"] for r in runs]):9.2f} '
            f'{np.mean([r["ms_per_step"] for r in runs]):8.3f}'
        )


if __name__ == '__main__':
    main()
