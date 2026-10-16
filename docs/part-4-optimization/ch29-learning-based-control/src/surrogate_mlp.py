"""A two-layer MLP in JAX as a surrogate of the IEEE 30 DC flows (Chapter 29, Lab alternative track).

Random injection vectors are turned into flows with the PTDF of ``ch27_ieee30`` (the Chapter 27 folder is put on
``sys.path``); the network learns injections -> flows, its per-line error and speed are reported, and then one line
not seen in training is switched off to measure the generalization gap. Run it with::

    python surrogate_mlp.py
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ch27-pareto-multi-objective' / 'src'))
import ch27_ieee30  # noqa: E402

jax.config.update('jax_enable_x64', True)

N_SAMPLES = 20_000
HIDDEN = 64
STEPS = 1500
SWITCHED_OFF = 34  # line 25-27, the best single switch of Chapter 27, unseen in training


# --8<-- [start:data]
def dataset(case: dict, n: int, seed: int, in_service: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Random injections (loads and generation scaled by 0.5 to 1.5 per bus, slack balancing) and their DC flows.

    Flows of an open line are zero, so the output always has one value per line of the intact network.
    """
    rng = np.random.default_rng(seed)
    mask = np.ones(len(case['lines']), dtype=bool) if in_service is None else in_service
    h = ch27_ieee30.ptdf(case, mask)
    p = case['p'][None, :] * rng.uniform(0.5, 1.5, (n, case['n_bus']))
    p[:, 0] = -p[:, 1:].sum(axis=1)
    flows = np.zeros((n, len(case['lines'])))
    flows[:, mask] = p @ h.T
    return p[:, 1:], flows  # the slack injection is implied


# --8<-- [end:data]
# --8<-- [start:mlp]
def init_params(n_in: int, n_out: int, hidden: int, key) -> dict:
    """Two dense layers with tanh in between, Glorot-style scaling."""
    k1, k2 = jax.random.split(key)
    return {
        'w1': jax.random.normal(k1, (n_in, hidden)) * jnp.sqrt(2 / (n_in + hidden)),
        'b1': jnp.zeros(hidden),
        'w2': jax.random.normal(k2, (hidden, n_out)) * jnp.sqrt(2 / (hidden + n_out)),
        'b2': jnp.zeros(n_out),
    }


def predict(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Normalized injections in, normalized flows out."""
    return jnp.tanh(x @ params['w1'] + params['b1']) @ params['w2'] + params['b2']


def train(
    x: np.ndarray, y: np.ndarray, steps: int = STEPS, hidden: int = HIDDEN, seed: int = 0, lr: float = 3e-3
) -> dict:
    """Full-batch Adam on the mean squared error of the normalized targets; returns params and the scalers."""
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-9
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-9
    xn, yn = jnp.asarray((x - x_mean) / x_std), jnp.asarray((y - y_mean) / y_std)
    params = init_params(x.shape[1], y.shape[1], hidden, jax.random.PRNGKey(seed))
    opt = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in params.items()}

    def loss(p):
        return jnp.mean((predict(p, xn) - yn) ** 2)

    @jax.jit
    def step(p, m_v, t):
        g = jax.grad(loss)(p)
        new_p, new_mv = {}, {}
        for k in p:
            m = 0.9 * m_v[k][0] + 0.1 * g[k]
            v = 0.999 * m_v[k][1] + 0.001 * g[k] ** 2
            m_hat, v_hat = m / (1 - 0.9**t), v / (1 - 0.999**t)
            new_p[k] = p[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            new_mv[k] = (m, v)
        return new_p, new_mv

    for t in range(1, steps + 1):
        params, opt = step(params, opt, t)
    return {'params': params, 'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}


def surrogate_flows(model: dict, x: np.ndarray) -> np.ndarray:
    """Flows in MW from raw injections."""
    xn = (x - model['x_mean']) / model['x_std']
    return np.asarray(predict(model['params'], jnp.asarray(xn))) * model['y_std'] + model['y_mean']


def relative_error(pred: np.ndarray, true: np.ndarray, rating: np.ndarray) -> np.ndarray:
    """Mean absolute error per line divided by the line's rating, in percent."""
    return 100 * np.abs(pred - true).mean(axis=0) / rating


# --8<-- [end:mlp]
def main() -> None:
    """Train on the intact network, report error and speed, then switch off one line."""
    case = ch27_ieee30.load_case(scale=1.0)
    x, y = dataset(case, N_SAMPLES, seed=1)
    x_test, y_test = dataset(case, 2000, seed=2)
    t0 = time.perf_counter()
    model = train(x, y)
    print(f'trained {STEPS} steps on {N_SAMPLES} samples in {time.perf_counter() - t0:.1f} s')
    err = relative_error(surrogate_flows(model, x_test), y_test, case['rating'])
    print(f'intact network, test error per line: mean {err.mean():.3f} % of rating', end=', ')
    print(f'worst {err.max():.3f} % (line {err.argmax()})')
    h = ch27_ieee30.ptdf(case, np.ones(41, dtype=bool))
    p_full = np.c_[-x_test.sum(axis=1), x_test]
    t0 = time.perf_counter()
    for _ in range(20):
        exact = p_full @ h.T
    t_exact = (time.perf_counter() - t0) / 20
    surrogate_flows(model, x_test)  # warm-up
    t0 = time.perf_counter()
    for _ in range(20):
        surrogate_flows(model, x_test)
    t_mlp = (time.perf_counter() - t0) / 20
    print(f'2000 cases: exact PTDF product {1e3 * t_exact:.2f} ms, MLP {1e3 * t_mlp:.2f} ms', end=', ')
    print(f'speed-up {t_exact / t_mlp:.2f}x (exact and dataset flows agree: {np.allclose(exact, y_test)})')
    mask = np.ones(41, dtype=bool)
    mask[SWITCHED_OFF] = False
    x_off, y_off = dataset(case, 2000, seed=3, in_service=mask)
    err_off = relative_error(surrogate_flows(model, x_off), y_off, case['rating'])
    i, j = case['lines'][SWITCHED_OFF] + 1
    print(f'line {i}-{j} switched off (unseen): error per line mean {err_off.mean():.2f} %', end=', ')
    print(f'worst {err_off.max():.2f} % (line {err_off.argmax()})')
    print(f'  ratio of mean errors, switched / intact: {err_off.mean() / err.mean():.0f}x')


if __name__ == '__main__':
    main()
