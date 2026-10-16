"""ToOp's DC security metrics recomputed in NumPy on a three-branch example (Chapter 19)."""

import numpy as np

f_max = np.array([100.0, 200.0, 50.0])  # MW
n_0 = np.array([[80.0, -150.0, 49.0]])  # (timesteps, branches)
n_1 = np.array([[[95.0, -210.0, 40.0], [120.0, -160.0, 55.0]]])  # (timesteps, contingencies, branches)


def overload_energy_n1(n_1: np.ndarray, f_max: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Per branch the worst contingency's overload in MW (times its ``weight``), summed over branches and timesteps."""
    over = np.clip(np.abs(n_1) - f_max, 0, None)
    if weight is not None:
        over = over * weight
    return float(over.max(axis=1).sum())


def power_law_n1(n_1: np.ndarray, f_max: np.ndarray, alpha: float = 1.5, weight: np.ndarray | None = None) -> float:
    """Like ``overload_energy_n1`` but each overload is weighted as (relᵅ − 1) f_max."""
    rel = np.clip(np.abs(n_1) / f_max, 1, None)
    over = (rel**alpha - 1) * f_max
    if weight is not None:
        over = over * weight
    return float(over.max(axis=1).sum())


def critical_branch_count_n1(n_1: np.ndarray, f_max: np.ndarray) -> int:
    """Number of branches overloaded in at least one contingency, worst timestep."""
    return int(np.any(np.abs(n_1) > f_max, axis=1).sum(axis=1).max())


def double_limits(flows_tcb: np.ndarray, f_max: np.ndarray, lower: float = 0.9, upper: float = 1.0) -> np.ndarray:
    """Limited limits: 90 % of the rating below it, the rating above it, the flow itself in the dead zone."""
    flows = np.abs(flows_tcb).max(axis=(0, 1))
    low, high = flows < lower * f_max, flows > upper * f_max
    return np.where(low, lower * f_max, np.where(high, upper * f_max, flows))


def n0_n1_delta(n_0: np.ndarray, n_1: np.ndarray) -> np.ndarray:
    """Largest increase of a branch flow from the base case to any contingency, per timestep."""
    return np.clip(np.abs(n_1) - np.abs(n_0)[:, None, :], 0, None).max(axis=1)


def main() -> None:
    """Print the metrics of the worked example."""
    print(overload_energy_n1(n_1, f_max))  # 20 + 10 + 5 = 35 MW
    print(power_law_n1(n_1, f_max))  # 31.45 + 15.19 + 7.68 = 54.32
    print(critical_branch_count_n1(n_1, f_max))  # 3
    print(double_limits(n_0[:, None, :], f_max))  # [ 90. 180.  49.]  (49 MW is in the dead zone)
    print(n0_n1_delta(n_0, n_1))  # [[40. 60.  6.]]


if __name__ == '__main__':
    main()
