import numpy as np
from metrics import (
    critical_branch_count_n1,
    double_limits,
    f_max,
    n0_n1_delta,
    n_0,
    n_1,
    overload_energy_n1,
    power_law_n1,
)


def test_worked_example_values():
    assert overload_energy_n1(n_1, f_max) == 35.0
    assert abs(power_law_n1(n_1, f_max) - 54.32) < 0.01
    assert critical_branch_count_n1(n_1, f_max) == 3
    np.testing.assert_allclose(double_limits(n_0[:, None, :], f_max), [90.0, 180.0, 49.0])
    np.testing.assert_allclose(n0_n1_delta(n_0, n_1), [[40.0, 60.0, 6.0]])


def test_overload_energy_sums_per_branch_not_per_contingency():
    # two contingencies overloading different branches by 10 MW each: per-branch sum gives 20, per-timestep max 10
    n1 = np.array([[[110.0, 0.0], [0.0, 110.0]]])
    assert overload_energy_n1(n1, np.array([100.0, 100.0])) == 20.0
    assert overload_energy_n1(np.zeros((1, 2, 2)), np.array([100.0, 100.0])) == 0.0
