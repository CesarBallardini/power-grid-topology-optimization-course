import numpy as np
import pytest
from line_switching import (
    brute_force,
    count_sets,
    dc_power_flow_check,
    fitness,
    greedy,
    interaction,
    is_connected,
    load_case30,
    n1_flows,
    overload_energy,
    per_line_energy,
    worst_loading,
)


@pytest.fixture(scope='module')
def case():
    return load_case30()


def test_case30_dimensions_and_scaling(case):
    assert case.n_bus == 30 and case.n_branch == 41
    assert case.names[0] == 'L1-2' and case.names[40] == 'L6-28'
    unscaled = load_case30(1.0)
    np.testing.assert_allclose(case.p, 1.4 * unscaled.p)
    assert overload_energy(unscaled) == pytest.approx(0.0, abs=1e-9)
    assert overload_energy(load_case30(1.2)) == pytest.approx(16.972, abs=1e-3)


def test_base_flows_balance_and_lodf_agrees_with_power_flow(case):
    f0, f1, bridge = n1_flows(case)
    assert f1.shape == (41, 41) and int(bridge.sum()) == 3
    assert overload_energy(case) == pytest.approx(145.351, abs=1e-3)
    assert worst_loading(case) == pytest.approx(140.0, abs=0.05)
    for switched, outage in (((), 0), ((32, 40), 5), ((34,), 10)):
        assert dc_power_flow_check(case, switched, outage) < 1e-9


def test_islanding_sets_get_minus_infinity(case):
    assert not is_connected(case, (15,))  # L12-13 is the only link to bus 13
    assert fitness(case, (15,)) == -np.inf
    assert is_connected(case, (32, 40))


def test_brute_force_two_lines(case):
    bf = brute_force(case, 2)
    assert bf.n_evaluations == count_sets(41, 2) == 862
    assert bf.n_islanding == 146
    best, value = bf.best
    assert best == (32, 40) and -value == pytest.approx(27.343, abs=1e-3)


def test_non_additive_pair_and_greedy_path(case):
    e0, ea, eb, eab = interaction(case, 32, 40)
    assert (e0, ea, eb, eab) == pytest.approx((145.351, 94.869, 138.910, 27.343), abs=1e-3)
    assert eab < ea + eb - e0  # the pair does far better than the sum of the single-line gains
    assert per_line_energy(case)['L6-8'] == pytest.approx(102.07, abs=0.01)
    assert 'L6-8' not in per_line_energy(case, (32, 40))
    path, energies = greedy(case, 3)
    assert path == [34, 40, 38] and energies[-1] == pytest.approx(28.883, abs=1e-3)
