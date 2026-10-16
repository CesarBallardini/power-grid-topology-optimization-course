import numpy as np
import pytest
from ch27_ieee30 import PENALTY, enumerate_front, load_case, map_elites, nsga2, overload_energy, ptdf


@pytest.fixture(scope='module')
def case():
    return load_case()


def test_network_data(case):
    assert case['n_bus'] == 30 and len(case['lines']) == 41
    assert case['rating'][0] == pytest.approx(130, abs=0.1)
    assert case['p'].sum() == pytest.approx(0)
    h = ptdf(case, np.ones(41, dtype=bool))
    f = h @ case['p']
    a = np.zeros((41, 30))
    a[np.arange(41), case['lines'][:, 0]] = 1
    a[np.arange(41), case['lines'][:, 1]] = -1
    np.testing.assert_allclose(a.T @ f, case['p'], atol=1e-9)  # Kirchhoff at every bus


def test_overload_energy_values(case):
    assert overload_energy(case) == pytest.approx(267.0, abs=0.1)
    assert overload_energy(load_case(scale=1.0)) < 1e-6  # the unscaled case is N-1 secure
    assert overload_energy(case, (34,)) == pytest.approx(171.3, abs=0.1)
    assert overload_energy(case, (32, 40)) == pytest.approx(44.6, abs=0.1)
    assert overload_energy(case, (12,)) == np.inf  # line 9-11 is the only line of bus 11: islanded


def test_front_up_to_one_line(case):
    front, sets, count = enumerate_front(case, k_max=1)
    assert count == 42
    np.testing.assert_allclose(front[:, 1], [0, 1])
    assert sets == [(), (34,)]


def test_heuristics_return_non_dominated_finite_fronts(case):
    for front in (nsga2(case, 0, evaluations=120, pop=12), map_elites(case, 0, evaluations=120)):
        assert (front[:, 0] < PENALTY).all()
        assert len(set(front[:, 1].tolist())) == len(front)  # one point per number of switched lines
