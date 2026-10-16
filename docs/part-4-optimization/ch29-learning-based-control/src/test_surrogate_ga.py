import numpy as np
import pytest
from surrogate_ga import K_MAX, PENALTY, TARGET, ch27_ieee30, exact, features, offspring, random_genome, run


@pytest.fixture(scope='module')
def case():
    return ch27_ieee30.load_case()


def test_features_and_exact(case):
    g = np.array([32, 40])
    f = features(g, 41)
    assert f.sum() == 3 and f[-1] == 1 and f[32] == 1 and f[40] == 1
    assert exact(case, g, 41) == pytest.approx(44.63, abs=0.01)
    assert exact(case, g, 41) <= TARGET
    assert exact(case, np.array([12, 41]), 41) == PENALTY  # islanding is penalized, not infinite


def test_genomes_are_well_formed(case):
    rng = np.random.default_rng(0)
    parents = np.array([random_genome(rng, 41) for _ in range(6)])
    for _ in range(50):
        g = offspring(rng, parents, 41)
        assert g.shape == (K_MAX,)
        real = g[g < 41]
        assert len(np.unique(real)) == len(real)
        assert (np.sort(g) == g).all()


def test_short_runs(case):
    plain = run(case, 0, assisted=False, budget=60, pop=10)
    assisted = run(case, 0, assisted=True, budget=60, pop=10)
    for r in (plain, assisted):
        assert r['evaluations'] == 60
        assert r['best'] < PENALTY
        assert r['reached_at'] is None or r['reached_at'] <= 60
