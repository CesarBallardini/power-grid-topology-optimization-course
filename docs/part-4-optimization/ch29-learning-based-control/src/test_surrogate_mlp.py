import numpy as np
import pytest
from surrogate_mlp import ch27_ieee30, dataset, relative_error, surrogate_flows, train


@pytest.fixture(scope='module')
def case():
    return ch27_ieee30.load_case(scale=1.0)


def test_dataset_is_balanced_and_linear(case):
    x, y = dataset(case, 50, seed=0)
    assert x.shape == (50, 29) and y.shape == (50, 41)
    h = ch27_ieee30.ptdf(case, np.ones(41, dtype=bool))
    p = np.c_[-x.sum(axis=1), x]
    np.testing.assert_allclose(y, p @ h.T, atol=1e-9)


def test_small_training_run_learns_the_linear_map(case):
    x, y = dataset(case, 2000, seed=1)
    model = train(x, y, steps=300, hidden=32)
    x_test, y_test = dataset(case, 200, seed=2)
    err = relative_error(surrogate_flows(model, x_test), y_test, case['rating'])
    assert err.mean() < 2.0  # percent of rating
    mask = np.ones(41, dtype=bool)
    mask[34] = False
    x_off, y_off = dataset(case, 200, seed=3, in_service=mask)
    assert relative_error(surrogate_flows(model, x_off), y_off, case['rating']).mean() > 3 * err.mean()
