import numpy as np
import pytest
from dc_state_estimation import SIGMA, chi_square_test, measurement_matrix, true_measurements, wls


def test_noise_free_measurements_are_recovered_exactly():
    h = measurement_matrix()
    z, theta_true = true_measurements(h)
    np.testing.assert_allclose(z[:3], [50.0, 50.0, 50.0])  # the Chapter 16 flows
    np.testing.assert_allclose(z[3:], [0.0, -100.0], atol=1e-9)
    theta, r, j = wls(h, z)
    np.testing.assert_allclose(theta, theta_true)
    assert j == pytest.approx(0.0, abs=1e-18)


def test_chi_square_threshold_and_bad_datum():
    h = measurement_matrix()
    threshold, _ = chi_square_test(0.0, *h.shape)
    assert threshold == pytest.approx(7.8147, abs=1e-3)  # three degrees of freedom
    z, _ = true_measurements(h)
    z = z + np.random.default_rng(16).normal(0.0, SIGMA)
    assert chi_square_test(wls(h, z)[2], *h.shape)[1]
    z[0] += 30.0
    _, r, j = wls(h, z)
    assert not chi_square_test(j, *h.shape)[1]
    assert int(np.argmax(np.abs(r) / SIGMA)) == 0


def test_j_is_a_chi_square_sample_on_average():
    h = measurement_matrix()
    z_true, _ = true_measurements(h)
    rng = np.random.default_rng(0)
    js = [wls(h, z_true + rng.normal(0.0, SIGMA))[2] for _ in range(2000)]
    assert np.mean(js) == pytest.approx(3.0, abs=0.3)  # E[chi2_3] = 3
