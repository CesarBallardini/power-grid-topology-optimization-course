import numpy as np
from coil_emf import (
    E_PEAK,
    OMEGA,
    emf_analytic,
    emf_numeric,
    flux,
    gradient_error,
    mean_power,
    periods_for_accuracy,
    rms,
    rms_window_error,
    rms_window_error_bound,
    sample_period,
)


def test_peak_and_rms_emf_match_exercise_3():
    assert abs(E_PEAK - 157.08) < 0.01
    t = sample_period(1000)
    assert abs(rms(emf_analytic(t)) - 111.07) < 0.01
    assert abs(rms(emf_analytic(t)) - E_PEAK / np.sqrt(2)) < 1e-9


def test_flux_and_emf_are_in_quadrature():
    t = sample_period(1000)
    assert abs(flux(t).max() - 0.5) < 1e-12 and abs(emf_analytic(t[0])) < 1e-12  # flux extreme where EMF is zero
    assert abs(np.mean(flux(t) * emf_analytic(t))) < 1e-9  # orthogonal over a period


def test_gradient_converges_to_the_analytic_emf():
    t = sample_period(1000)
    assert np.max(np.abs(emf_numeric(t) - emf_analytic(t))) < 0.5
    errors = [gradient_error(n) for n in (20, 100, 1000)]
    assert errors[0] > errors[1] > errors[2]
    assert abs(errors[1] / errors[2] - 10) < 0.5  # second-order inside, first-order at the two end points


def test_rms_of_components_add_in_squares():
    t = sample_period(1000)
    e1, e3 = emf_analytic(t), 40.0 * np.sin(3 * OMEGA * t)
    assert abs(rms(e3) - 40 / np.sqrt(2)) < 1e-9
    assert abs(rms(e1 + e3) - np.hypot(rms(e1), rms(e3))) < 1e-9


def test_mean_power_is_rms_squared_over_r():
    e = emf_analytic(sample_period(1000))
    assert abs(mean_power(e, 100.0) - rms(e) ** 2 / 100.0) < 1e-9
    assert abs(mean_power(e, 100.0) - 123.37) < 0.01


def test_exercise_5_window_not_sample_density_sets_the_error():
    assert abs(rms(325 * np.sin(OMEGA * sample_period(1000))) - 229.81) < 0.01
    for n in (3, 4, 10, 100):
        assert abs(rms_window_error(10, n, 325)) < 1e-12  # integer windows: exact from 3 samples per period
    e10, e1000 = rms_window_error(10.4, 10, 325), rms_window_error(10.4, 1000, 325)
    assert 1e-3 < e10 < e1000 < rms_window_error_bound(10.4)  # more samples do not help
    assert abs(rms_window_error(40.4, 1000, 325)) < 1e-3 < abs(rms_window_error(10.4, 1000, 325))
    assert abs(periods_for_accuracy(1e-3) - 39.8) < 0.1
