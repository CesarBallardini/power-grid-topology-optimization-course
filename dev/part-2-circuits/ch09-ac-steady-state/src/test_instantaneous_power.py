import numpy as np
from instantaneous_power import mean_and_swing, sample_power


def test_mean_is_active_power_and_swing_is_apparent_power():
    for phi in (0.0, 36.87, 60.0, 90.0):
        _, _, _, p = sample_power(230.0, 10.0, phi)
        mean, swing = mean_and_swing(p)
        np.testing.assert_allclose(mean, 2300 * np.cos(np.radians(phi)), atol=1e-6)
        np.testing.assert_allclose(swing, 2300.0, rtol=1e-5)


def test_power_oscillates_at_twice_the_frequency():
    t, _, _, p = sample_power(230.0, 10.0, 30.0, f_hz=50.0, n=4000)
    spectrum = np.abs(np.fft.rfft(p - p.mean()))
    assert np.argmax(spectrum) == 2  # second harmonic of the period
    assert p.min() < 0  # energy flows back to the source twice a period
    assert t[-1] < 0.02
