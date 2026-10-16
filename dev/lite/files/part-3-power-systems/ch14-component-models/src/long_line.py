"""Exact long-line (ABCD) parameters against the nominal pi model as a function of length (Chapter 14, Exercise 5).

For a line with series impedance z and shunt admittance y per km, the propagation constant is gamma = sqrt(z y)
and the characteristic impedance Zc = sqrt(z / y). Over a length l the exact two-port has A = D = cosh(gamma l),
B = Zc sinh(gamma l) and C = sinh(gamma l) / Zc, and the exact equivalent pi has series impedance
Z' = Z sinh(gamma l) / (gamma l) and half-shunts (Y/2) tanh(gamma l / 2) / (gamma l / 2), where Z = z l and
Y = y l are the nominal-pi values. The relative error of the nominal series admittance is
|gamma l / sinh(gamma l) − 1|.
"""

import numpy as np

# a typical single-circuit 400 kV overhead line at 50 Hz
R_PER_KM = 0.028  # ohm/km
X_PER_KM = 2 * np.pi * 50 * 0.9e-3  # ohm/km, from 0.9 mH/km
B_PER_KM = 2 * np.pi * 50 * 12.5e-9  # S/km, from 12.5 nF/km
LENGTH_KM = 300.0


def abcd(length_km: float, r: float = R_PER_KM, x: float = X_PER_KM, b: float = B_PER_KM) -> np.ndarray:
    """The exact ABCD matrix of a lossy line of the given length."""
    z, y = r + 1j * x, 1j * b
    gamma, zc = np.sqrt(z * y), np.sqrt(z / y)
    gl = gamma * length_km
    return np.array([[np.cosh(gl), zc * np.sinh(gl)], [np.sinh(gl) / zc, np.cosh(gl)]])


def nominal_pi(
    length_km: float, r: float = R_PER_KM, x: float = X_PER_KM, b: float = B_PER_KM
) -> tuple[complex, complex]:
    """Series impedance Z = z l and total shunt admittance Y = y l of the nominal pi model."""
    return (r + 1j * x) * length_km, 1j * b * length_km


def exact_pi(
    length_km: float, r: float = R_PER_KM, x: float = X_PER_KM, b: float = B_PER_KM
) -> tuple[complex, complex]:
    """Series impedance and total shunt admittance of the exact equivalent pi, from the ABCD parameters."""
    m = abcd(length_km, r, x, b)
    series = m[0, 1]  # B = Z'
    shunt_half = (m[0, 0] - 1) / m[0, 1]  # (A - 1) / B = Y'/2
    return series, 2 * shunt_half


def series_admittance_error(
    length_km: np.ndarray, r: float = R_PER_KM, x: float = X_PER_KM, b: float = B_PER_KM
) -> np.ndarray:
    """Relative error |Z/Z' − 1| of the nominal series admittance 1/Z against the exact 1/Z', elementwise."""
    gl = np.sqrt((r + 1j * x) * 1j * b) * np.asarray(length_km, dtype=float)
    return np.abs(gl / np.sinh(gl) - 1)


def main() -> None:
    """Print the ABCD parameters at 300 km, the nominal and exact pi values and the error at several lengths."""
    z, y = R_PER_KM + 1j * X_PER_KM, 1j * B_PER_KM
    gamma, zc = np.sqrt(z * y), np.sqrt(z / y)
    print(f'z = {z:.4f} ohm/km, y = {y.imag:.3e}j S/km, Zc = {zc:.1f} ohm, gamma = {gamma:.3e} /km')
    print(f'wavelength 2 pi / Im(gamma) = {2 * np.pi / gamma.imag:.0f} km')
    m = abcd(LENGTH_KM)
    print(f'ABCD at {LENGTH_KM:.0f} km: A = {m[0, 0]:.4f}, B = {m[0, 1]:.2f} ohm, C = {m[1, 0]:.3e} S')
    print(f'  AD - BC = {np.linalg.det(m):.6f}')
    (z_nom, y_nom), (z_ex, y_ex) = nominal_pi(LENGTH_KM), exact_pi(LENGTH_KM)
    print(f'nominal pi: Z = {z_nom:.2f} ohm, Y = {y_nom.imag:.3e}j S')
    print(f'exact pi:   Z = {z_ex:.2f} ohm, Y = {y_ex.imag:.3e}j S')
    error_300 = 100 * series_admittance_error(LENGTH_KM)
    print(f'relative error of the series admittance at {LENGTH_KM:.0f} km: {error_300:.2f} %')
    lengths = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    errors = series_admittance_error(lengths)
    print('length km   ' + ' '.join(f'{v:7.0f}' for v in lengths))
    print('error %     ' + ' '.join(f'{100 * v:7.2f}' for v in errors))
    print(f'quadratic rule of thumb |gamma l|^2 / 6 at 300 km: {100 * abs(gamma * LENGTH_KM) ** 2 / 6:.2f} %')


if __name__ == '__main__':
    main()
