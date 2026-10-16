"""DC state estimation on the three-bus ring of Chapter 16: WLS, residuals, J and chi-square test (Chapter 22, Ex. 5).

Measurements are the three branch flows and the two injections at the non-slack buses, so z = H theta + e with five
rows and two unknown angles (in the units of the Chapter 16 model, MW times pu reactance). The weighted least squares
estimate is theta = (H' W H)^-1 H' W z with W = diag(1/sigma^2); J(theta), the sum of (residual / sigma)^2, follows a
chi-square law with m - n degrees of freedom when the model is right.
"""

import numpy as np
from scipy.stats import chi2

FRM, TO = np.array([0, 1, 0]), np.array([1, 2, 2])
X = np.array([0.1, 0.1, 0.2])
P_TRUE = np.array([100.0, 0.0, -100.0])  # MW; bus 0 is the slack
SIGMA = np.array([2.0, 2.0, 2.0, 3.0, 3.0])  # MW: three flow meters, two injection meters (buses 1 and 2)


def measurement_matrix(frm: np.ndarray = FRM, to: np.ndarray = TO, x: np.ndarray = X) -> np.ndarray:
    """H (5 x 2): flows b_k (theta_f - theta_t) and injections at buses 1 and 2, with theta_0 = 0 dropped."""
    n = len(x)
    a = np.zeros((len(frm), n))
    a[np.arange(len(frm)), frm] = 1.0
    a[np.arange(len(frm)), to] = -1.0
    h_flow = a / x[:, None]
    bbus = a.T @ h_flow
    return np.vstack([h_flow, bbus[1:]])[:, 1:]


def true_measurements(h: np.ndarray, p: np.ndarray = P_TRUE) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free measurements and the angles that produce them."""
    bbus_red = h[3:]
    theta = np.linalg.solve(bbus_red, p[1:])
    return h @ theta, theta


def wls(h: np.ndarray, z: np.ndarray, sigma: np.ndarray = SIGMA) -> tuple[np.ndarray, np.ndarray, float]:
    """WLS estimate, residuals z - H theta and the weighted sum of squares J."""
    w = np.diag(1.0 / sigma**2)
    gain = h.T @ w @ h
    theta = np.linalg.solve(gain, h.T @ w @ z)
    r = z - h @ theta
    return theta, r, float(r @ w @ r)


def chi_square_test(j: float, m: int, n: int, confidence: float = 0.95) -> tuple[float, bool]:
    """Threshold for ``m - n`` degrees of freedom and whether J passes (no bad data suspected)."""
    threshold = float(chi2.ppf(confidence, m - n))
    return threshold, j <= threshold


def main() -> None:
    """Clean noisy measurements pass the test; a 30 MW error on one flow meter fails it."""
    rng = np.random.default_rng(16)
    h = measurement_matrix()
    z_true, theta_true = true_measurements(h)
    z = z_true + rng.normal(0.0, SIGMA)
    theta, r, j = wls(h, z)
    threshold, passes = chi_square_test(j, *h.shape)
    print('true angles:', np.round(theta_true, 4), ' estimate:', np.round(theta, 4))
    print('measurements:', np.round(z, 1), ' residuals:', np.round(r, 2))
    dof = h.shape[0] - h.shape[1]
    print(f'J = {j:.2f}, chi-square threshold at 95 % with {dof} degrees of freedom {threshold:.3f}, passes: {passes}')
    z_bad = z.copy()
    z_bad[0] += 30.0
    theta_bad, r_bad, j_bad = wls(h, z_bad)
    print(f'with a 30 MW error on the flow 0-1: J = {j_bad:.1f}, passes: {chi_square_test(j_bad, *h.shape)[1]}')
    print('largest normalized residual at measurement', int(np.argmax(np.abs(r_bad) / SIGMA)))


if __name__ == '__main__':
    main()
