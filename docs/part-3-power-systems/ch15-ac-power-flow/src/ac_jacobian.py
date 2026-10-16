"""Complex power injections and the four Jacobian blocks of the AC power flow, in matrix form (Chapter 15)."""

import numpy as np


def injections(theta: np.ndarray, vm: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Complex nodal injections S = [V] conj(Y V) for angles ``theta`` (rad), magnitudes ``vm`` and Ybus ``y``."""
    v = vm * np.exp(1j * theta)
    return v * np.conj(y @ v)


def jacobian_blocks(theta: np.ndarray, vm: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """dP/dtheta, dP/dV, dQ/dtheta, dQ/dV (J11, J12, J21, J22) for every bus, before removing slack and PV rows."""
    v = vm * np.exp(1j * theta)
    i = y @ v
    e = np.diag(v / vm)
    ds_dth = 1j * np.diag(v) @ np.conj(np.diag(i) - y @ np.diag(v))
    ds_dvm = np.diag(v) @ np.conj(y @ e) + np.diag(np.conj(i)) @ e
    return ds_dth.real, ds_dvm.real, ds_dth.imag, ds_dvm.imag


def main() -> None:
    """Print the injections and J11 of a two-bus example: a line of 0.01 + j0.1 pu between a source and a load."""
    y_line = 1 / (0.01 + 0.1j)
    y = np.array([[y_line, -y_line], [-y_line, y_line]])
    theta, vm = np.array([0.0, -0.05]), np.array([1.0, 0.98])
    print(np.round(injections(theta, vm, y), 4))
    print(np.round(jacobian_blocks(theta, vm, y)[0], 4))


if __name__ == '__main__':
    main()
