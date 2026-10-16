"""Flow-based capacity calculation on a two-zone, four-bus grid: zonal PTDF and remaining margins (Chapter 21)."""

import numpy as np

BRANCHES = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]  # x = 1 pu each
GLSK = np.array([[0.6, 0.0], [0.4, 0.0], [0.0, 0.5], [0.0, 0.5]])  # rows: buses, columns: zones A, B
P_REF = np.array([300.0, -50.0, -150.0, -100.0])  # reference nodal injections, MW
F_MAX, FRM = 200.0, 10.0  # MW, from -> to direction


def nodal_ptdf(branches: list[tuple[int, int]], n_bus: int, slack: int = 0) -> np.ndarray:
    """PTDF (branches x buses) of a grid of unit reactances."""
    a = np.zeros((len(branches), n_bus))
    for k, (f, t) in enumerate(branches):
        a[k, f], a[k, t] = 1.0, -1.0
    ns = [i for i in range(n_bus) if i != slack]
    ptdf = np.zeros((len(branches), n_bus))
    ptdf[:, ns] = a[:, ns] @ np.linalg.inv((a.T @ a)[np.ix_(ns, ns)])
    return ptdf


def flow_based_domain(ptdf: np.ndarray, glsk: np.ndarray, p_ref: np.ndarray, f_max: float, frm: float):
    """Zonal PTDF, sensitivity to an A -> B exchange, remaining available margins and the exchange each CNEC allows."""
    ptdf_zone = ptdf @ glsk  # (CNECs, zones)
    ptdf_ab = ptdf_zone[:, 0] - ptdf_zone[:, 1]  # sensitivity to an A -> B exchange
    f_ref = ptdf @ p_ref
    np_ref = np.array([p_ref[:2].sum(), p_ref[2:].sum()])  # net positions A, B
    f_0 = f_ref - ptdf_zone @ np_ref  # flows with all net positions at zero
    ram = f_max - frm - f_0
    return ptdf_ab, ram, ram / ptdf_ab


def main() -> None:
    """Print the zone-to-zone sensitivities, the margins and the largest A -> B exchange per CNEC."""
    ptdf_ab, ram, exchange = flow_based_domain(nodal_ptdf(BRANCHES, 4), GLSK, P_REF, F_MAX, FRM)
    print(np.round(ptdf_ab, 3))  # [0.188 0.412 0.362 0.138 0.225]
    print(np.round(ram, 1))  # [ 99.4 130.6 218.1 186.9 221.2]
    print(np.round(exchange))  # [ 530.  317.  602. 1359.  983.]


if __name__ == '__main__':
    main()
