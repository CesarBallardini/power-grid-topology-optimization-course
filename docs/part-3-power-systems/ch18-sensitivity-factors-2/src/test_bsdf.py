import numpy as np
import pytest
from bsdf import bsdf_split


def ptdf(frm, to, b, n, slack=0):
    a = np.zeros((len(frm), n))
    a[np.arange(len(frm)), frm], a[np.arange(len(frm)), to] = 1, -1
    bf = b[:, None] * a
    bbus = a.T @ bf
    keep = np.arange(n) != slack
    h = np.zeros_like(a)
    h[:, keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], bf[:, keep].T).T
    return h


# five buses, seven branches, node 2 has four branches; the new node gets index 5
FRM = np.array([0, 0, 1, 1, 2, 2, 3])
TO = np.array([1, 2, 2, 3, 3, 4, 4])
B = np.array([10.0, 5.0, 8.0, 4.0, 6.0, 3.0, 2.0])
P = np.array([1.0, 0.2, -0.5, -0.3, -0.4, 0.0])


@pytest.mark.parametrize('bus_a', [[1, 4], [2, 5], [1, 2], [4, 5]])
def test_split_matches_a_fresh_ptdf_of_the_split_grid(bus_a):
    i, nb = 2, 5
    h = ptdf(FRM, TO, B, 5)
    h_ext = np.hstack([h, h[:, [i]]])  # extra column nb, a copy of column i
    h_new, f_new = bsdf_split(h_ext, FRM, TO, B, i, nb, bus_a, P)

    frm2, to2 = FRM.copy(), TO.copy()
    for k in range(len(FRM)):
        if k not in bus_a:  # branches not in bus_a leave node i for node nb
            if frm2[k] == i:
                frm2[k] = nb
            elif to2[k] == i:
                to2[k] = nb
    h_ref = ptdf(frm2, to2, B, 6)
    np.testing.assert_allclose(f_new, h_ref @ P, atol=1e-9)
    np.testing.assert_allclose(h_new @ P, h_ref @ P, atol=1e-9)
