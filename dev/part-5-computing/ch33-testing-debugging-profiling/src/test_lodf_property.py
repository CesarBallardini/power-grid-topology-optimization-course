"""A property-based test: the LODF update equals re-solving the grid without the branch (Chapter 33)."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def ptdf_matrix(fr, to, x, n_bus, slack=0):
    a = np.zeros((len(fr), n_bus))
    a[np.arange(len(fr)), fr] = 1
    a[np.arange(len(fr)), to] = -1
    bf = a / x[:, None]
    b = a.T @ bf
    keep = np.arange(n_bus) != slack
    ptdf = np.zeros_like(a)
    ptdf[:, keep] = np.linalg.solve(b[np.ix_(keep, keep)], bf[:, keep].T).T
    return ptdf


@st.composite
def meshed_grids(draw, max_bus=8):
    n = draw(st.integers(3, max_bus))
    ring = [(i, (i + 1) % n) for i in range(n)]
    chords = draw(
        st.lists(st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)).filter(lambda e: e[0] != e[1]), max_size=4)
    )
    fr, to = map(np.array, zip(*(ring + chords), strict=True))
    x = draw(arrays(np.float64, len(fr), elements=st.floats(0.01, 1.0)))
    p = draw(arrays(np.float64, n, elements=st.floats(-1.0, 1.0)))
    return fr, to, x, p


@settings(max_examples=200, deadline=None)
@given(meshed_grids())
def test_lodf_matches_resolve(grid):
    fr, to, x, p = grid
    n = len(p)
    ptdf = ptdf_matrix(fr, to, x, n)
    f0 = ptdf @ p
    for b in range(len(fr)):
        col = ptdf[:, fr[b]] - ptdf[:, to[b]]
        lodf = col / (1.0 - col[b])
        lodf[b] = -1.0  # the outaged branch loses all its flow
        keep = np.arange(len(fr)) != b
        f_ref = np.zeros(len(fr))
        f_ref[keep] = ptdf_matrix(fr[keep], to[keep], x[keep], n) @ p
        np.testing.assert_allclose(f0 + lodf * f0[b], f_ref, atol=1e-8)
