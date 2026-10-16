"""Bus split distribution factors on random meshed grids, with an optional off-by-one in the busbar mask (Exercise 4).

``split_ptdf`` applies the Chapter 18 update to a PTDF when one node is split: the branches flagged in ``to_b``
move to a new busbar (the last column), the others and the node's injection stay. ``rebuilt_ptdf`` builds the
split network from scratch. Hypothesis compares the two and checks Kirchhoff's current law at both busbars; with
``off_by_one=True`` the mask is shifted by one branch, the planted fault, and ``smallest_counterexample`` reports
the simplest grid on which it shows.
"""

import dcgrid
import numpy as np
from hypothesis import find, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def branches_at(fr, to, node):
    """Indices of the branches with one end at ``node``."""
    return np.flatnonzero((fr == node) | (to == node))


def split_ptdf(h, fr, to, b, node, to_b, off_by_one=False):
    """PTDF after splitting ``node``, with the new busbar as the last column; ``to_b`` flags the branches that move."""
    at = branches_at(fr, to, node)
    mask = np.roll(to_b, 1) if off_by_one else np.asarray(to_b)
    kept = at[~mask]
    nb = h.shape[1]
    h_ext = np.column_stack([h, h[:, node]])
    other = np.where(fr[kept] == node, to[kept], fr[kept])
    p_c = h_ext[kept[to[kept] == node]].sum(axis=0) - h_ext[kept[fr[kept] == node]].sum(axis=0)
    p_c[node] += 1.0
    num = (b[None, kept] * (h_ext[:, other] - h_ext[:, [node]])).sum(axis=1)
    num[kept] += np.where(fr[kept] == node, 1.0, -1.0) * b[kept]
    den = np.sum(b[kept] * (1.0 - (p_c[other] - p_c[nb])))
    return h_ext + np.outer(num / den, p_c)


def rebuilt_ptdf(fr, to, b, n_bus, node, to_b, slack=0):
    """The PTDF of the split network built directly: moved branches are re-attached to the new busbar."""
    at = branches_at(fr, to, node)
    fr2, to2 = fr.copy(), to.copy()
    for k in at[np.asarray(to_b)]:
        if fr2[k] == node:
            fr2[k] = n_bus
        else:
            to2[k] = n_bus
    return dcgrid.ptdf_matrix(fr2, to2, b, n_bus + 1, slack)


def kcl_residuals(h_split, fr, to, p, node, to_b):
    """Net flow leaving each busbar minus its injection: zero when Kirchhoff's current law holds."""
    at = branches_at(fr, to, node)
    flows = h_split @ np.append(p, 0.0)
    sign = np.where(fr[at] == node, 1.0, -1.0)  # +1 when the flow leaves the node through its from end
    out_a = np.sum(sign[~np.asarray(to_b)] * flows[at[~np.asarray(to_b)]])
    out_b = np.sum(sign[np.asarray(to_b)] * flows[at[np.asarray(to_b)]])
    return out_a - p[node], out_b


@st.composite
def split_grids(draw, max_bus=7):
    """A ring with random chords, a non-slack node with four or more branches, and a mask that moves 1..d-1 of them."""
    n = draw(st.integers(3, max_bus))
    ring = [(i, (i + 1) % n) for i in range(n)]
    chords = draw(
        st.lists(
            st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)).filter(lambda e: e[0] != e[1]),
            min_size=2,
            max_size=5,
        )
    )
    fr, to = (np.array(v) for v in zip(*(ring + chords), strict=True))
    degree = np.bincount(np.concatenate([fr, to]), minlength=n)
    candidates = [v for v in range(1, n) if degree[v] >= 4]
    if not candidates:
        draw(st.nothing())
    node = draw(st.sampled_from(candidates))
    d = int(degree[node])
    to_b = draw(arrays(bool, d).filter(lambda m: 0 < m.sum() < d))
    b = draw(arrays(np.float64, len(fr), elements=st.floats(1.0, 10.0)))
    p = draw(arrays(np.float64, n, elements=st.floats(-1.0, 1.0)))
    return fr, to, b, p, node, to_b


def violation(grid, off_by_one=False):
    """Largest violation of the two properties on one grid: PTDF mismatch and KCL residual."""
    fr, to, b, p, node, to_b = grid
    h = dcgrid.ptdf_matrix(fr, to, b, len(p))
    h_split = split_ptdf(h, fr, to, b, node, to_b, off_by_one)
    mismatch = np.abs(h_split - rebuilt_ptdf(fr, to, b, len(p), node, to_b)).max()
    return float(mismatch), float(np.abs(kcl_residuals(h_split, fr, to, p, node, to_b)).max())


def smallest_counterexample():
    """The simplest grid on which the off-by-one breaks the PTDF property, shrunk by Hypothesis."""
    return find(
        split_grids(),
        lambda g: violation(g, off_by_one=True)[0] > 1e-6,
        settings=settings(max_examples=500, database=None, derandomize=True),
    )


def main():
    """Print one shrunk counterexample and its violations."""
    fr, to, b, p, node, to_b = smallest_counterexample()
    print(f'smallest counterexample: {len(p)} buses, {len(fr)} branches')
    print('  from', fr.tolist(), 'to', to.tolist())
    print('  susceptances', b.tolist(), 'injections', p.tolist())
    print(f'  split node {node}, branches at it {branches_at(fr, to, node).tolist()}, moved mask {to_b.tolist()}')
    good, bad = violation((fr, to, b, p, node, to_b)), violation((fr, to, b, p, node, to_b), off_by_one=True)
    print(f'  correct code: PTDF mismatch {good[0]:.1e}, KCL residual {good[1]:.1e}')
    print(f'  off by one:   PTDF mismatch {bad[0]:.3f}, KCL residual {bad[1]:.3f}')


if __name__ == '__main__':
    main()
