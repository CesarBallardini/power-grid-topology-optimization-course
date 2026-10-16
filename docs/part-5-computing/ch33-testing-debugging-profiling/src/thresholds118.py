"""ToOp's three singularity denominators on IEEE 118 when every reactance is scaled by 10 and by 0.1 (Exercise 1).

The LODF denominator 1 - (PTDF[b, f_b] - PTDF[b, t_b]) and the MODF matrix I - PTDF_OO are built from PTDF entries,
which do not change when all reactances are multiplied by the same factor; the BSDF denominator is a sum of
susceptances times (1 - PTDF differences), so it scales with 1/s.
"""

import dcgrid
import numpy as np
import pandapower.networks as pn

THRESHOLDS = {'LODF': 1e-11, 'BSDF': 1e-5, 'determinant': 1e-10}


def ieee118():
    """IEEE 118 as pandapower builds it: 186 branches, 118 buses."""
    return dcgrid.pandapower_dc(pn.case118())


def scaled(data, factor):
    """The same grid with every reactance multiplied by ``factor`` (susceptances divided by it)."""
    return dcgrid.DCData(
        data.fr, data.to, data.b / factor, data.shift, data.p, data.slack, data.n_bus, data.base_mva, data.flows
    )


def lodf_denominators(data):
    """1 - (PTDF[b, f_b] - PTDF[b, t_b]) for every branch b; 0 for a bridge."""
    h = dcgrid.ptdf_matrix(data.fr, data.to, data.b, data.n_bus, data.slack)
    return dcgrid.lodf_matrix(h, data.fr, data.to)[1]


def bsdf_denominator(h, fr, to, b, node, kept):
    """BSDF denominator when ``kept`` (branch indices) stay on ``node`` and the other branches at ``node`` move.

    With p_c the PTDF-difference vector of the branches that stay (plus one at the node, which is not the slack)
    and the new busbar a copy of the node's column, the denominator is sum over kept branches k of
    b_k (1 - (p_c[other end of k] - p_c[node])), the formula of Chapter 18 and of ToOp's numpy reference.
    """
    kept = np.asarray(kept)
    other = np.where(fr[kept] == node, to[kept], fr[kept])
    p_c = h[kept[to[kept] == node]].sum(axis=0) - h[kept[fr[kept] == node]].sum(axis=0)
    p_c[node] += 1.0
    return float(np.sum(b[kept] * (1.0 - (p_c[other] - p_c[node]))))


def bsdf_denominators(data, n_kept=2):
    """One split per bus with at least four branches: the first ``n_kept`` branches stay, the rest move."""
    h = dcgrid.ptdf_matrix(data.fr, data.to, data.b, data.n_bus, data.slack)
    out = {}
    for node in range(data.n_bus):
        at = np.flatnonzero((data.fr == node) | (data.to == node))
        if len(at) >= 4 and node != data.slack:
            out[node] = bsdf_denominator(h, data.fr, data.to, data.b, node, at[:n_kept])
    return out


def modf_determinants(data):
    """det(I - PTDF_OO) for every pair of consecutive branches (o, o + 1), ToOp's 2x2 multi-outage case."""
    h = dcgrid.ptdf_matrix(data.fr, data.to, data.b, data.n_bus, data.slack)
    col = h[:, data.fr] - h[:, data.to]  # col[k, o]: flow on k per unit transfer across o
    dets = []
    for o in range(len(data.fr) - 1):
        pair = [o, o + 1]
        dets.append(np.linalg.det(np.identity(2) - col[np.ix_(pair, pair)]))
    return np.array(dets)


def experiment(factors=(0.1, 1.0, 10.0)):
    """Smallest |denominator| of each kind per scaling factor, and how many fall below ToOp's threshold."""
    base = ieee118()
    rows = {}
    for s in factors:
        data = scaled(base, s)
        lodf = np.abs(lodf_denominators(data))
        bsdf = np.abs(np.array(list(bsdf_denominators(data).values())))
        det = np.abs(modf_determinants(data))
        rows[s] = {
            'LODF': (lodf[lodf > THRESHOLDS['LODF']].min(), int((lodf <= THRESHOLDS['LODF']).sum()), len(lodf)),
            'BSDF': (bsdf.min(), int((bsdf < THRESHOLDS['BSDF']).sum()), len(bsdf)),
            'determinant': (
                det[det > THRESHOLDS['determinant']].min(),
                int((det <= THRESHOLDS['determinant']).sum()),
                len(det),
            ),
        }
    return rows


def ratios(base, s):
    """Scaled over unscaled denominators (smallest, largest) of each kind, leaving out the ones below threshold."""
    data = scaled(base, s)
    lodf_1, lodf_s = lodf_denominators(base), lodf_denominators(data)
    keep = np.abs(lodf_1) > THRESHOLDS['LODF']
    det_1, det_s = modf_determinants(base), modf_determinants(data)
    keep_det = np.abs(det_1) > THRESHOLDS['determinant']
    bsdf_1, bsdf_s = bsdf_denominators(base), bsdf_denominators(data)
    bsdf = np.array([bsdf_s[k] / bsdf_1[k] for k in bsdf_1])
    return {
        'LODF': (float((lodf_s[keep] / lodf_1[keep]).min()), float((lodf_s[keep] / lodf_1[keep]).max())),
        'BSDF': (float(bsdf.min()), float(bsdf.max())),
        'determinant': (
            float((det_s[keep_det] / det_1[keep_det]).min()),
            float((det_s[keep_det] / det_1[keep_det]).max()),
        ),
    }


def main():
    """Print the experiment table and the ratios between the scaled and the unscaled denominators."""
    rows = experiment()
    print(
        f'{"factor":>8s} {"kind":12s} {"min |denominator| above threshold":>34s} {"below threshold":>16s} {"of":>5s}'
    )
    for s, kinds in rows.items():
        for kind, (smallest, below, total) in kinds.items():
            print(f'{s:8.1f} {kind:12s} {smallest:34.6e} {below:16d} {total:5d}')
    base = ieee118()
    for s in (0.1, 10.0):
        print(
            f'x{s:g}: ratio scaled/unscaled (min, max):',
            {k: (round(lo, 12), round(hi, 12)) for k, (lo, hi) in ratios(base, s).items()},
        )
    bridges = int((lodf_denominators(base) == 0).sum())
    print(f'bridges (LODF denominator exactly 0 in double precision): {bridges}')


if __name__ == '__main__':
    main()
