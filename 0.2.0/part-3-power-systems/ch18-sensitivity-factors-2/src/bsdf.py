"""Bus split distribution factors: update a PTDF and the flows when one node is split in two (Chapter 18)."""

import numpy as np


def bsdf_split(h, frm, to, b, i, nb, bus_a, p):
    """PTDF and flows after splitting node ``i``: the branches in ``bus_a`` stay on ``i``, the others move to ``nb``.

    ``h`` is the PTDF with an extra column ``nb`` that copies column ``i`` (the new node starts out merged with
    ``i``), ``b`` the branch susceptances and ``p`` the nodal injections with ``p[nb] = 0``. The injections of
    node ``i`` stay on ``i``. Returns the updated PTDF and the branch flows.
    """
    a_from = [k for k in bus_a if frm[k] == i]
    a_to = [k for k in bus_a if to[k] == i]
    other = {k: (to[k] if frm[k] == i else frm[k]) for k in bus_a}
    p_c = h[a_to].sum(0) - h[a_from].sum(0)
    p_c[i] += 1.0  # i is not the slack
    num = sum(b[k] * (h[:, other[k]] - h[:, i]) for k in bus_a)
    for k in bus_a:
        num[k] += (1.0 if k in a_from else -1.0) * b[k]
    den = sum(b[k] for k in bus_a) - sum(b[k] * (p_c[other[k]] - p_c[nb]) for k in bus_a)
    beta = num / den
    c = p_c @ p
    return h + np.outer(beta, p_c), h @ p + c * beta  # updated PTDF and flows
