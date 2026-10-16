import numpy as np
from lab_network import fr, g, to
from laplacian import incidence, laplacian


def test_lab_network_is_connected_with_six_nodes():
    assert len(fr) == len(to) == len(g) == 7
    gmat = laplacian(incidence(np.array(fr), np.array(to), 6), np.array(g, float))
    w = np.linalg.eigvalsh(gmat)
    assert np.sum(np.abs(w) < 1e-9) == 1  # exactly one zero eigenvalue: connected
