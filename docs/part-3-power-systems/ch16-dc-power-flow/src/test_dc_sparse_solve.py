import numpy as np
import scipy.sparse.linalg as spla
from dc_sparse_solve import flows_many, flows_one_by_one, susceptance_matrices
from dc_vs_ac import dc_case


def test_block_solve_equals_separate_solves_and_rundcpp():
    _, table, p0, slack = dc_case('case14')
    n_bus = int(max(table['from'].max(), table['to'].max())) + 1
    b = 1 / (table['x'].to_numpy() * table['tap'].to_numpy())
    bbus, bf = susceptance_matrices(table['from'].to_numpy(), table['to'].to_numpy(), b, n_bus)
    assert bbus.shape == (14, 14) and bf.shape == (20, 14)
    np.testing.assert_allclose(bbus.sum(axis=1), 0, atol=1e-12)
    keep = np.arange(n_bus) != slack
    rng = np.random.default_rng(1)
    p = p0 + rng.normal(scale=5.0, size=(7, n_bus))
    p[0] = p0
    p[:, slack] -= p.sum(axis=1)
    many = flows_many(spla.splu(bbus[keep][:, keep]), bf[:, keep], p, keep)
    np.testing.assert_allclose(many, flows_one_by_one(bbus[keep][:, keep], bf[:, keep], p, keep), atol=1e-9)
    np.testing.assert_allclose(many[0], table['p_pandapower'], atol=1e-9)
