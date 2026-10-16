import numpy as np
import pytest

pytestmark = pytest.mark.labs
pytest.importorskip('qdax')

import ch27_ieee30 as ieee30  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from mome_labs import (  # noqa: E402
    build_table,
    cell_list,
    exact_cells,
    make_mutation,
    make_scoring,
    moqd_score,
    n1_metrics,
    repertoire_points,
    run_mome,
)


@pytest.fixture(scope='module')
def case():
    return ieee30.load_case()


@pytest.fixture(scope='module')
def table(case):
    return build_table(case)


def test_n1_metrics_matches_lab_6(case):
    for switched in [(), (34,), (32, 40), (5, 17, 33)]:
        energy, critical = n1_metrics(case, switched)
        assert energy == pytest.approx(ieee30.overload_energy(case, switched))
        assert 0 <= critical <= 41 if np.isfinite(energy) else critical == -1
    assert n1_metrics(case) == (pytest.approx(267.0, abs=0.1), 11)
    assert n1_metrics(case, (12,)) == (np.inf, -1)  # bus 11 islanded


def test_table_and_exact_cells(table):
    assert len(table['feasible']) == 8220
    assert len(cell_list()) == 7
    exact = exact_cells(table)
    scale = table['scale']
    cell_35 = exact[(3, 5)] * scale
    np.testing.assert_allclose(cell_35, [[25.2, 7], [149.3, 6]], atol=0.05)
    assert sum(len(f) for f in exact.values()) == 9
    assert moqd_score(exact.values()) == pytest.approx(2.0353, abs=1e-4)


def test_scoring_and_mutation(table):
    genotypes = jnp.asarray(table['feasible'][:64], dtype=jnp.float32)
    fitness, descriptor, _ = make_scoring(table)(genotypes, jax.random.key(0))
    g = tuple(table['feasible'][:64].T)
    np.testing.assert_allclose(fitness, table['fitness'][g], rtol=1e-6)
    np.testing.assert_allclose(descriptor, table['descriptor'][g])
    children = np.asarray(make_mutation(table)(genotypes, jax.random.key(1))).astype(int)
    n_line = table['n_line']
    for child in children:
        assert np.all(np.diff(child) >= 0)  # sorted
        lines = child[child < n_line]
        assert len(np.unique(lines)) == len(lines)  # no duplicates
        assert np.isfinite(table['energy'][tuple(child)])  # never islanded


def test_short_run_stores_table_values(table):
    repertoire, metrics = run_mome(table, seed=0, iterations=3)
    genotypes = np.round(np.asarray(repertoire.genotypes)).astype(int)
    fitnesses = np.asarray(repertoire.fitnesses)
    filled = np.all(np.isfinite(fitnesses), axis=-1)
    assert filled.any()
    stored = genotypes[filled]
    np.testing.assert_allclose(fitnesses[filled], table['fitness'][tuple(np.sort(stored, axis=1).T)], rtol=1e-6)
    assert moqd_score(repertoire_points(repertoire)) <= moqd_score(exact_cells(table).values()) + 1e-9
