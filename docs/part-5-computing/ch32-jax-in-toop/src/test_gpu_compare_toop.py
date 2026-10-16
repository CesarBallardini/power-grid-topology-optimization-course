import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from gpu_compare_toop import (  # noqa: E402
    SmiSampler,
    agreement,
    differences,
    environment,
    grid_folder,
    measure_point,
    setup,
    to_float32,
)


@pytest.fixture(scope='module')
def si():
    return setup(grid_folder('case57'))


def test_environment_names_the_versions():
    env = environment()
    assert env['jax'] == jax.__version__ and env['backend'] == jax.default_backend()
    assert env['devices']


def test_to_float32_narrows_every_array_and_keeps_the_sentinel():
    big = np.iinfo(np.int64).max
    tree = {'f': jnp.ones(3, dtype=jnp.float64), 'i': jnp.array([1, big]), 'b': jnp.array([True])}
    out = to_float32(tree)
    assert out['f'].dtype == jnp.float32 and out['i'].dtype == jnp.int32 and out['b'].dtype == jnp.bool_
    assert int(out['i'][1]) == np.iinfo(np.int32).max


def test_smi_sampler_parses_a_reading():
    parsed = SmiSampler._parse(['2138', '1650', '35.07', '59', '97', '0x0000000000000004'])  # noqa: SLF001
    assert parsed[:5] == [2138.0, 1650.0, 35.07, 59.0, 97.0] and parsed[5] == 4.0
    assert np.isnan(SmiSampler._parse(['[N/A]', '1', '1', '1', '1', '[N/A]'])[0])  # noqa: SLF001


def test_differences_ignore_islanding_outages_but_count_the_missed_ones():
    ok = np.array([[True, False], [True, True]])
    a = {'success': np.ones((2, 2), bool), 'n_0': np.ones((2, 1, 3)), 'pf_n_1_max': np.ones((2, 1, 4))}
    a['n_1'] = np.ones((2, 1, 2, 3))
    a['n_1'][0, 0, 1] = 1e6  # garbage row of an islanding outage that ``a`` did not flag
    b = {'success': ok, 'n_0': np.ones((2, 1, 3)), 'pf_n_1_max': np.ones((2, 1, 4)), 'n_1': np.ones((2, 1, 2, 3))}
    d = differences(a, b)
    assert d['n_1_max_abs'] == 0 and d['topologies_compared_pf'] == 1
    assert d['missed_failures'] == 1 and d['missed_max_flow'] == 1e6 and not d['success_flags_equal']


def test_measure_point_reports_compile_time_and_throughput(si):
    row = measure_point(si, bsdf=4, injection=1, n_topologies=8, repeats=1)
    assert 'error' not in row, row
    assert row['compile_s'] > 0 and row['topologies_per_s'] > 0
    if jax.default_backend() == 'gpu':
        assert 0 < row['peak_device_mb'] <= row['device_limit_mb']


def test_both_devices_agree_in_float64_and_float32_is_close(si):
    r = agreement(si, n_topologies=8)
    exact = r['cpu_vs_gpu_float64']
    assert exact['success_flags_equal'] and exact['n_1_max_rel'] < 1e-12 and exact['n_0_max_rel'] < 1e-12
    single = r['gpu_float32_vs_float64']
    assert single['n_0_max_rel'] < 1e-4 and single['n_1_max_rel'] < 1e-4
