import numpy as np
from environment_check import device_kind, report, solve_laplacian, time_twice


class FakeDevice:
    def __init__(self, platform):
        self.platform = platform


def test_device_kind_reads_the_platform_attribute():
    assert device_kind([FakeDevice('cpu')]) == 'CPU'
    assert device_kind([FakeDevice('gpu')]) == 'GPU'
    assert device_kind([FakeDevice('cuda'), FakeDevice('cpu')]) == 'GPU'
    assert device_kind([]) == 'CPU'


def test_report_matches_jax_devices():
    r = report()
    assert set(r) == {'jax', 'backend', 'devices', 'kind', 'x64'}
    assert r['kind'] in ('CPU', 'GPU')
    assert r['backend'].lower() in r['devices'].lower() or r['kind'] == 'GPU'


def test_solve_laplacian_is_the_tridiagonal_solution():
    x = np.asarray(solve_laplacian(5))
    a = 2 * np.eye(5) - np.eye(5, k=1) - np.eye(5, k=-1)
    np.testing.assert_allclose(a @ x, np.ones(5), atol=1e-5)


def test_second_call_skips_compilation():
    first, second = time_twice(solve_laplacian, 300)
    assert first > 0 and second > 0
    assert second < first
