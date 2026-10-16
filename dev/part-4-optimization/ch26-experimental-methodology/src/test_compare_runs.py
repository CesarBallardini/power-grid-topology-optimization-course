import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from compare_runs import compare, plot_ecdf  # noqa: E402


def test_seeded_example():
    rng = np.random.default_rng(0)
    a, b = rng.normal(42.0, 3.0, size=10), rng.normal(39.0, 3.0, size=10)
    u, p, a12 = compare(a, b)
    assert u == 92.0 and abs(p - 0.0017) < 2e-4 and abs(a12 - 0.08) < 1e-9


def test_identical_samples_give_half():
    x = np.arange(10.0)
    _, p, a12 = compare(x, x)
    assert a12 == 0.5 and p > 0.99


def test_ecdf_plot_draws_one_line_per_configuration():
    fig, ax = plt.subplots()
    plot_ecdf({'A': np.arange(5.0), 'B': np.arange(5.0) + 1}, ax)
    assert len(ax.get_lines()) == 2
    plt.close(fig)
