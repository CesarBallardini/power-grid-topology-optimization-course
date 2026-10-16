import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from performance_profile import plot_profile, profile, ratios, synthetic_times  # noqa: E402


def test_small_table_by_hand():
    times = pd.DataFrame({'s1': [1, 2, 3, 1], 's2': [2, 1, 3, np.inf], 's3': [np.nan, 4, 1, 2]})
    r = ratios(times)
    assert r['s3'].tolist()[0] == np.inf and r['s1'].tolist() == [1, 2, 3, 1]
    rho = profile(times, np.array([1.0, 2.0, np.inf]))
    assert rho['s1'].tolist() == [0.5, 0.75, 1.0]
    assert rho['s2'].tolist() == [0.25, 0.5, 0.75]
    assert rho['s3'].tolist() == [0.25, 0.5, 0.75]


def test_synthetic_profile_numbers():
    times = synthetic_times()
    assert times.shape == (20, 3) and int(times['GA'].isna().sum()) == 7
    rho = profile(times, np.array([1.0, 2.0, 5.0, np.inf]))
    assert rho.loc[1.0].tolist() == pytest.approx([0.30, 0.20, 0.50])
    assert rho.loc[5.0].tolist() == pytest.approx([0.65, 1.00, 0.95])
    assert rho.loc[np.inf].tolist() == pytest.approx([0.65, 1.0, 1.0])  # rho(inf): fraction solved, 7 GA failures
    fig, ax = plt.subplots()
    plot_profile(times, ax=ax)
    assert len(ax.get_lines()) == 3
    plt.close(fig)
