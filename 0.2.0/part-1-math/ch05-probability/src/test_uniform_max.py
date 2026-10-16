import pytest
import uniform_max as um


def test_exact_mean():
    assert um.exact_mean_of_max(1) == 0.5
    assert um.exact_mean_of_max(10) == pytest.approx(10 / 11)
    assert um.exact_mean_of_max(100) == pytest.approx(100 / 101)


def test_simulation_is_within_a_few_standard_errors():
    table = um.comparison(seed=0)
    assert list(table['n']) == [1, 10, 100]
    for _, row in table.iterrows():
        assert abs(row['simulated'] - row['exact']) < 4 * row['std_error']
    assert table['P(max > 0.9)'].to_numpy() == pytest.approx([0.1, 1 - 0.9**10, 1 - 0.9**100], abs=0.005)
    assert um.simulated_mean_of_max(10, seed=0) == pytest.approx(10 / 11, abs=2e-3)
