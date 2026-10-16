import numpy as np
import planted
import pytest

EXPECTED = {
    # check: (correct, psdf_sign, float32, wrong_slack)
    'oracle, signed flows, atol 1e-6 MW': (True, False, False, False),
    'oracle, |flows|, atol 1e-6 MW': (True, False, False, False),
    'oracle, signed flows, atol 1e-2 MW': (True, False, True, False),
    'oracle at zero shift, atol 1e-6 MW': (True, True, False, False),
    'slack column of the PTDF is zero': (True, True, True, False),
    'LODF N-1 equals re-solve, atol 1e-8': (True, True, False, True),
}


@pytest.fixture(scope='module')
def data():
    return planted.ieee14()


def test_correct_code_matches_the_oracle_with_a_phase_shift(data):
    np.testing.assert_allclose(planted.n0_flows(data), data.flows, rtol=0, atol=1e-9)


def test_lodf_n1_matches_resolve_with_phase_shift(data):
    f1, outages = planted.n1_flows(data)
    f1_ref, outages_ref = planted.resolved_n1_flows(data)
    assert outages.tolist() == outages_ref.tolist()
    assert len(outages) == len(data.fr) - 1  # one bridge: the transformer feeding bus 8
    np.testing.assert_allclose(f1, f1_ref, rtol=0, atol=1e-9)


@pytest.mark.parametrize('bug', [None, *planted.BUGS])
def test_symptom_table(data, bug):
    column = (None, *planted.BUGS).index(bug)
    table = planted.symptoms(data, bug)
    assert {k: v[column] for k, v in EXPECTED.items()} == table


def test_every_bug_fails_the_signed_oracle_and_each_has_a_check_that_misses_it(data):
    for bug in planted.BUGS:
        table = planted.symptoms(data, bug)
        assert not table['oracle, signed flows, atol 1e-6 MW']
        assert any(table.values()), bug


def test_localization_points_at_the_faulty_component(data):
    loc = {bug: planted.localize(data, bug) for bug in planted.BUGS}
    assert loc['psdf_sign']['PTDF'] == 0.0 and loc['psdf_sign']['PSDF (from the correct PTDF)'] > 100
    assert loc['psdf_sign']['N-0 flows at zero shift'] < 1e-9
    assert 1e-9 < loc['float32']['PTDF'] < 1e-6 and loc['float32']['N-0 flows at 10 degrees'] < 1e-4
    assert loc['wrong_slack']['PTDF'] > 0.5 and loc['wrong_slack']['PSDF (from the correct PTDF)'] == 0.0
