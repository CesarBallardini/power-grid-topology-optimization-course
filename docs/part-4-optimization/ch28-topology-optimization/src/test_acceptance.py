import json

from acceptance import evaluate_acceptance, run_table

UNSPLIT = {'non_converging_loadflows': 2, 'overload_energy_n_1': 10.0, 'critical_branch_count_n_1': 4}


def test_accepted_when_every_check_passes():
    assert (
        evaluate_acceptance(
            {'non_converging_loadflows': 2, 'overload_energy_n_1': 9.5, 'critical_branch_count_n_1': 4}, UNSPLIT
        )
        is None
    )
    assert (
        evaluate_acceptance(
            {
                'non_converging_loadflows': 3,
                'disconnected_branches': 1,
                'overload_energy_n_1': 0,
                'critical_branch_count_n_1': 0,
            },
            UNSPLIT,
        )
        is None
    )


def test_checks_run_in_order():
    r = evaluate_acceptance(
        {'non_converging_loadflows': 3, 'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 0}, UNSPLIT
    )
    assert r['criterion'] == 'convergence' and r['value_after'] == 3 and r['threshold'] == 1.0
    r = evaluate_acceptance(
        {'non_converging_loadflows': 2, 'overload_energy_n_1': 9.6, 'critical_branch_count_n_1': 0}, UNSPLIT
    )
    assert r['criterion'] == 'overload-energy'
    r = evaluate_acceptance(
        {'non_converging_loadflows': 2, 'overload_energy_n_1': 1.0, 'critical_branch_count_n_1': 5}, UNSPLIT
    )
    assert r['criterion'] == 'critical-branch-count' and r['value_before'] == 4


def test_zero_baseline_trap_and_exercise_5():
    zero = {'non_converging_loadflows': 2, 'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 0}
    r = evaluate_acceptance(
        {'non_converging_loadflows': 1, 'overload_energy_n_1': 0.4, 'critical_branch_count_n_1': 1}, zero
    )
    assert r == {'criterion': 'overload-energy', 'value_after': 0.4, 'value_before': 0.0, 'threshold': 0.95}
    assert (
        evaluate_acceptance(
            {'non_converging_loadflows': 1, 'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 1}, zero
        )['criterion']
        == 'critical-branch-count'
    )


def test_voltage_checks_only_when_enabled():
    split = {
        'non_converging_loadflows': 2,
        'overload_energy_n_1': 1.0,
        'critical_branch_count_n_1': 1,
        'voltage_jump_count_n_1': 5,
    }
    unsplit = {**UNSPLIT, 'voltage_jump_count_n_1': 1, 'critical_va_diff_count_n_1': 0}
    assert evaluate_acceptance(split, unsplit) is None
    assert evaluate_acceptance(split, unsplit, critical_voltage=True)['criterion'] == 'voltage-magnitude'


def test_run_table(tmp_path):
    (tmp_path / 'unsplit_ac_metrics.json').write_text(
        json.dumps({'fitness': -10.0, 'extra_scores': UNSPLIT}), encoding='utf-8'
    )
    for i, (overload, fitness) in enumerate([(5.0, -5.0), (9.9, -9.9)]):
        folder = tmp_path / f'topology_{i}'
        folder.mkdir()
        res = {
            'fitness': fitness,
            'extra_scores': {
                'non_converging_loadflows': 2,
                'overload_energy_n_1': overload,
                'critical_branch_count_n_1': 4,
            },
            'dc_info': {'metrics': {'fitness': -1.0 - i}},
        }
        (folder / 'ac_metrics.json').write_text(json.dumps(res), encoding='utf-8')
    table = run_table(tmp_path)
    assert table['rejected_by'].tolist() == [None, 'overload-energy']
    assert table['overload_ok'].tolist() == [True, False]
    assert table['dc_rank'].tolist() == [1.0, 2.0]
