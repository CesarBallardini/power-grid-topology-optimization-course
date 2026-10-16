import numpy as np
import pareto
import pytest
import toop_front


def test_every_run_is_stored_and_fitness_is_the_weighted_sum():
    for name in [f'{c}_s{s}' for c in ('fine', 'coarse') for s in toop_front.SEEDS]:
        assert len(toop_front.table(name)) == (29 if name.startswith('fine') else 17)
    for w in toop_front.WEIGHTS:
        t = toop_front.weight_runs()[w]
        expected = -(t['overload_energy_n_1'] + w * t['critical_branch_count_n_1'])
        assert np.allclose(t['fitness'], expected, atol=1e-5)


def test_lab1_the_weighted_sum_winner_is_only_weakly_efficient():
    r = toop_front.lab1()
    assert (r['elites'], r['non_dominated'], r['dominated']) == (29, 6, 23)
    t = toop_front.table('fine_s1')
    best = t.loc[r['best']]
    assert best['actions'] == [1, 54, 68, 118] and not r['best_non_dominated']
    # a topology with the same overload, one split and two switching operations fewer dominates it
    same = t[(t['overload_energy_n_1'] == best['overload_energy_n_1']) & (t['split_subs'] == 3)]
    assert [1, 54, 68] in same['actions'].tolist()
    coarse = toop_front.lab1('coarse_s1')
    assert coarse['edge_elites'] == 1 and coarse['edge_elites_dominated'] == 1


def test_lab3_three_unsupported_points_reached_by_epsilon_constraint():
    r = toop_front.lab3()
    assert len(r['front']) == 8 and int((~r['supported']).sum()) == 3
    for point, pick in r['epsilon'].items():
        assert pick == point
    assert r['winners'][0] == r['best_known'][0]
    assert r['dominated_winners'] == [3]
    for p in r['unsupported']:  # no weight w >= 0 selects an unsupported point of the joint front
        for w in np.linspace(0, 200, 401):
            assert not np.array_equal(r['front'][pareto.weighted_sum_argmin(r['front'], np.array([1, w]))[0]], p)


def test_lab4_both_resolutions_reach_the_same_front_but_coarse_runs_more_epochs():
    r = toop_front.lab4()
    assert r.attrs['joint_hv'] == pytest.approx(0.836404688, abs=1e-8)
    assert np.allclose(r.loc[r['config'] == 'coarse', 'hv'], r.attrs['joint_hv'])
    worst = r.loc[r['hv'].idxmin()]
    assert worst['run'] == 'fine_s4' and worst['igd_plus'] > 0
    assert toop_front.compare(r, 'epochs')['p'] < 0.01
    assert toop_front.compare(r, 'hv')['p'] > 0.5


def test_lab7_knee_and_achievement_function_pick_the_one_split_topology():
    r = toop_front.lab7()
    assert [toop_front.as_tuple(p) for p in r['knees']] == [(1184.119297, 2.0)]
    assert toop_front.as_tuple(r['asf']) == r['best_feasible'] == (1184.119297, 2.0, 1.0)
    assert np.allclose(r['pseudo_weights'].sum(axis=1), 1.0)
    assert r['weighted_sum_winner'] == (1160.36575, 10.0, 4.0)
