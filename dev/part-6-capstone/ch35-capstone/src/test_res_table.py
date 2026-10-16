import res_table


def test_node_breaker_runs_agree_and_fitness_is_the_negated_weighted_sum():
    for i in range(3):
        res = res_table.load_res(res_table.BASELINE / 'node_breaker' / f'run_{i}' / 'res.json')
        table = res_table.weighted_sum_check(res)
        assert table['matches'].all()
        assert res_table.target_metrics(res) == {'overload_energy_n_1': 1.0}
        assert table['actions'].tolist() == [[5]]
        assert abs(table['fitness'].iloc[0] + 42.3453259854399) < 1e-9


def test_operator_pick_respects_the_split_limit():
    res = res_table.load_res(res_table.NODE_BREAKER)
    table = res_table.best_topos_table(res)
    assert res_table.for_operator(table, 1)['actions'] == [5]
    assert res_table.for_operator(table, 0) is None  # the unsplit grid is not in best_topos


def test_every_baseline_res_passes_the_check():
    for path in sorted(res_table.BASELINE.glob('*/run_*/res.json')):
        table = res_table.weighted_sum_check(res_table.load_res(path))
        assert table['matches'].all(), path
        assert {'fitness', 'split_subs', 'overload_energy_n_1'} <= set(table.columns)
