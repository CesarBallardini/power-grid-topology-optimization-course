import recompile


def test_fresh_lambda_retraces_on_every_call():
    traces, seconds = recompile.run(4, fresh_lambda=True, batch=8, n_branches=50)
    assert traces == 4
    assert len(seconds) == 4


def test_module_level_function_is_traced_once():
    traces, _ = recompile.run(4, fresh_lambda=False, batch=8, n_branches=50)
    assert traces == 1
