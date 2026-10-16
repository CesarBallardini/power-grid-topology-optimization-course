from contextlib import closing

import mini_pipeline
import pytest


def test_dc_run_produces_started_pushes_and_stopped():
    msgs = mini_pipeline.dc_run().messages['results']
    kinds = [mini_pipeline.Result.model_validate_json(m).kind for m in msgs]
    assert kinds == ['started', 'push', 'push', 'push', 'push', 'push', 'stopped']


def test_poll_deletes_the_count_of_a_stopped_optimization_and_rejects_duplicates():
    msgs = mini_pipeline.dc_run().messages['results']
    with closing(mini_pipeline.create_session()) as db, closing(mini_pipeline.create_session()) as db2:
        added, stopped = mini_pipeline.poll_results(db, mini_pipeline.FakeConsumer({'results': msgs}))
        assert added == {} and stopped == ['demo']
        assert db.execute('SELECT COUNT(*) FROM topology').fetchone()[0] == 4  # five pushes, one repeated (3,)
        added, stopped = mini_pipeline.poll_results(db2, mini_pipeline.FakeConsumer({'results': msgs[:-1]}))
        assert added == {'demo': 4} and stopped == []
        added, _ = mini_pipeline.poll_results(db2, mini_pipeline.FakeConsumer({'results': msgs[:-1]}))
        assert added == {}


def test_scrub_and_kill_on_empty():
    msgs = mini_pipeline.dc_run().messages['results']
    with closing(mini_pipeline.create_session()) as db:
        mini_pipeline.poll_results(db, mini_pipeline.FakeConsumer({'results': msgs}))
        assert mini_pipeline.scrub(db, max_age_seconds=3600) == 4
        assert mini_pipeline.scrub(db, max_age_seconds=0) == 0
    consumer = mini_pipeline.FakeConsumer({'results': msgs}, kill_on_empty=True)
    assert len(consumer.consume()) == 7
    with pytest.raises(mini_pipeline.FakeConsumerEmptyError):
        consumer.consume()
