"""ToOp's message-to-database hop in sixty lines: a fake broker, a Pydantic result and a SQLite store (Chapter 34).

This is the shape of ``packages/topology_optimizer_pkg/tests/fake_kafka.py``, ``optimizer/ac/listener.py`` and
``optimizer/ac/storage.py`` with the standard library only, so that the mechanisms of the chapter run without ToOp:
a producer that keeps messages per topic, a consumer that replays them and raises when it runs dry, a store that
rejects duplicates through a unique hash and forgets old rows, and a poll function that returns the count of
rows added per optimization, minus the optimizations that have already stopped.
"""

import hashlib
import json
import sqlite3
import time
from typing import Literal

from pydantic import BaseModel


class Topology(BaseModel):
    """One candidate: the action indices and its fitness."""

    actions: list[int]
    fitness: float


class Result(BaseModel):
    """What travels on the ``results`` topic: a push, or a stopped notice."""

    optimization_id: str
    kind: Literal['started', 'push', 'stopped']
    topology: Topology | None = None


class FakeProducer:
    """Stores the bytes it would have sent, per topic."""

    def __init__(self):
        """Start with no topics."""
        self.messages: dict[str, list[bytes]] = {}

    def produce(self, topic, value):
        """Append ``value`` to ``topic``."""
        self.messages.setdefault(topic, []).append(value)


class FakeConsumerEmptyError(Exception):
    """Raised by a consumer with ``kill_on_empty`` once every topic has been read: the worker loop's exit."""


class FakeConsumer:
    """Replays stored messages; with ``kill_on_empty`` it raises instead of returning an empty batch."""

    def __init__(self, messages, kill_on_empty=False):
        """``messages`` maps topic to a list of bytes."""
        self.messages = messages
        self.offsets = dict.fromkeys(messages, 0)
        self.kill_on_empty = kill_on_empty

    def consume(self, num_messages=10_000):
        """The next messages, across topics, in order."""
        out = []
        for topic, msgs in self.messages.items():
            take = msgs[self.offsets[topic] : self.offsets[topic] + num_messages - len(out)]
            self.offsets[topic] += len(take)
            out.extend(take)
        if not out and self.kill_on_empty:
            raise FakeConsumerEmptyError('no more messages to consume')
        return out


def create_session():
    """An in-memory SQLite database with the topology table and a unique strategy hash, as ``create_session`` does."""
    db = sqlite3.connect(':memory:')
    db.execute(
        'CREATE TABLE topology (id INTEGER PRIMARY KEY, optimization_id TEXT, actions TEXT, fitness REAL, '
        'strategy_hash TEXT UNIQUE, created_at REAL)'
    )
    db.execute('CREATE TABLE finished (optimization_id TEXT PRIMARY KEY, finished_at REAL)')
    return db


def strategy_hash(topology):
    """The unique key of a topology: a hash of its actions, independent of the message that carried it."""
    return hashlib.sha256(json.dumps(sorted(topology.actions)).encode()).hexdigest()


def poll_results(db, consumer):
    """Store every pushed topology; return rows added per optimization and the optimizations that stopped.

    Duplicates raise ``IntegrityError`` on the unique hash and are skipped. Like ToOp's ``poll_results_topic``,
    the count of an optimization whose ``stopped`` message is in the same batch is deleted before returning.
    """
    added, stopped = {}, []
    for raw in consumer.consume():
        result = Result.model_validate_json(raw)
        if result.kind == 'stopped':
            stopped.append(result.optimization_id)
            db.execute('INSERT OR IGNORE INTO finished VALUES (?, ?)', (result.optimization_id, time.time()))
            continue
        if result.topology is None:
            continue
        try:
            db.execute(
                'INSERT INTO topology (optimization_id, actions, fitness, strategy_hash, created_at) '
                'VALUES (?, ?, ?, ?, ?)',
                (
                    result.optimization_id,
                    json.dumps(result.topology.actions),
                    result.topology.fitness,
                    strategy_hash(result.topology),
                    time.time(),
                ),
            )
            added[result.optimization_id] = added.get(result.optimization_id, 0) + 1
        except sqlite3.IntegrityError:
            pass
    for optimization_id in stopped:
        added.pop(optimization_id, None)
    return added, stopped


def scrub(db, max_age_seconds=86_400):
    """Delete topologies older than ``max_age_seconds`` (ToOp keeps one day)."""
    db.execute('DELETE FROM topology WHERE created_at < ?', (time.time() - max_age_seconds,))
    return db.execute('SELECT COUNT(*) FROM topology').fetchone()[0]


def dc_run(optimization_id='demo', topologies=((), (3,), (5,), (3,), (4,))):
    """A DC worker in miniature: one started message, one push per topology (with a repeat), one stopped message."""
    producer = FakeProducer()
    producer.produce('results', Result(optimization_id=optimization_id, kind='started').model_dump_json().encode())
    for i, actions in enumerate(topologies):
        topo = Topology(actions=list(actions), fitness=-60.0 + 5.0 * i)
        producer.produce(
            'results', Result(optimization_id=optimization_id, kind='push', topology=topo).model_dump_json().encode()
        )
    producer.produce('results', Result(optimization_id=optimization_id, kind='stopped').model_dump_json().encode())
    return producer


def main():
    """Run the miniature end to end and print what Exercise 1 asks about."""
    producer = dc_run()
    msgs = producer.messages['results']
    print(f'{len(msgs)} messages on "results":', [Result.model_validate_json(m).kind for m in msgs])
    db = create_session()
    print('with the stopped message   : added', *poll_results(db, FakeConsumer({'results': msgs})))
    print(
        'rows stored                :',
        db.execute('SELECT COUNT(*) FROM topology').fetchone()[0],
        '(one duplicate rejected)',
    )
    db2 = create_session()
    print('without the stopped message: added', *poll_results(db2, FakeConsumer({'results': msgs[:-1]})))
    print('replaying the same messages: added', *poll_results(db2, FakeConsumer({'results': msgs[:-1]})))
    print('rows after scrub(max_age_seconds=0):', scrub(db, 0))
    consumer = FakeConsumer({'results': msgs}, kill_on_empty=True)
    consumer.consume()
    try:
        consumer.consume()
    except FakeConsumerEmptyError as err:
        print('second consume() ->', type(err).__name__, err)
    db.close()
    db2.close()


if __name__ == '__main__':
    main()
