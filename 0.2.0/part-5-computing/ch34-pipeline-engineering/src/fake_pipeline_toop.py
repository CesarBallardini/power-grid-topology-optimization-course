"""Run ToOp's DC worker on an in-process fake broker and feed its results to the AC worker's store (Chapter 34).

Lab steps 2 and 3 and Exercise 1. Run with ToOp's interpreter from this folder::

    ../../../../../ToOp/.venv/Scripts/python.exe fake_pipeline_toop.py

The worker reads the tracked preprocessed folder ``data/grid_node_breaker/grid`` and writes nothing to it; the
results stay in the fake producer's memory and then in an in-memory SQLite database.
"""

import contextlib
import importlib.util
import os
from pathlib import Path

import pandas as pd
from fsspec.implementations.dirfs import DirFileSystem
from sqlmodel import select
from toop_engine_interfaces.messages.protobuf_message_factory import deserialize_message, serialize_message
from toop_engine_topology_optimizer.ac.listener import poll_results_topic
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, FinishedOptimizations, create_session, scrub_db
from toop_engine_topology_optimizer.dc.worker.worker import Args
from toop_engine_topology_optimizer.dc.worker.worker import main as dc_main
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters, ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commands import Command, StartOptimizationCommand
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile
from toop_engine_topology_optimizer.interfaces.messages.dc_params import (
    BatchedMEParameters,
    DCOptimizerParameters,
    LoadflowSolverParameters,
)
from toop_engine_topology_optimizer.interfaces.messages.results import Result

TOOP = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
GRID_ROOT = TOOP / 'data' / 'grid_node_breaker'  # the GridFile's grid_folder "grid" is relative to this root


def fake_kafka():
    """ToOp's ``tests/fake_kafka.py`` loaded by path (it is a test helper, not part of the installed packages)."""
    path = TOOP / 'packages' / 'topology_optimizer_pkg' / 'tests' / 'fake_kafka.py'
    spec = importlib.util.spec_from_file_location('fake_kafka', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_command(runtime_seconds=6.0, seed=42, n_worst_contingencies=2, optimization_id='lab34'):
    """The command a backend would publish on ``commands``; the node-breaker grid has 11 N-1 cases, so k <= 11."""
    return Command(
        command=StartOptimizationCommand(
            ac_params=ACOptimizerParameters(ga_config=ACGAParameters(runtime_seconds=runtime_seconds, seed=seed)),
            dc_params=DCOptimizerParameters(
                ga_config=BatchedMEParameters(
                    iterations_per_epoch=2,
                    runtime_seconds=runtime_seconds,
                    random_seed=seed,
                    n_worst_contingencies=n_worst_contingencies,
                ),
                loadflow_solver_config=LoadflowSolverParameters(batch_size=8),
            ),
            grid_files=[GridFile(framework=Framework.PYPOWSYBL, grid_folder='grid')],
            optimization_id=optimization_id,
        )
    )


def run_dc_worker(command, grid_root=GRID_ROOT):
    """Run ``dc_main`` until the fake consumer runs dry; returns the producer's messages per topic."""
    kafka = fake_kafka()
    consumer = kafka.FakeConsumer({'commands': [serialize_message(command.model_dump_json())]}, kill_on_empty=True)
    producer = kafka.FakeProducer()
    with contextlib.suppress(kafka.FakeConsumerEmptyException):
        dc_main(
            Args(
                optimizer_command_topic='commands',
                optimizer_heartbeat_topic='heartbeats',
                optimizer_results_topic='results',
            ),
            processed_gridfile_fs=DirFileSystem(str(grid_root)),
            producer=producer,
            command_consumer=consumer,
        )
    return producer.messages


def decode(messages):
    """Protobuf envelope -> JSON -> ``Result`` for every message of a topic."""
    return [Result.model_validate_json(deserialize_message(m)) for m in messages]


def message_table(messages):
    """One row per result message: type, optimizer, timesteps, first timestep's actions and fitness, stop reason."""
    rows = []
    for result in decode(messages):
        inner = result.result
        strategy = getattr(inner, 'strategy', None) or getattr(inner, 'initial_topology', None)
        first = strategy.timesteps[0] if strategy is not None else None
        rows.append(
            {
                'type': type(inner).__name__,
                'optimizer': result.optimizer_type.value,
                'optimization_id': result.optimization_id,
                'timesteps': len(strategy.timesteps) if strategy is not None else None,
                'actions': first.actions if first is not None else None,
                'fitness': first.metrics.fitness if first is not None else None,
                'reason': getattr(inner, 'reason', None),
            }
        )
    return pd.DataFrame(rows)


def store_results(messages, session=None):
    """Feed result messages to the AC store through ``poll_results_topic``; returns (session, added, finished)."""
    session = session or create_session()
    added, finished = poll_results_topic(db=session, consumer=fake_kafka().FakeConsumer({'results': list(messages)}))
    return session, added, finished


def topology_table(session):
    """The ``ac_optim_topology`` table as a data frame (the hash shown in hex)."""
    rows = [t.model_dump() for t in session.exec(select(ACOptimTopology)).all()]
    table = pd.DataFrame(rows)
    if not table.empty:
        table['strategy_hash'] = table['strategy_hash'].map(lambda h: h.hex()[:12])
    return table


def main():
    """Run the worker, tabulate the results, store them, answer Exercise 1 and scrub."""
    messages = run_dc_worker(start_command())
    print({topic: len(msgs) for topic, msgs in messages.items()})
    table = message_table(messages['results'])
    print(table.to_string(index=False))
    print(table['type'].value_counts().to_dict())
    session, added, finished = store_results(messages['results'])
    topos = topology_table(session)
    print(f'\nwith the stop message: added={added} finished={finished}; rows stored {len(topos)}')
    print('columns:', topos.columns.tolist())
    print(
        topos[
            ['id', 'actions', 'fitness', 'unsplit', 'optimizer_type', 'worst_k_contingency_cases', 'strategy_hash']
        ].to_string(index=False)
    )
    print('finished_optimizations:', [f.model_dump() for f in session.exec(select(FinishedOptimizations)).all()])
    session2, added2, finished2 = store_results(messages['results'][:-1])
    print(f'without the stop message: added={added2} finished={finished2}')
    _, added3, _ = store_results(messages['results'][:-1], session2)
    print(f'replaying the same messages: added={added3} (duplicates rejected by the unique strategy hash)')
    scrub_db(session, max_age_seconds=0)
    print('rows after scrub_db(max_age_seconds=0):', len(topology_table(session)))


if __name__ == '__main__':
    main()
