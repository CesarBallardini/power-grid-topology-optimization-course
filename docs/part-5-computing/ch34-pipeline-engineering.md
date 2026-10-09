# Chapter 34. Engineering the pipeline

**Weeks:** 2 · **Semester:** 5 · **Prerequisites:** [Chapter 25](../part-4-optimization/ch25-quality-diversity.md), [Chapter 30](ch30-scientific-python.md)

!!! info "Why it matters for ToOp"

    Stages communicate through a processed grid folder whose file names are
    centralized in `interfaces/folder_structure.py`; large matrices are stored in HDF5; parameters and Kafka
    messages are Pydantic models; data frames are validated with Pandera; DC and AC workers run concurrently over
    Kafka; pandapower N-1 runs in parallel with Ray (`docs/architecture/`).

    Around that core sit the tools you meet as soon as you run or change the pipeline:

    - The AC worker keeps every topology it has seen in an **in-memory SQLite database** through SQLModel
      (`optimizer/ac/storage.py`), so its state lives only as long as the process.
    - The DC optimizer logs fitness and metrics per epoch with **tensorboardX** and is started through a **tyro**
      command line (`optimizer/dc/main.py`); the AC load flow worker has its own tyro CLI
      (`contingency/ac_loadflow_service/lf_worker.py`).
    - Benchmarks sweep parameters with **Hydra** multiruns (`docs/benchmark.md`, `toop-engine-benchmark/`).
    - Integration tests replace Kafka with an **in-process fake broker**
      (`packages/topology_optimizer_pkg/tests/fake_kafka.py`), which is also the easiest way to watch the message
      flow yourself.

## Topics

- Schemas and validation: Pydantic models for parameters and messages; Pandera for data frames (ToOp normalizes
  dtypes explicitly rather than relying on `coerce=True`, so behavior holds with validation disabled).
- Storage: HDF5 with h5py (`dc_solver/jax/inputs.py` saves and loads `StaticInformation`), JSON, fsspec file systems
  (`DirFileSystem`); the processed grid folder and its constants `PREPROCESSING_PATHS`, `POSTPROCESSING_PATHS`,
  `NETWORK_MASK_NAMES`, `OUTPUT_FILE_NAMES` (`interfaces/folder_structure.py`).
- Relational state inside a worker: SQLModel tables `ACOptimTopology` and `FinishedOptimizations` on an in-memory
  SQLite engine (`"sqlite://"`, `StaticPool`, `check_same_thread=False`) created by `create_session()`;
  `scrub_db()` deletes topologies older than a day; duplicates are rejected through a unique strategy hash and an
  `IntegrityError` (`optimizer/ac/storage.py`, `optimizer/ac/listener.py`).
- Message-driven architecture: six Kafka topics (`importer_commands`, `importer_results`, `importer_heartbeat`,
  `commands`, `results`, `heartbeat`), with `results` shared by both stages and consumed by the AC validator
  (`docs/architecture/model/02-messaging.c4`); a single-field Protobuf `MessageWrapper` around Pydantic JSON
  (`interfaces/messages/protobuf_message_factory.py`); the broker for local runs in `dev-deployment/docker-compose.yaml`.
- Testing message flows without a broker: `FakeProducer` stores messages per topic, `FakeConsumer` replays them and
  can raise `FakeConsumerEmptyException` to end a worker loop (`fake_kafka.py`, used by
  `packages/topology_optimizer_pkg/tests/test_ac_dc_integration.py`).
- Entry points and configuration: tyro CLIs (`optimizer/dc/main.py`, `contingency/ac_loadflow_service/lf_worker.py`);
  worker `main()` functions that expect ready-made producers, consumers and file systems
  (`optimizer/dc/worker/worker.py`, `optimizer/ac/worker.py`), which is why the worker launch commands in
  `docs/usage.md` do not work; Hydra configuration groups and `--multirun` sweeps, used only by the benchmark
  scripts (`docs/benchmark.md`, `toop-engine-benchmark/`); Optuna is declared as a dependency of
  `packages/topology_optimizer_pkg/pyproject.toml` but is not imported anywhere.
- Experiment logging: `SummaryWriter` from tensorboardX writes `fitness` and every observed metric per epoch
  (`optimizer/dc/main.py`); TensorBoard to see whether a run had plateaued (port 6006 is published by the course
  container). At ToOp commit `ba1874e` the log directory name is built from `str(datetime.datetime.now())`, whose
  colons are illegal in Windows paths (fixed on the course fork's `feat/add-docker-compose` branch).
- Parallel CPU work with Ray (`contingency/pandapower/contingency_analysis_pandapower.py`); in containers Ray and
  JAX allocate through `/dev/shm`, whose 64 MB Docker default crashes opaquely (`--shm-size=2gb`); Ray's memory
  monitor kills workers near the memory limit; `CLEANUP_RAY_AFTER_TESTS=true` in tests.
- Containers with Docker; documentation with MkDocs; C4 architecture diagrams (`docs/architecture/`).

## Learning outcomes

- Trace one optimization command from its Pydantic model through the Protobuf envelope, the DC worker, the
  `results` topic and the AC worker's database, naming the code at each hop.
- Map every file of a processed grid folder to the stage that writes it and the stages that read it.
- Query and reason about worker state stored with SQLModel/SQLite, including what is lost on restart.
- Run a Hydra multirun sweep and read its results together with the TensorBoard logs.

## Resources

- Wilson et al., *Research Software Engineering with Python* (free) <https://third-bit.com/py-rse/>;
  Scientific Python Development Guide <https://learn.scientific-python.org/development/>.
- Confluent Developer free Kafka courses <https://developer.confluent.io/courses/>; confluent-kafka Python client
  <https://docs.confluent.io/kafka-clients/python/current/overview.html>; Protobuf Python tutorial
  <https://protobuf.dev/getting-started/pythontutorial/>; Ray Core walkthrough
  <https://docs.ray.io/en/latest/ray-core/walkthrough.html> and Ray Dashboard
  <https://docs.ray.io/en/latest/ray-observability/getting-started.html>; h5py quick start
  <https://docs.h5py.org/en/stable/quick.html>; Pydantic <https://docs.pydantic.dev/>; Pandera
  <https://pandera.readthedocs.io/>; Docker get started <https://docs.docker.com/get-started/>; C4 model
  <https://c4model.com/>.
- SQLModel <https://sqlmodel.tiangolo.com/>; Hydra, "Getting started" <https://hydra.cc/docs/intro/> and "Multi-run"
  <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/>; TensorBoard, "Get started"
  <https://www.tensorflow.org/tensorboard/get_started>; tensorboardX <https://tensorboardx.readthedocs.io/>; tyro
  <https://brentyi.github.io/tyro/>.
- ToOp: `docs/architecture/README.md` and the C4 model in `docs/architecture/model/`; `docs/benchmark.md`;
  `interfaces/folder_structure.py` and `docs/quickstart.md` (processed grid folder);
  `notebooks/example3_e2e_pipeline.ipynb` (staged pipeline, and its "Folder Structure Overview" for reading
  results).

## Lab

!!! example "Lab: follow a command through the pipeline"

    **Tools:** the ToOp container; Jupyter + pandas; h5py; SQLModel; pytest; TensorBoard.

    1. **The on-disk contract.** Open a processed grid folder produced in
       [Chapter 25](../part-4-optimization/ch25-quality-diversity.md)'s lab (or `data/case57_data_powsybl/`
       written by `notebooks/example1_dc_loadflow_example.ipynb`). List every file, identify which stage wrote it
       and which reads it, using the constants in `interfaces/folder_structure.py`, and load
       `static_information.hdf5` with h5py to find the PTDF dataset and check its shape.
    2. **Messages without a broker.** Run
       `uv run pytest packages/topology_optimizer_pkg/tests/test_ac_dc_integration.py::test_ac_dc_integration_sequential`
       and read the test. Then reproduce its first half in a notebook: add `packages/topology_optimizer_pkg` to
       `sys.path` and import `FakeProducer` and `FakeConsumer` from `tests.fake_kafka`; build a
       `StartOptimizationCommand` for a `GridFile` pointing at a preprocessed folder; wrap it with
       `serialize_message(command.model_dump_json())` into a `FakeConsumer(..., kill_on_empty=True)`; run the DC
       worker `main` from `optimizer/dc/worker/worker.py` until it raises `FakeConsumerEmptyException`. Decode every
       message in `producer.messages["results"]` with `deserialize_message` and `Result.model_validate_json`, and
       tabulate the result types with pandas.
    3. **The AC store.** Create `session = create_session()` (`optimizer/ac/storage.py`) and feed it the DC results
       with `poll_results_topic(db=session, consumer=FakeConsumer({"results": ...}))` (`optimizer/ac/listener.py`).
       Query `select(ACOptimTopology)` into a data frame and describe its columns. Then call
       `scrub_db(session, max_age_seconds=0)` and query again. What would an AC worker restart lose?
    4. **Logs.** Run a short DC optimization with `run_dc_optimization_stage` as in
       `notebooks/example3_e2e_pipeline.ipynb`, start
       `uv run --with tensorboard tensorboard --logdir <run folder> --host 0.0.0.0` (ToOp's lock file has tensorboardX
       for writing logs but not the TensorBoard viewer), open port 6006 and decide whether the fitness had plateaued
       when the run stopped.

## Exercises

1. *(Jupyter: pandas)* In the lab's step 3, `poll_results_topic` returned an empty "added" dictionary although rows
   were stored. Read `optimizer/ac/listener.py` and explain why. What would change if the DC run had not finished?
2. *(Jupyter: h5py + NumPy)* Load `static_information.hdf5` twice: with h5py directly and with
   `load_static_information` from `dc_solver/jax/inputs.py`. List which datasets map to which `DynamicInformation`
   fields, and which `SolverConfig` values are stored as HDF5 attributes (`file.attrs`) instead.
3. *(Hydra)* Following `docs/benchmark.md`, run a multirun on `grid=config_grid_node_breaker` over two values of
   `ga_config.runtime_seconds` and two values of a parameter that really exists in `BatchedMEParameters` (the page's
   `ga_config.split_subs=2,5` is not a field; its config folder is `toop-engine-benchmark/configs/` and the script
   is `assess_benchmarks.py`), then aggregate the results with `assess_benchmarks`. Draw the directory tree Hydra
   creates and explain how you would make the sweep reproducible
   ([Chapter 26](../part-4-optimization/ch26-experimental-methodology.md)).
4. *(tyro)* Run `uv run python -m toop_engine_topology_optimizer.dc.main --help` and map five command-line flags to
   fields of `CLIArgs` in `optimizer/dc/main.py` (for example inside `ga_config` and `lf_config`). How does tyro turn
   a nested Pydantic model into flags?
5. *(Docker)* Using the `docker run` form from ToOp's `docker/README.md` (course fork), start the CPU image once with
   `--shm-size=64m` and once with `--shm-size=2gb`, and in each run
   `uv run pytest packages/contingency_analysis_pkg/tests/pandapower/test_contingency_analysis_pandapower.py -q`
   (these tests start Ray). Compare the outcomes and failure signatures, and explain why a small `/dev/shm` is not
   reported as "out of shared memory".
6. Draw a C4 container diagram of a deployment where the DC worker runs on a GPU node and three AC workers run on
   CPU nodes. Mark which state is in Kafka, which on the shared file system, and which only in a worker's SQLite
   memory, and propose what must change so that an AC worker can be restarted without losing work.
