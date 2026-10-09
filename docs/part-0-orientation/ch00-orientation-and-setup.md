# Chapter 0. Orientation and setup

**Weeks:** 1 · **Semester:** 1 · **Prerequisites:** none

!!! info "Why it matters for ToOp"

    Students who have seen the pipeline run once know what every later chapter
    is for.

    The environment set up here is the one every later lab uses. ToOp pins Python to `>=3.11,<3.12`, and the GPU
    build of JAX (`jax[cuda12]`) is published for Linux only (JAX installation guide,
    <https://docs.jax.dev/en/latest/installation.html#supported-platforms>), so on Windows the course runs ToOp in
    a Linux container (the course fork's `docker/README.md`, branch `feat/add-docker-compose`).
    [Chapter 30](../part-5-computing/ch30-scientific-python.md) explains what containers, `uv` and git are doing;
    in this chapter they only need to work.

## Topics

- Congestion in transmission grids; redispatch as the costly remedy; topological actions as non-costly
  remedial actions.
- The four action types: line switching, busbar splitting, busbar reassignment, phase-shifting
  transformer (PST) taps.
- The three stages: import → DC optimization on GPU → AC validation; the processed grid folder as the
  contract between stages.
- Getting a working ToOp without a manual install: a course JupyterHub account, an instructor-provided
  container image, or Docker Desktop with the WSL2 backend and `docker compose` on your own machine.
- Which device does the work: CPU vs NVIDIA GPU, how to prove which one JAX uses, and why the first run is slow
  (CUDA context creation and XLA compilation, revisited in
  [Chapter 31](../part-5-computing/ch31-jax-gpu-fundamentals.md)).

## Learning outcomes

- Describe in one paragraph what problem ToOp solves and what a "topology" is in ToOp.
- Start a working ToOp environment (JupyterHub or container) and report, with evidence, which compute device JAX
  uses.
- Run a DC load flow notebook and name each stage it passes through.

## Resources

- ToOp README and documentation: <https://eliagroup.github.io/ToOp/>.
- Talks: LF Energy Summit 2024, "A GPU-Native Approach on Tackling Grid Topology Optimization"
  (<https://lfenergy.org/lf-energy-summit-recap-and-video-a-gpu-native-approach-on-tackling-grid-topology-optimization/>);
  LF Energy Summit 2025 (<https://www.youtube.com/watch?v=XteDpNsX75A>).
- Elia innovation project page:
  <https://innovation.eliagroup.eu/en/projects/toop-topology-optimization-for-congestion-management>.
- Setup documentation:
  - Docker Desktop WSL 2 backend: <https://docs.docker.com/desktop/features/wsl/>; Docker Compose:
    <https://docs.docker.com/compose/>.
  - GPU support in Docker Desktop for Windows: <https://docs.docker.com/desktop/features/gpu/>; NVIDIA, *CUDA on
    WSL User Guide*: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>.
  - JupyterHub: <https://jupyter.org/hub>; for instructors hosting a small course server, The Littlest JupyterHub:
    <https://tljh.jupyter.org/>.
- ToOp container files: `docker-compose.yaml`, `docker/Dockerfile`, `docker/README.md` and
  `docker/device_banner.py` live on the `feat/add-docker-compose` branch of the course fork
  (<https://github.com/CesarBallardini/ToOp/tree/feat/add-docker-compose>); at the time of writing they are not in
  upstream `main`, which only has the dev-container base `Dockerfile` and `.devcontainer/`.
- Upstream installation: ToOp's `README.md` ("Getting Started") and `docs/contribution_guide.md` ("Local Development
  Setup").

## Setup

Pick the first option that is available to you.

1. **Course JupyterHub.** If the instructor runs one, log in, open the ToOp `notebooks/` folder and go to
   "Check the device" below. Nothing is installed on your machine.
2. **Instructor-provided container.** The instructor gives you a prebuilt image (or a registry name) of the same
   `docker/Dockerfile`. Follow option 3 but skip the `docker compose build` step; you still clone the repository,
   because the container mounts it at `/app` and runs your checkout.
3. **Build the container yourself** (Windows 11, macOS or Linux). Install Docker Desktop with the **WSL2** backend
   (on Windows), start it, and check that `docker version` reports a Linux server engine. Keep about 15 GB free
   (more for the GPU image, which the course fork's `docker/README.md` describes as substantially larger). Then:

    ```bash
    git clone --branch feat/add-docker-compose https://github.com/CesarBallardini/ToOp.git
    cd ToOp
    docker compose build toop-cpu      # about 9 minutes the first time
    docker compose up -d               # Jupyter Lab on http://localhost:8888
    docker compose exec toop-cpu bash  # a shell inside; run every Python command as `uv run ...`
    ```

    Use `git clone`, not a ZIP download: the packages compute their version numbers from git tags
    (`uv-dynamic-versioning`), so `uv sync` needs the repository history. Stop with `docker compose down`.

Facts that bite if forgotten (all from the course fork's `docker/README.md`, `docker/Dockerfile` and
`docker-compose.yaml`):

- **Shared memory.** Ray and JAX allocate through `/dev/shm`, and Docker's 64 MB default shows up as opaque
  crashes. The compose file sets `shm_size: "2gb"`; if you ever use `docker run` by hand, pass the equivalent
  `--shm-size=2gb`.
- **Python version.** Do not change the image's Python: ToOp requires `>=3.11,<3.12`, and the base image
  (`python:3.11-slim-bookworm`) matches it on purpose.
- **Memory.** A container that exits with code 137 ran out of memory. WSL2 takes about half the host RAM by
  default; raise it in `%UserProfile%\.wslconfig`. When running the test suite, prefer `-n 4` over `-n auto`.
- **Git Bash.** Prefix any `docker` command that contains a container path such as `/app` with
  `MSYS_NO_PATHCONV=1`, or run it from PowerShell.

**GPU (optional).** Only NVIDIA cards work (JAX has no backend here for AMD or Intel integrated GPUs), and on
Windows only through WSL2 with a CUDA-capable driver and Docker Desktop's GPU support enabled. Use the separate
`toop-gpu` service, enabled by the `gpu` compose profile; the VS Code dev container does *not* enable the GPU (the
fork's `.devcontainer/devcontainer.json` builds the CPU image and passes no GPU through).

```bash
docker compose --profile gpu build toop-gpu   # downloads about 2.8 GB of CUDA wheels (fork docker/README.md)
docker compose --profile gpu up -d toop-gpu   # Jupyter Lab on http://localhost:8889
```

**Check the device.** Three of the four ways the GPU setup can break are silent: the container starts and
everything runs on the CPU. So do not trust the configuration; ask JAX.

- Read the banner printed at container start (`docker compose logs toop-cpu`, or `toop-gpu`). It comes from
  `docker/device_banner.py`, which runs a real computation on the GPU instead of trusting the device list and
  names the cause when it falls back to the CPU (CUDA wheels missing, `JAX_PLATFORMS=cpu`, or the card not passed
  into the container).
- Or run `uv run python -c "import jax; print(jax.devices())"`: `[CudaDevice(id=0)]` means GPU,
  `[CpuDevice(id=0)]` means CPU.

Two expectations to set now. First, **time a GPU run twice**: the first run in a session also pays for CUDA context
creation and XLA compilation, so only the second timing measures the solver ([Chapter 32](../part-5-computing/ch32-jax-in-toop.md)
has you measure both). Second, **a GPU is not faster on the small example grids**: ToOp forces
`jax_enable_x64` (`optimizer/dc/main.py`, `dc_solver/preprocess/convert_to_jax.py`), consumer GeForce cards are
reported to run float64 at about 1/32 of their float32 speed (the GPU banner printed by `docker/device_banner.py`),
and a toy grid has too little work to fill the card. Every lab in this book runs on CPU.

## Lab

!!! example "Lab: first run of the pipeline"

    **Tools:** the ToOp container or the course JupyterHub; Jupyter.

    1. Start your environment (see Setup) and record which compute device JAX reports, copying the banner or
       the `jax.devices()` output into your notes.
    2. Open `notebooks/example1_dc_loadflow_example.ipynb` (IEEE 57-bus) and run all cells as a black box.
       Record the wall-clock time of the first run, then restart the kernel, run all again and record the time
       again. Which part of the difference do you think is compilation?
    3. The notebook writes a processed grid folder to `data/case57_data_powsybl/`. List its files and, using the
       stage diagram in the Topics, guess which stage each file belongs to. (The notebook skips the import stage:
       a generator writes an already-imported folder.)
    4. Write down every term you do not understand yet; revisit the list at the end of Parts III and IV.

## Exercises

1. In one paragraph for a non-engineer, explain why switching a line *out* of service can reduce congestion on
   another line. Keep your answer; you will check it against the DC power flow model in
   [Chapter 16](../part-3-power-systems/ch16-dc-power-flow.md).
2. For each of the four action types, name what physically changes in a substation or on a line, and whether you
   expect it to cost money to execute.
3. *(ToOp container)* Start a container with `JAX_PLATFORMS=cpu` set (for example
   `docker compose exec -e JAX_PLATFORMS=cpu toop-cpu uv run python -c "import jax; print(jax.devices())"`) and
   explain what the output shows. On a GPU machine, what would this variable let you compare?
4. *(ToOp container)* Run `uv run pytest packages/interfaces_pkg/tests -q` inside the container. How many tests
   pass, and how long does the run take? Keep the number: [Chapter 30](../part-5-computing/ch30-scientific-python.md)
   explains what a test suite is for.
5. *(Jupyter)* In `example1_dc_loadflow_example.ipynb`, print `static_information.dynamic_information.n_nodes` and
   `n_branches`. The grid is called "57-bus", yet the numbers differ from 57. Write a hypothesis;
   `case57_data_powsybl` in `dc_solver/example_grids.py` and
   [Chapter 13](../part-3-power-systems/ch13-substations-grid-models.md) give the answer.
