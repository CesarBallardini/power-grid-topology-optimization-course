# Labs container

A Docker image with the libraries some labs need and the course `.venv` does not carry: Grid2Op with the
LightSim2Grid backend, Gymnasium and Stable-Baselines3 with CPU-only PyTorch (Chapter 29, Labs 1–4), QDax on JAX
(Chapter 27, Lab 5), pymoo and moocore (Chapter 27, Exercises 3 and 4). Everything runs on the CPU, on Windows,
macOS and Linux, with Docker Desktop or Docker Engine.

| File | Purpose |
|---|---|
| `pyproject.toml` | the pinned dependencies (a uv project that is not a package; CPU torch from the PyTorch index) |
| `uv.lock` | the whole dependency tree, locked by `uv lock` for Linux |
| `Dockerfile` | `python:3.11-slim-bookworm`, uv 0.13.0, `uv sync --frozen` into `/opt/venv` |

## Build

From the repository root:

```bash
docker build -t course-labs:latest docker/labs
```

The build downloads about 0.7 GB of wheels and takes a few minutes. The image is 3.4 GB on disk (0.74 GB
compressed); the virtual environment in `/opt/venv` is 1.9 GB, most of it PyTorch, JAX and their dependencies.
`UV_PROJECT_ENVIRONMENT=/opt/venv` keeps the environment outside `/course`, where the repository is mounted.

To change a version, edit `pyproject.toml`, run `uv lock` in `docker/labs` and rebuild. The lock covers Linux only
(`environments` in `pyproject.toml`), because the CPU builds of PyTorch exist only for the image's platform.

## Grid2Op data, once

Chapter 29 uses the full `l2rpn_case14_sandbox` dataset (1,004 chronics, a 294 MB download, 310 MB unpacked). It
lives in the named volume `course-labs-data`, mounted where Grid2Op looks for its datasets, so it is downloaded
once and survives rebuilds of the image:

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v course-labs-data:/root/data_grid2op course-labs:latest \
  python -c "import grid2op; grid2op.make('l2rpn_case14_sandbox')"
```

`MSYS_NO_PATHCONV=1` is only needed in Git Bash on Windows (it stops Git Bash from rewriting `/root/...` into a
Windows path); it does nothing elsewhere. Without the volume, the chapter code falls back to the two test chronics
that ship with Grid2Op (`test=True`), and the numbers differ from the solutions.

## Tests and notebooks

```bash
make test-labs
```

runs, inside the container, every chapter test named `test_*_labs.py` and every notebook named `*_labs.ipynb`
(both carry the pytest marker `labs`). The repository is bind-mounted at `/course`; byte-code and pytest caches go
to `/tmp` in the container, so nothing is written into the repository. In the course `.venv` (`make test`) the
same tests skip themselves, because the libraries are missing.

## Running a chapter script or Jupyter

A chapter script prints the numbers its solutions quote. From the repository root in Git Bash (on Linux or macOS,
use `"$(pwd)"` as the mount source and drop `MSYS_NO_PATHCONV=1`):

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "$(cygpath -m "$PWD"):/course" -v course-labs-data:/root/data_grid2op \
  -w /course/docs/part-4-optimization/ch29-learning-based-control/src course-labs:latest \
  python grid2op_agents_labs.py
```

The solutions state the CPU budget behind each number; to reproduce it, limit the container to one core with
`--cpus=1`. For Jupyter, install it into a scratch layer of a running container and publish the port:

```bash
MSYS_NO_PATHCONV=1 docker run --rm -it -p 8888:8888 -v "$(cygpath -m "$PWD"):/course" \
  -v course-labs-data:/root/data_grid2op course-labs:latest \
  sh -c "uv pip install --python /opt/venv/bin/python jupyterlab && jupyter lab --ip=0.0.0.0 --allow-root --no-browser"
```

and open the `*_labs.ipynb` notebook of the chapter from the link that Jupyter prints.
