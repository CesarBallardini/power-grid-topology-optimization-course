# Labs container of the course: the libraries that the course .venv does not carry (Grid2Op, LightSim2Grid,
# Gymnasium, Stable-Baselines3 with CPU-only torch, QDax, pymoo, moocore), pinned in pyproject.toml and locked in
# uv.lock. Build and run: see README.md next to this file (make test-labs runs the `labs` tests and notebooks in it).
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

ENV UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    # uv uses the image's Python 3.11 and installs into /opt/venv, outside the bind mount at /course
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    # Byte-code and pytest caches of the course code go to /tmp, so the bind-mounted repository stays clean
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest_cache" \
    # No GPU in the image: JAX and torch run on the CPU and must not look for one
    JAX_PLATFORMS=cpu \
    PATH=/opt/venv/bin:$PATH

WORKDIR /opt/labs
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Grid2Op looks for its datasets here; make test-labs mounts the named volume course-labs-data on it
VOLUME /root/data_grid2op
WORKDIR /course
CMD ["python"]
