# syntax=docker/dockerfile:1
# Nod viewer (design 3.1, 4.2.5): MuJoCo 3.14 plus mjviser/viser, CPU only. It subscribes to
# the sim's pose stream (ZMQ PUB, port 5563) and serves the 3D view on port 8080. It is a separate
# process because mjviser needs MuJoCo 3.6 or newer while the sim is pinned to 3.3.1.
#
#   docker build -f docker/viewer.Dockerfile -t nod/viewer:dev .
#
# The scene file (recordings/scene_libero_goal_v1.mjb) is mounted at run time, not baked in:
# it comes from `make fetch-mock`, or from `viewer/nod_viewer/export_scene.py` run in the sim image.
# Versions are the ones the streaming report tested (mujoco 3.14.0, mjviser 0.0.14, viser 1.1.1).

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# mujoco's Python wheel loads libGL at import on some paths; keep the small GL/EGL libs around.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libegl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ARG MUJOCO_VERSION=3.14.0
ARG MJVISER_VERSION=0.0.14
ARG VISER_VERSION=1.1.1

WORKDIR /app
COPY protocol/pyproject.toml protocol/README.md protocol/
COPY protocol/nod_protocol protocol/nod_protocol
COPY viewer viewer

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python /usr/local/bin/python3.12 \
 && uv pip install --python /opt/venv/bin/python \
      "mujoco==${MUJOCO_VERSION}" "mjviser==${MJVISER_VERSION}" "viser==${VISER_VERSION}" \
      "pyzmq>=26" "numpy>=1.26" -e "/app/protocol[numpy,fast]" \
 && if [ -f /app/viewer/pyproject.toml ]; then \
      uv pip install --python /opt/venv/bin/python -e /app/viewer; \
    fi \
 && python -c "import mujoco, viser, mjviser; print('mujoco', mujoco.__version__)"

ENV PYTHONPATH=/app/viewer
RUN useradd --create-home --uid 10001 nod
USER nod

EXPOSE 8080
CMD ["python", "-m", "nod_viewer.main", "--mjb", "/recordings/scene_libero_goal_v1.mjb", \
     "--sub", "tcp://sim:5563", "--port", "8080"]
