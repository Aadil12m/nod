# syntax=docker/dockerfile:1
# Nod sim runner (design 3.1, 4.2.4): LIBERO-Goal on robosuite 1.4.0 and MuJoCo 3.3.1, rendered
# headless with EGL, in the Python 3.12 LIBERO venv that Isaac-GR00T's setup_libero.sh builds.
# It talks to the policy container through Isaac-GR00T's PolicyClient (ZMQ, port 5555) and to
# the gateway on ports 5560-5563.
#
#   docker build -f docker/sim.Dockerfile -t nod/sim:dev .
#
# Notes
# - setup_libero.sh ends with an env.reset() smoke test that needs a GPU EGL device. Docker
#   builds have no GPU, so that last block is cut off here; run the check once on the VM with
#   `docker compose run --rm sim python -c "..."` (docs/runbooks/demo-day.md).
# - LIBERO asks an interactive question on first import unless its config file exists. The
#   config is written up front into $LIBERO_CONFIG_PATH.
# - The root GR00T venv is not needed: the LIBERO venv imports gr00t from the repo checkout via
#   a .pth file and carries the client-side dependencies itself.

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    LIBERO_CONFIG_PATH=/opt/libero_config \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential cmake pkg-config \
      python3.12 python3.12-venv python3.12-dev \
      libegl1 libgl1 libgles2 libglvnd0 libglu1-mesa libosmesa6 ffmpeg jq \
 && rm -rf /var/lib/apt/lists/*

# Tell glvnd to use the NVIDIA EGL driver that the container toolkit mounts at run time
# (same file as Isaac-GR00T's Dockerfile).
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
 && printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "libEGL_nvidia.so.0"}\n}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /usr/local/bin/

ARG GR00T_REPO=https://github.com/NVIDIA/Isaac-GR00T
ARG GR00T_REF=51d4c89f72fda44cbf77285c6a8114b52676b8a1
RUN git clone "$GR00T_REPO" /opt/Isaac-GR00T \
 && git -C /opt/Isaac-GR00T checkout --quiet "$GR00T_REF"

WORKDIR /opt/Isaac-GR00T
ENV LIBERO_VENV=/opt/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv

# LIBERO config (paths inside the submodule), so importing libero never prompts.
RUN mkdir -p "$LIBERO_CONFIG_PATH" \
 && R=/opt/Isaac-GR00T/external_dependencies/LIBERO/libero/libero \
 && printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
      "$R" "$R" "$R" "$R" "$R" > "$LIBERO_CONFIG_PATH/config.yaml"

# Build the LIBERO venv with the upstream script, minus its final GPU smoke test.
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i "/^python - <<'PY'\$/,\$d" gr00t/eval/sim/LIBERO/setup_libero.sh \
 && bash gr00t/eval/sim/LIBERO/setup_libero.sh \
 && "$LIBERO_VENV/bin/python" -c "import mujoco, robosuite; assert mujoco.__version__ == '3.3.1', mujoco.__version__; print('mujoco', mujoco.__version__, 'robosuite', robosuite.__version__)"

# Nod code into the same venv. --no-deps for nod-sim keeps the LIBERO pins (numpy 1.26.4,
# mujoco 3.3.1) untouched; the venv already has pyzmq, msgpack and msgpack-numpy.
WORKDIR /app
COPY protocol/pyproject.toml protocol/README.md protocol/
COPY protocol/nod_protocol protocol/nod_protocol
COPY sim sim
COPY registry registry
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "$LIBERO_VENV/bin/python" -e "/app/protocol" \
 && uv pip install --python "$LIBERO_VENV/bin/python" --no-deps -e /app/sim \
 && "$LIBERO_VENV/bin/python" -c "import nod_protocol, mujoco; assert mujoco.__version__ == '3.3.1'; print('nod_protocol ok')"

ENV PATH=$LIBERO_VENV/bin:$PATH

RUN mkdir -p /data/runs
EXPOSE 5560 5561 5562 5563
CMD ["python", "-m", "nod_sim.runner", "--policy", "tcp://policy:5555"]
