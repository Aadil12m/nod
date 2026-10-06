# syntax=docker/dockerfile:1
# Nod nightly eval image (design 3.1, 4.2.5, runbook nightly.md). One container holds both
# Python environments:
#   - the Isaac-GR00T root venv (/opt/Isaac-GR00T/.venv) for the GR00T policy server, and
#   - the LIBERO venv (sim + nod_sim + nod_eval), which drives episodes through PolicyClient.
# docker/eval_entrypoint.sh starts the server in the background, waits for it, runs
# `python -m nod_eval.run`, then `python -m nod_eval.publish`, and writes everything to
# /mnt/results/<date>/ (a Nebius Object Storage bucket mounted by the Serverless Job).
#
#   docker build -f docker/eval.Dockerfile -t nod/eval:dev .
#   docker tag nod/eval:dev cr.uk-south2.nebius.cloud/<registry_id>/nod-eval:<git-sha>   # see infra/nebius/job_nightly.sh
#
# Same base, pins and LIBERO workarounds as policy.Dockerfile and sim.Dockerfile; see the notes there.

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    HF_HOME=/root/.cache/huggingface \
    NVIDIA_DRIVER_CAPABILITIES=all \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    LIBERO_CONFIG_PATH=/opt/libero_config \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs build-essential cmake ninja-build pkg-config \
      python3.12 python3.12-venv python3.12-dev python-is-python3 \
      ffmpeg libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libavutil-dev \
      libswresample-dev libswscale-dev \
      libegl1 libgl1 libgles2 libglvnd0 libglu1-mesa libosmesa6 jq \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /usr/share/glvnd/egl_vendor.d \
 && printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "libEGL_nvidia.so.0"}\n}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /usr/local/bin/

ARG GR00T_REPO=https://github.com/NVIDIA/Isaac-GR00T
ARG GR00T_REF=51d4c89f72fda44cbf77285c6a8114b52676b8a1
RUN git clone "$GR00T_REPO" /opt/Isaac-GR00T \
 && git -C /opt/Isaac-GR00T checkout --quiet "$GR00T_REF"
WORKDIR /opt/Isaac-GR00T

# 1. Policy server venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.12 --frozen \
 && .venv/bin/python -c "import gr00t; print('gr00t ok')"

# 2. LIBERO venv (no GPU smoke test at build time; see sim.Dockerfile).
ENV LIBERO_VENV=/opt/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv
RUN mkdir -p "$LIBERO_CONFIG_PATH" \
 && R=/opt/Isaac-GR00T/external_dependencies/LIBERO/libero/libero \
 && printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
      "$R" "$R" "$R" "$R" "$R" > "$LIBERO_CONFIG_PATH/config.yaml"
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i "/^python - <<'PY'\$/,\$d" gr00t/eval/sim/LIBERO/setup_libero.sh \
 && bash gr00t/eval/sim/LIBERO/setup_libero.sh \
 && "$LIBERO_VENV/bin/python" -c "import mujoco; assert mujoco.__version__ == '3.3.1', mujoco.__version__"

# 3. Nod code into the LIBERO venv (no-deps for our packages keeps the LIBERO pins).
WORKDIR /app
COPY protocol/pyproject.toml protocol/README.md protocol/
COPY protocol/nod_protocol protocol/nod_protocol
COPY sim sim
COPY eval eval
COPY registry registry
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "$LIBERO_VENV/bin/python" -e "/app/protocol" \
 && uv pip install --python "$LIBERO_VENV/bin/python" --no-deps -e /app/sim \
 && if [ -f /app/eval/pyproject.toml ]; then \
      uv pip install --python "$LIBERO_VENV/bin/python" --no-deps -e /app/eval; \
    fi \
 && "$LIBERO_VENV/bin/python" -c "import nod_protocol, mujoco; assert mujoco.__version__ == '3.3.1'"

ENV PYTHONPATH=/app/eval:/app/sim \
    NOD_REPO=/app

COPY docker/fetch_ckpt.sh /usr/local/bin/fetch_ckpt.sh
COPY docker/eval_entrypoint.sh /usr/local/bin/eval_entrypoint.sh
RUN chmod +x /usr/local/bin/fetch_ckpt.sh /usr/local/bin/eval_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/fetch_ckpt.sh", "/usr/local/bin/eval_entrypoint.sh"]
