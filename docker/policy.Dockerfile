# syntax=docker/dockerfile:1
# GR00T N1.7 policy server (design 3.1, 4.5): Isaac-GR00T's run_gr00t_server.py serving the
# LIBERO-Goal checkpoint on ZMQ port 5555 inside the compose network (never published).
#
#   docker build -f docker/policy.Dockerfile -t nod/policy:dev .
#
# Base and steps follow Isaac-GR00T's own docker/Dockerfile and README at the commit the design
# was checked against (51d4c89, 2026-08-20): CUDA 12.8 devel on Ubuntu 24.04, Python 3.12,
# `uv sync`. The Nebius host image is CUDA 13.0 (driver 580), which runs CUDA 12.8 containers,
# so the Triton CUDA 13 patch from the README is not needed inside this image [unverified].
# The checkpoint is NOT baked in: docker/fetch_ckpt.sh downloads it into the ckpt volume on
# first start (HF_TOKEN required; the Cosmos-Reason2-2B backbone is gated).

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    HF_HOME=/root/.cache/huggingface \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs build-essential cmake ninja-build pkg-config \
      python3.12 python3.12-venv python3.12-dev python-is-python3 \
      ffmpeg libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libavutil-dev \
      libswresample-dev libswscale-dev libegl1 jq \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install --system

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /usr/local/bin/

# Pin the Isaac-GR00T commit. Record it in docs/runbooks/day1-results.md if you change it.
ARG GR00T_REPO=https://github.com/NVIDIA/Isaac-GR00T
ARG GR00T_REF=51d4c89f72fda44cbf77285c6a8114b52676b8a1
RUN git clone "$GR00T_REPO" /opt/Isaac-GR00T \
 && git -C /opt/Isaac-GR00T checkout --quiet "$GR00T_REF"

WORKDIR /opt/Isaac-GR00T
# Root project venv at /opt/Isaac-GR00T/.venv (the server environment). --frozen keeps the
# upstream uv.lock; the first build takes a while (torch 2.9 cu128, flash-attn wheels).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.12 --frozen \
 && .venv/bin/python -c "import gr00t; print('gr00t ok')"

ENV PATH=/opt/Isaac-GR00T/.venv/bin:$PATH

COPY docker/fetch_ckpt.sh /usr/local/bin/fetch_ckpt.sh
RUN chmod +x /usr/local/bin/fetch_ckpt.sh

EXPOSE 5555
ENTRYPOINT ["/usr/local/bin/fetch_ckpt.sh"]
CMD ["uv", "run", "--no-sync", "python", "gr00t/eval/run_gr00t_server.py", \
     "--model-path=/ckpt/libero_goal", "--embodiment-tag=LIBERO_PANDA", "--use-sim-policy-wrapper", \
     "--host=0.0.0.0", "--port=5555"]
