# syntax=docker/dockerfile:1
# Nod gateway (design 3.1, 4.2.2). CPU only. Used by the mock profile on a laptop and by the
# live profile on the GPU VM (with GATEWAY_EXTRAS=asr for Parakeet through onnx-asr).
#
#   docker build -f docker/gateway.Dockerfile -t nod/gateway:mock .
#   docker build -f docker/gateway.Dockerfile --build-arg GATEWAY_EXTRAS=asr -t nod/gateway:live .
#
# Build context: the repo root. The packages are installed editable from /app so that the
# gateway finds registry/, pwa/ and recordings/ next to it (config.py resolves REPO_ROOT from
# its own path).

FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# "" for mock; "asr" adds onnx-asr (Parakeet TDT 0.6B v3 int8, CPU) for live mode.
ARG GATEWAY_EXTRAS=""

WORKDIR /app

# Dependencies first, for layer caching: copy only the package metadata and code.
COPY protocol/pyproject.toml protocol/README.md protocol/
COPY protocol/nod_protocol protocol/nod_protocol
COPY protocol/schema protocol/schema
COPY gateway/pyproject.toml gateway/README.md gateway/
COPY gateway/nod_gateway gateway/nod_gateway

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python /usr/local/bin/python3.11 \
 && uv pip install --python /opt/venv/bin/python \
      -e "/app/protocol[numpy,fast]" \
      -e "/app/gateway${GATEWAY_EXTRAS:+[$GATEWAY_EXTRAS]}"

# Data the gateway serves or reads. registry/ and recordings/ are also bind-mounted by compose
# so the nightly stats and the mock library can change without a rebuild. pwa/vendor/ is
# included when `make vendor` has been run; without it the PWA falls back to the CDN.
COPY registry registry
COPY pwa pwa
RUN mkdir -p /app/recordings /data/runs

RUN useradd --create-home --uid 10001 nod \
 && chown -R nod:nod /data /app/recordings
USER nod

ENV NOD_RUNS_DIR=/data/runs \
    NOD_HOST=0.0.0.0 \
    NOD_PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

# NOD_MODE (mock | live) comes from the environment; mock when unset.
CMD ["nod-gateway", "--host", "0.0.0.0", "--port", "8000"]
