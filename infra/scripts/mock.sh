#!/usr/bin/env bash
# Mock mode without Docker (design 4.4): the gateway with NOD_MODE=mock, plus the viewer when it
# and the scene file are available. CPU only, no API keys. What `make mock` runs.
#
#   infra/scripts/mock.sh        # then open http://localhost:8000 and http://localhost:8000/presenter
#
# The gateway venv (gateway/.venv) is created with uv on first use. The viewer runs only if
# viewer/.venv exists (make setup-viewer) and recordings/scene_libero_goal_v1.mjb exists
# (make fetch-mock); without them the presenter page simply has no 3D view.
#
# Environment: NOD_PORT (8000), VIEWER_PORT (8080), UV (path to uv).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
PORT="${NOD_PORT:-8000}"
VIEWER_PORT="${VIEWER_PORT:-8080}"
MJB="recordings/scene_libero_goal_v1.mjb"

if [[ ! -x gateway/.venv/bin/nod-gateway ]]; then
  [[ -x "$UV" ]] || { echo "mock: uv not found; install it from https://docs.astral.sh/uv/ or set UV=" >&2; exit 1; }
  echo "mock: creating gateway/.venv (first run only)"
  "$UV" venv -q -p 3.11 gateway/.venv
  "$UV" pip install -q -p gateway/.venv/bin/python -e "protocol[numpy,fast]" -e gateway
fi

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# The scene ships gzipped; nod_viewer falls back from <name>.mjb to <name>.mjb.gz by itself.
if [[ -x viewer/.venv/bin/python && ( -f "$MJB" || -f "$MJB.gz" ) ]]; then
  echo "mock: viewer on http://localhost:$VIEWER_PORT"
  viewer/.venv/bin/python -m nod_viewer.main --mjb "$MJB" --sub tcp://127.0.0.1:5563 --port "$VIEWER_PORT" &
  pids+=($!)
else
  echo "mock: no viewer (needs viewer/.venv and $MJB or $MJB.gz); the presenter page will show no 3D view"
fi

echo "mock: gateway on http://localhost:$PORT (phone UI) and http://localhost:$PORT/presenter"
echo "mock: keyboard: Left/Right turn, hold Enter long blink, B brow raise, Space STOP"
NOD_MODE=mock gateway/.venv/bin/nod-gateway --mock --host 127.0.0.1 --port "$PORT" &
pids+=($!)
wait -n "${pids[@]}"
