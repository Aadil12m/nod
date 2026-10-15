#!/usr/bin/env bash
# Laptop mode: the zero-cost live setup. Everything runs on an ordinary laptop CPU except
# Nemotron, which runs on Nebius Token Factory (paid from the free hackathon credits).
#
#   - sim: the real LIBERO-Goal scene (MuJoCo, CPU) with the scripted actor (design tier T4).
#     Only skills whose scripted controller was measured in this scene are offered: the stats
#     file from `make eval-scripted` (registry/stats/scripted.json) hides the rest (rule R6).
#   - gateway: NOD_MODE=live, Nemotron planner, fake Home Assistant, no Cosmos (the sim oracle
#     decides done/failed). Without a Token Factory key the planner falls back to the
#     deterministic menu, so the demo still works offline.
#   - viewer: the 3D view on http://localhost:8080 when viewer/.venv exists.
#
#   export NEBIUS_API_KEY=...        # Token Factory key (never commit it)
#   infra/scripts/laptop.sh          # then open http://localhost:8000 and http://localhost:8000/presenter
#
# Needs sim/.venv-libero (see sim/README.md) and gateway/.venv (created here on first run).
# Environment: NOD_PORT (8000), VIEWER_PORT (8080), UV, MUJOCO_GL (osmesa; egl on a GPU box).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
PORT="${NOD_PORT:-8000}"
VIEWER_PORT="${VIEWER_PORT:-8080}"
MJB="recordings/scene_libero_goal_v1.mjb"
SIM_PY=sim/.venv-libero/bin/python
STATS="${NOD_STATS_PATH:-registry/stats/scripted.json}"

[[ -x "$SIM_PY" ]] || { echo "laptop: $SIM_PY missing; install the CPU LIBERO venv (sim/README.md)" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "laptop: $STATS missing; run 'make eval-scripted' first" >&2; exit 1; }
if [[ ! -x gateway/.venv/bin/nod-gateway ]]; then
  [[ -x "$UV" ]] || { echo "laptop: uv not found; install it from https://docs.astral.sh/uv/ or set UV=" >&2; exit 1; }
  echo "laptop: creating gateway/.venv (first run only)"
  "$UV" venv -q -p 3.11 gateway/.venv
  "$UV" pip install -q -p gateway/.venv/bin/python -e "protocol[numpy,fast]" -e gateway
fi
[[ -n "${NEBIUS_API_KEY:-${NOD_TF_API_KEY:-}}" ]] || echo "laptop: no NEBIUS_API_KEY set; Nemotron is off and the phone shows the fixed menu"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PWD/sim/.venv-libero/libero_config}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL:-osmesa}}"

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "laptop: sim (real LIBERO-Goal scene, scripted actor)"
"$SIM_PY" -m nod_sim.runner --policy none --actor scripted --lazy-cameras \
  --ctl tcp://127.0.0.1:5560 --estop tcp://127.0.0.1:5561 --pub tcp://127.0.0.1:5562 --pose tcp://127.0.0.1:5563 &
pids+=($!)

if [[ -x viewer/.venv/bin/python && ( -f "$MJB" || -f "$MJB.gz" ) ]]; then
  echo "laptop: viewer on http://localhost:$VIEWER_PORT"
  viewer/.venv/bin/python -m nod_viewer.main --mjb "$MJB" --sub tcp://127.0.0.1:5563 --port "$VIEWER_PORT" &
  pids+=($!)
fi

echo "laptop: gateway on http://localhost:$PORT (phone UI) and http://localhost:$PORT/presenter?viewer=http://localhost:$VIEWER_PORT/"
NOD_MODE=live NOD_DEV=1 NOD_HOME=fake NOD_COSMOS=none NOD_ASR="${NOD_ASR:-fake}" NOD_STATS_PATH="$STATS" \
NOD_SIM_CTL=tcp://127.0.0.1:5560 NOD_SIM_ESTOP=tcp://127.0.0.1:5561 NOD_SIM_PUB=tcp://127.0.0.1:5562 \
NOD_SIM_POSE_PUB=tcp://127.0.0.1:5563 \
  gateway/.venv/bin/nod-gateway --live --host 127.0.0.1 --port "$PORT" &
pids+=($!)
wait -n "${pids[@]}"
