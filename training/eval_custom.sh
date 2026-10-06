#!/usr/bin/env bash
# Evaluate a fine-tuned checkpoint (design 9.1 week 3): the custom skill alone and in chain B,
# plus all LIBERO-Goal skills for regression, with the same harness as the nightly job.
#
#   CKPT=~/nod_ckpt/libero_goal_handover/checkpoint-5000 training/eval_custom.sh
#
# Starts a second GR00T policy server for the new checkpoint on port 5556 (the stock libero_goal
# server is expected on 5555, as in the nightly job) and runs nod_eval.run with the LIBERO venv.
# With REGRESSION=1 the new checkpoint also serves every LIBERO-Goal skill (port 5556 for all).
#
# Environment: CKPT (required), GR00T_DIR (~/Isaac-GR00T), EPISODES (20), OUT (~/nod_eval/<date>-custom),
#              REGRESSION (0), LIBERO_PY (the LIBERO venv python from Isaac-GR00T's setup_libero.sh)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CKPT:?set CKPT to the fine-tuned checkpoint directory}"
GR00T_DIR="${GR00T_DIR:-$HOME/Isaac-GR00T}"
LIBERO_PY="${LIBERO_PY:-$GR00T_DIR/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
EPISODES="${EPISODES:-20}"
OUT="${OUT:-$HOME/nod_eval/$(date -u +%F)-custom}"
CUSTOM_CKPT_ID="nod/GR00T-N1.7-LIBERO-nod/libero_goal_handover"   # registry checkpoint id of bowl_to_user
mkdir -p "$OUT"

(cd "$GR00T_DIR" && exec .venv/bin/python gr00t/eval/run_gr00t_server.py --model-path "$CKPT" \
   --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --host 127.0.0.1 --port 5556) > "$OUT/policy_server_custom.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 150); do
  if "$LIBERO_PY" -c "import sys; from gr00t.policy.server_client import PolicyClient; sys.exit(0 if PolicyClient(host='127.0.0.1', port=5556, timeout_ms=2000).ping() else 1)" 2>/dev/null; then
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "custom policy server exited; see $OUT/policy_server_custom.log" >&2; exit 1; }
  sleep 10
done

BASE_URL="tcp://127.0.0.1:5555"
[[ "${REGRESSION:-0}" == "1" ]] && BASE_URL="tcp://127.0.0.1:5556"

cd "$REPO"
PYTHONPATH="$REPO/eval:$REPO/sim:${PYTHONPATH:-}" "$LIBERO_PY" -m nod_eval.run --registry registry/skills.yaml \
  --episodes "$EPISODES" --chains B --out "$OUT" \
  --policy "$BASE_URL" --policy-for "$CUSTOM_CKPT_ID=tcp://127.0.0.1:5556" 2>&1 | tee "$OUT/eval.log"
PYTHONPATH="$REPO/eval:$REPO/sim:${PYTHONPATH:-}" "$LIBERO_PY" -m nod_eval.publish --in "$OUT" --out "$OUT/stats.json"
echo "stats: $OUT/stats.json"
