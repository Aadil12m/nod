#!/usr/bin/env bash
# Nightly eval inside the eval image (design 3.1, 4.2.5; runbook docs/runbooks/nightly.md).
#
# 1. Start the GR00T policy server on 127.0.0.1:5555 in the background (root GR00T venv).
# 2. Wait until PolicyClient.ping() answers (the first start also downloads Cosmos-Reason2-2B).
# 3. Run `python -m nod_eval.run` in the LIBERO venv: every skill alone and in chains A and B.
# 4. Run `python -m nod_eval.publish` to write stats.json next to the episodes.
# 5. Write _SUCCESS and point /mnt/results/latest.txt at this date.
#
# Environment (all optional):
#   NOD_EVAL_DATE        output folder name (default: today, UTC, YYYY-MM-DD)
#   NOD_RESULTS_ROOT     bucket mount (default /mnt/results)
#   NOD_EVAL_EPISODES    episodes per skill and per chain (default 20)
#   NOD_EVAL_CHAINS      chains to run (default A,B)
#   NOD_EVAL_EXTRA_ARGS  extra arguments for nod_eval.run (word-split)
#   NOD_EVAL_SMOKE=1     only start the server, ping it, record nvidia-smi, and exit (image smoke test)
#   NOD_SERVER_WAIT_S    how long to wait for the policy server (default 1500)
#   NOD_JOB_ID, NOD_GIT_SHA  recorded in stats.json by nod_eval.publish
set -euo pipefail

GR00T_DIR="${GR00T_DIR:-/opt/Isaac-GR00T}"
LIBERO_PY="${LIBERO_VENV:-$GR00T_DIR/gr00t/eval/sim/LIBERO/libero_uv/.venv}/bin/python"
ROOT="${NOD_RESULTS_ROOT:-/mnt/results}"
DATE="${NOD_EVAL_DATE:-$(date -u +%F)}"
OUT="$ROOT/$DATE"
EPISODES="${NOD_EVAL_EPISODES:-20}"
CHAINS="${NOD_EVAL_CHAINS:-A,B}"
WAIT_S="${NOD_SERVER_WAIT_S:-1500}"
CKPT="${NOD_CKPT_DIR:-/ckpt}/${NOD_CKPT_SUBDIR:-libero_goal}"
REPO="${NOD_REPO:-/app}"

mkdir -p "$OUT"
log() { echo "[eval $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/eval.log"; }

log "out=$OUT episodes=$EPISODES chains=$CHAINS job=${NOD_JOB_ID:-local} sha=${NOD_GIT_SHA:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/nvidia-smi.csv" 2>&1 || true

(
  cd "$GR00T_DIR"
  exec .venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path "$CKPT" --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper \
    --host 127.0.0.1 --port 5555
) > "$OUT/policy_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

ping_server() {
  "$LIBERO_PY" - <<'PY'
import sys
from gr00t.policy.server_client import PolicyClient
sys.exit(0 if PolicyClient(host="127.0.0.1", port=5555, timeout_ms=2000).ping() else 1)
PY
}

log "waiting for the policy server (up to ${WAIT_S}s)"
start=$(date +%s)
until ping_server 2>/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "policy server exited; last lines of policy_server.log:"
    tail -n 40 "$OUT/policy_server.log" | tee -a "$OUT/eval.log"
    exit 1
  fi
  if (( $(date +%s) - start > WAIT_S )); then
    log "policy server did not answer within ${WAIT_S}s"
    exit 1
  fi
  sleep 10
done
log "policy server up after $(( $(date +%s) - start ))s"

if [[ "${NOD_EVAL_SMOKE:-0}" == "1" ]]; then
  log "smoke mode: server answered; not running episodes"
  touch "$OUT/_SMOKE_OK"
  exit 0
fi

cd "$REPO"
# shellcheck disable=SC2086  # NOD_EVAL_EXTRA_ARGS is meant to be word-split
"$LIBERO_PY" -m nod_eval.run --registry "$REPO/registry/skills.yaml" \
  --episodes "$EPISODES" --chains "$CHAINS" --out "$OUT" ${NOD_EVAL_EXTRA_ARGS:-} 2>&1 | tee -a "$OUT/eval.log"

"$LIBERO_PY" -m nod_eval.publish --in "$OUT" --out "$OUT/stats.json" 2>&1 | tee -a "$OUT/eval.log"

if [[ ! -s "$OUT/stats.json" ]]; then
  log "stats.json was not written"
  exit 1
fi
touch "$OUT/_SUCCESS"
echo "$DATE" > "$ROOT/latest.txt"
log "done: $OUT/stats.json"
