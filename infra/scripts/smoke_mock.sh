#!/usr/bin/env bash
# Smoke test of the mock gateway as a real process (used by `make smoke` and CI): start
# `nod-gateway --mock` on free ports, wait for /healthz, check the PWA, the presenter and the
# registry API, then stop it. Exits non-zero on any failure and prints the gateway log.
#
#   infra/scripts/smoke_mock.sh [path/to/nod-gateway]     # default gateway/.venv/bin/nod-gateway
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${1:-$REPO_ROOT/gateway/.venv/bin/nod-gateway}"
[[ -x "$BIN" ]] || { echo "smoke: $BIN not found (run make setup-gateway)" >&2; exit 1; }
PY="$(dirname "$BIN")/python"

# Free TCP ports for HTTP and the four MockRunner sockets, so a running mock does not collide.
read -r PORT CTL ESTOP PUB POSE < <("$PY" - <<'PY'
import socket
socks = [socket.socket() for _ in range(5)]
for s in socks:
    s.bind(("127.0.0.1", 0))
print(*[s.getsockname()[1] for s in socks])
for s in socks:
    s.close()
PY
)

LOG="$(mktemp)"
RUNS="$(mktemp -d)"
cleanup() { kill "$GW" 2>/dev/null || true; wait "$GW" 2>/dev/null || true; rm -rf "$RUNS"; }
fail() { echo "smoke: FAIL: $*" >&2; echo "--- gateway log ---" >&2; tail -n 60 "$LOG" >&2; exit 1; }

cd "$REPO_ROOT"
NOD_MODE=mock NOD_RUNS_DIR="$RUNS" \
NOD_SIM_CTL="tcp://127.0.0.1:$CTL" NOD_SIM_ESTOP="tcp://127.0.0.1:$ESTOP" \
NOD_SIM_PUB="tcp://127.0.0.1:$PUB" NOD_SIM_POSE_PUB="tcp://127.0.0.1:$POSE" \
  "$BIN" --mock --host 127.0.0.1 --port "$PORT" --log-level warning >"$LOG" 2>&1 &
GW=$!
trap cleanup EXIT

get() { "$PY" -c 'import sys, urllib.request; sys.stdout.write(urllib.request.urlopen(sys.argv[1], timeout=3).read().decode())' "http://127.0.0.1:$PORT$1"; }

for _ in $(seq 1 60); do
  kill -0 "$GW" 2>/dev/null || fail "gateway exited during start-up"
  if health="$(get /healthz 2>/dev/null)"; then break; fi
  sleep 0.5
done
[[ -n "${health:-}" ]] || fail "/healthz did not answer within 30 s"
grep -q '"mock"' <<<"$health" || fail "/healthz does not report mock mode: $health"

get / | grep -qi "<html" || fail "/ did not return the PWA"
get /presenter | grep -qi "<html" || fail "/presenter did not return the presenter page"
reg="$(get /api/registry)" || fail "/api/registry failed"
"$PY" - "$reg" <<'PY' || fail "/api/registry is not a skill list with bowl_on_plate"
import json, sys
data = json.loads(sys.argv[1])
skills = data.get("skills", data) if isinstance(data, dict) else data
ids = {s.get("skill_id") for s in skills}
assert "bowl_on_plate" in ids, ids
PY
get /api/world >/dev/null || fail "/api/world failed"
echo "smoke: OK (port $PORT): healthz, /, /presenter, /api/registry, /api/world"
