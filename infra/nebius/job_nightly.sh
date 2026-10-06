#!/usr/bin/env bash
# Launch the nightly reliability eval as a Nebius Serverless AI Job (design 3.1, 3.2, 4.2.5).
# Called by .github/workflows/nightly.yml; can also be run by hand. Runbook: docs/runbooks/nightly.md.
#
#   infra/nebius/job_nightly.sh                    # create the job and return
#   infra/nebius/job_nightly.sh --wait             # also poll until it ends, print logs, exit 0 only on success
#   infra/nebius/job_nightly.sh --wait --fetch-stats registry/stats/latest.json
#   infra/nebius/job_nightly.sh --smoke --wait     # image smoke test: start GR00T, ping it, exit
#   infra/nebius/job_nightly.sh --dry-run          # print the commands only
#
# Required environment (TODO markers name where to find each one):
#   NEBIUS_PROJECT_ID   uk-south2 project id
#   NOD_EVAL_IMAGE      cr.uk-south2.nebius.cloud/<registry_id>/nod-eval:<tag> (built from docker/eval.Dockerfile)
#   NOD_BUCKET_ID       Object Storage bucket *resource id* for results, mounted at /mnt/results
#   NOD_HF_SECRET       MysteryBox secret selector holding HF_TOKEN (passed with --env-secret)
# Optional:
#   NOD_CKPT_BUCKET_ID  a second bucket mounted at /ckpt, so the 7 GB checkpoint is downloaded once
#   NOD_REGISTRY_SECRET MysteryBox selector for registry credentials (--registry-secret), if needed
#   NOD_EVAL_EPISODES (20)  NOD_EVAL_CHAINS (A,B)  NOD_JOB_TIMEOUT (2h)  NOD_JOB_DISK (250Gi)
#   PREEMPTIBLE (1)     nightly runs are preemptible; on-failure restart covers a preemption
#   NOD_GIT_SHA         recorded in stats.json (GitHub sets GITHUB_SHA)
#   NOD_EVAL_DATE       results folder (default: today, UTC)
#   --fetch-stats needs NOD_BUCKET_NAME plus AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY for the
#   Nebius S3 endpoint https://storage.$NEBIUS_REGION.nebius.cloud, and the aws CLI.
#
# Job flags follow docs/research/nebius-ops.md 3.3 [V flags]. The job status field
# (.status.state) and the terminal state names are taken from the Nebius cookbooks and are
# [unverified] for jobs; the poll loop treats any unknown state as still running until the timeout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/nebius/lib.sh
source "$HERE/lib.sh"

WAIT=0
SMOKE=0
FETCH_DEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --wait) WAIT=1 ;;
    --smoke) SMOKE=1 ;;
    --fetch-stats) FETCH_DEST="${2:?--fetch-stats needs a destination path}"; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

need NEBIUS_PROJECT_ID "uk-south2 project id (Serverless AI is not offered in eu-south1)"
need NOD_EVAL_IMAGE "push docker/eval.Dockerfile to cr.$NEBIUS_REGION.nebius.cloud/<registry_id>/nod-eval:<tag> (see docs/runbooks/nightly.md)"
need NOD_BUCKET_ID "Object Storage bucket resource id (Console > Storage > bucket > id), not the bucket name"
need NOD_HF_SECRET "MysteryBox secret selector for HF_TOKEN (nebius mysterybox secret create, see nightly.md)"

DATE="${NOD_EVAL_DATE:-$(date -u +%F)}"
EPISODES="${NOD_EVAL_EPISODES:-20}"
CHAINS="${NOD_EVAL_CHAINS:-A,B}"
TIMEOUT="${NOD_JOB_TIMEOUT:-2h}"
DISK="${NOD_JOB_DISK:-250Gi}"
GIT_SHA="${NOD_GIT_SHA:-${GITHUB_SHA:-unknown}}"
SUFFIX="$(date -u +%H%M%S)"
if [[ "$SMOKE" == "1" ]]; then
  JOB_NAME="nod-eval-smoke-${DATE//-/}-$SUFFIX"
else
  JOB_NAME="nod-eval-${DATE//-/}-$SUFFIX"
fi

# Design section 8: a possible per-tenant maintenance on 2026-10-07 06:00 UTC. Do not start a
# 2 h job that would run across it.
if [[ "$(date -u +%F)" == "2026-10-07" && "$(date -u +%H)" =~ ^0[3-6]$ && "${NOD_ALLOW_MAINTENANCE_WINDOW:-0}" != "1" ]]; then
  die "inside the 2026-10-07 03:00-07:00 UTC window around the announced maintenance; set NOD_ALLOW_MAINTENANCE_WINDOW=1 to override"
fi

args=(
  nebius ai job create
  --parent-id "$NEBIUS_PROJECT_ID"
  --name "$JOB_NAME"
  --image "$NOD_EVAL_IMAGE"
  --platform "$NOD_PLATFORM"
  --preset "$NOD_PRESET"
  --timeout "$TIMEOUT"
  --restart-policy on-failure
  --disk-size "$DISK"
  --volume "$NOD_BUCKET_ID:/mnt/results:rw"
  --env PYTHONUNBUFFERED=1
  --env "NOD_EVAL_DATE=$DATE"
  --env "NOD_EVAL_EPISODES=$EPISODES"
  --env "NOD_EVAL_CHAINS=$CHAINS"
  --env "NOD_GIT_SHA=$GIT_SHA"
  --env "NOD_JOB_ID=$JOB_NAME"
  --env "NOD_EVAL_SMOKE=$SMOKE"
  --env-secret "HF_TOKEN=$NOD_HF_SECRET"
  --format "jsonpath={.metadata.id}"
)
[[ "${PREEMPTIBLE:-1}" == "1" ]] && args+=(--preemptible)
[[ -n "${NOD_CKPT_BUCKET_ID:-}" ]] && args+=(--volume "$NOD_CKPT_BUCKET_ID:/ckpt:rw")
[[ -n "${NOD_REGISTRY_SECRET:-}" ]] && args+=(--registry-secret "$NOD_REGISTRY_SECRET")

if [[ "$DRY_RUN" == "1" ]]; then
  run "${args[@]}"
  JOB_ID="TODO-job-id"
  [[ "$WAIT" == "1" ]] && run nebius ai job get "$JOB_ID" --format json
  [[ "$WAIT" == "1" ]] && run nebius ai job logs "$JOB_ID"
  if [[ -n "$FETCH_DEST" ]]; then
    run aws s3 cp "s3://${NOD_BUCKET_NAME:-TODO-bucket-name}/$DATE/stats.json" "$FETCH_DEST" \
      --endpoint-url "https://storage.$NEBIUS_REGION.nebius.cloud"
  fi
  exit 0
fi

have_cli
JOB_ID="$("${args[@]}")"
[[ -n "$JOB_ID" ]] || die "job create returned no id"
echo "JOB_ID=$JOB_ID"
echo "JOB_NAME=$JOB_NAME"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  { echo "job_id=$JOB_ID"; echo "job_name=$JOB_NAME"; echo "date=$DATE"; } >> "$GITHUB_OUTPUT"
fi

[[ "$WAIT" == "1" ]] || exit 0

# Poll. Budget: the job timeout plus provisioning (about 10-15 min) plus a margin.
POLL_S="${NOD_POLL_S:-60}"
MAX_WAIT_S="${NOD_MAX_WAIT_S:-10800}"
start=$(date +%s)
state="UNKNOWN"
while :; do
  state="$(nebius ai job get "$JOB_ID" --format json | jq -r '.status.state // "UNKNOWN"')" || state="UNKNOWN"
  echo "$(date -u +%H:%M:%S) $JOB_NAME: $state"
  case "$state" in
    COMPLETED|SUCCEEDED) break ;;
    FAILED|ERROR|CANCELLED|CANCELED) break ;;
  esac
  if (( $(date +%s) - start > MAX_WAIT_S )); then
    echo "gave up waiting after ${MAX_WAIT_S}s; the job keeps running: nebius ai job get $JOB_ID" >&2
    exit 2
  fi
  sleep "$POLL_S"
done

echo "--- job logs (tail) ---"
nebius ai job logs "$JOB_ID" 2>&1 | tail -n 200 || true

case "$state" in
  COMPLETED|SUCCEEDED) ;;
  *) die "job $JOB_NAME ended in state $state" ;;
esac

if [[ -n "$FETCH_DEST" && "$SMOKE" != "1" ]]; then
  need NOD_BUCKET_NAME "bucket name (not id) for the S3 API"
  command -v aws >/dev/null 2>&1 || die "aws CLI not found (needed for --fetch-stats)"
  tmp="$(mktemp)"
  aws s3 cp "s3://$NOD_BUCKET_NAME/$DATE/stats.json" "$tmp" \
    --endpoint-url "https://storage.$NEBIUS_REGION.nebius.cloud"
  jq -e 'type == "object"' "$tmp" >/dev/null || die "stats.json is not a JSON object"
  mkdir -p "$(dirname "$FETCH_DEST")"
  mv "$tmp" "$FETCH_DEST"
  echo "stats written to $FETCH_DEST"
fi
