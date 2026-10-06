#!/usr/bin/env bash
# Download the GR00T N1.7 LIBERO-Goal checkpoint into $NOD_CKPT_DIR (default /ckpt) if it is not
# there yet, then exec the given command. Used as the ENTRYPOINT of the policy and eval images.
#
#   fetch_ckpt.sh uv run --no-sync python gr00t/eval/run_gr00t_server.py --model-path /ckpt/libero_goal ...
#
# The checkpoint is about 6.9 GB (design 1.1). Only the files the server needs are fetched, with
# the same --include list as runbook 8.2. N1.7 also loads the gated nvidia/Cosmos-Reason2-2B on
# first use, so HF_TOKEN must belong to an account that was granted access, or the server fails
# with GatedRepoError.
#
# Environment:
#   NOD_CKPT_DIR      where to put the checkpoint (default /ckpt; the server uses $NOD_CKPT_DIR/libero_goal)
#   NOD_CKPT_REPO     HF repo (default nvidia/GR00T-N1.7-LIBERO)
#   NOD_CKPT_SUBDIR   subfolder (default libero_goal)
#   NOD_SKIP_CKPT=1   do not download (for a checkpoint mounted from elsewhere, e.g. our fine-tune)
#   HF_TOKEN          Hugging Face read token
set -euo pipefail

CKPT_DIR="${NOD_CKPT_DIR:-/ckpt}"
REPO="${NOD_CKPT_REPO:-nvidia/GR00T-N1.7-LIBERO}"
SUB="${NOD_CKPT_SUBDIR:-libero_goal}"

if [[ "${NOD_SKIP_CKPT:-0}" != "1" && ! -f "$CKPT_DIR/$SUB/config.json" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "fetch_ckpt: HF_TOKEN is not set; downloads of gated models will fail" >&2
  fi
  echo "fetch_ckpt: downloading $REPO/$SUB into $CKPT_DIR (about 7 GB)"
  mkdir -p "$CKPT_DIR"
  # The Isaac-GR00T venv ships huggingface_hub and its `hf` CLI.
  HF_BIN="${HF_BIN:-$(command -v hf || echo /opt/Isaac-GR00T/.venv/bin/hf)}"
  "$HF_BIN" download "$REPO" \
    --include "$SUB/config.json" "$SUB/embodiment_id.json" "$SUB/model-*.safetensors" \
              "$SUB/model.safetensors.index.json" "$SUB/processor_config.json" "$SUB/statistics.json" \
    --local-dir "$CKPT_DIR"
fi

if [[ "${NOD_SKIP_CKPT:-0}" != "1" && ! -f "$CKPT_DIR/$SUB/config.json" ]]; then
  echo "fetch_ckpt: $CKPT_DIR/$SUB/config.json missing after download" >&2
  exit 1
fi

exec "$@"
