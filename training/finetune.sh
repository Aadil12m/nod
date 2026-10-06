#!/usr/bin/env bash
# Fine-tune the custom skill (design 4.2.5, 9.1 week 3): GR00T N1.7 from the libero_goal
# checkpoint on LIBERO-Goal plus our scripted bowl_to_user demos. Wraps Isaac-GR00T's
# examples/finetune.sh; run on the GPU VM with cosmos stopped (design 4.5).
#
#   training/finetune.sh                                  # defaults below
#   MAX_STEPS=10 training/finetune.sh                     # dry run (design 10: "Week 3 dry run (10 steps)")
#
# Inputs (environment, all optional):
#   GR00T_DIR       Isaac-GR00T checkout with its root venv (default ~/Isaac-GR00T)
#   BASE_MODEL      start checkpoint (default /ckpt/libero_goal, the one the policy server serves;
#                   fallback per design: nvidia/GR00T-N1.7-3B with MAX_STEPS=20000)
#   NOD_DATASET     our LeRobot dataset from to_lerobot.py (default training/data/bowl_to_user_lerobot)
#   LIBERO_DATASET  the original libero_goal_no_noops LeRobot dataset ("old plus new data", so the
#                   ten LIBERO-Goal skills do not regress). Downloaded if missing. Set to "none" to
#                   train on our demos only.
#   OUT_DIR         output checkpoint dir (default ~/nod_ckpt/libero_goal_handover)
#   MAX_STEPS (5000), GLOBAL_BATCH_SIZE (64), SAVE_STEPS (1000), NUM_GPUS (1), USE_WANDB (0)
#
# [unverified] Starting from a fine-tuned checkpoint rather than the 3B base (design 10), the
# batch size that fits one RTX PRO 6000, and the step count. Isaac-GR00T's launch_finetune.py
# accepts several datasets joined with ':' (os.pathsep); their mixture weights use its defaults.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_DIR="${GR00T_DIR:-$HOME/Isaac-GR00T}"
BASE_MODEL="${BASE_MODEL:-/ckpt/libero_goal}"
NOD_DATASET="$(realpath -m "${NOD_DATASET:-$REPO/training/data/bowl_to_user_lerobot}")"
LIBERO_DATASET="${LIBERO_DATASET:-$GR00T_DIR/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot}"
OUT_DIR="${OUT_DIR:-$HOME/nod_ckpt/libero_goal_handover}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export NUM_GPUS="${NUM_GPUS:-1}"
export USE_WANDB="${USE_WANDB:-0}"

[[ -f "$NOD_DATASET/meta/info.json" ]] || { echo "no LeRobot dataset at $NOD_DATASET (run gen_demos.py, then to_lerobot.py)" >&2; exit 1; }
[[ -f "$NOD_DATASET/meta/modality.json" ]] || { echo "$NOD_DATASET/meta/modality.json missing" >&2; exit 1; }

DATASETS="$NOD_DATASET"
if [[ "$LIBERO_DATASET" != "none" ]]; then
  if [[ ! -f "$LIBERO_DATASET/meta/info.json" ]]; then
    echo "downloading the LIBERO-Goal LeRobot dataset to $LIBERO_DATASET"
    (cd "$GR00T_DIR" && uv run hf download --repo-type dataset IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
       --local-dir "$LIBERO_DATASET")
    # Same preparation as Isaac-GR00T examples/LIBERO/README.md (Fine-tune LIBERO goal).
    cp "$GR00T_DIR/examples/LIBERO/modality.json" "$LIBERO_DATASET/meta/"
    cp "$GR00T_DIR/examples/LIBERO/patches/episode_000082.mp4" \
       "$LIBERO_DATASET/videos/chunk-000/observation.images.wrist_image/"
  fi
  DATASETS="$LIBERO_DATASET:$NOD_DATASET"
fi

mkdir -p "$OUT_DIR"
echo "base=$BASE_MODEL datasets=$DATASETS out=$OUT_DIR steps=$MAX_STEPS batch=$GLOBAL_BATCH_SIZE"
cd "$GR00T_DIR"
uv run bash examples/finetune.sh \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DATASETS" \
  --embodiment-tag LIBERO_PANDA \
  --output-dir "$OUT_DIR" \
  --state-dropout-prob 0.2 \
  "$@" 2>&1 | tee "$OUT_DIR/finetune.log"
