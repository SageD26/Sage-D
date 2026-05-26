#!/usr/bin/env bash
# Run the Sage-D pipeline (delta calc -> SVD -> compress -> tune -> merge -> eval)
# using the best-known setting (uniform 4-bit, lr=0.02, 5 epochs, per-epoch eval).
#
# Usage: ./run.sh <TAG: wm|mc> [GPU]
set -euo pipefail

cd "$(dirname "$0")"

TAG=${1:?TAG required (wm|mc)}
GPU=${2:-0}

PY=''your_python_path''

case "$TAG" in
  wm) CFG=./config/wizardmath13b.yaml ;;
  mc) CFG=./config/wizardcoder13b.yaml ;;
  *) echo "[err] TAG must be wm or mc"; exit 1 ;;
esac

NAME=${TAG}13_lr2e2_5ep_4bit
SAVE_ROOT=./out/$NAME
mkdir -p "$SAVE_ROOT"

export CUDA_VISIBLE_DEVICES=$GPU

echo "==== [$(date '+%F %T')] Sage-D $TAG start ===="
"$PY" main.py \
    --config "$CFG" \
    --model_tag "$TAG" \
    --save_root "$SAVE_ROOT" \
    --save_per_epoch
echo "==== [$(date '+%F %T')] Sage-D $TAG done ===="
