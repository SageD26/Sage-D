#!/usr/bin/env bash
# WizardMath-13B full run — GSM8K eval.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=${PY:-python}

NAME=run_wm13b
SAVE_ROOT=./out/$NAME
mkdir -p "$SAVE_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU

echo "==== [$(date '+%F %T')] wm13b full start ===="
PYTHONPATH="./src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/main.py \
    --config ./config/wizardmath13b.yaml \
    --model_tag wm \
    --save_root "$SAVE_ROOT" \
    --save_per_epoch
echo "==== [$(date '+%F %T')] wm13b full done ===="
