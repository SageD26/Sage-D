#!/usr/bin/env bash
# BEiT-3 full run — 2000 train samples, default config, full 5000-image
# Karpathy test eval. Reports BLEU-4 (and other captioning metrics).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=${PY:-python}

NAME=run_beit3
SAVE_ROOT=./out/$NAME
mkdir -p "$SAVE_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU

echo "==== [$(date '+%F %T')] beit3 full start ===="
PYTHONPATH="./src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/main.py \
    --config ./config/beit3.yaml \
    --model_tag beit3 \
    --save_root "$SAVE_ROOT" \
    --save_per_epoch
echo "==== [$(date '+%F %T')] beit3 full done ===="
