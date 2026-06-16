#!/usr/bin/env bash
# Magicoder-S-CL-7B full run — MBPP (EvalPlus) eval.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=${PY:-python}

NAME=run_magic7b
SAVE_ROOT=./out/$NAME
mkdir -p "$SAVE_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU

echo "==== [$(date '+%F %T')] magic7b full start ===="
PYTHONPATH="./src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/main.py \
    --config ./config/magicoder7b.yaml \
    --model_tag mc \
    --save_root "$SAVE_ROOT" \
    --save_per_epoch
echo "==== [$(date '+%F %T')] magic7b full done ===="
