#!/usr/bin/env bash
# LLaVA-V1.5-7B full run — TextVQA only (skip GQA).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=${PY:-python}

MROOT=${MROOT:-./data/models}
LLAVA_LIU=${LLAVA_LIU:-$MROOT/llava-v1.5-7b}
LLAVA_HF=${LLAVA_HF:-$MROOT/llava-1.5-7b-hf}

NAME=run_llava7b_textvqa
SAVE_ROOT=./out/$NAME
mkdir -p "$SAVE_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.4}

echo "==== [$(date '+%F %T')] llava7b full (TextVQA only) start ===="
PYTHONPATH="./src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/main.py \
    --config ./config/llava.yaml \
    --model_tag llava \
    --save_root "$SAVE_ROOT" \
    --llava_template "$LLAVA_LIU" \
    --llava_template_hf "$LLAVA_HF" \
    --save_per_epoch
echo "==== [$(date '+%F %T')] llava7b full done ===="
