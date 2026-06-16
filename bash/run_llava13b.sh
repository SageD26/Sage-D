#!/usr/bin/env bash
# LLaVA-V1.5-13B full run — TextVQA only.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=${PY:-python}

MROOT=${MROOT:-./data/models}
LLAVA_LIU=${LLAVA_LIU:-$MROOT/llava-v1.5-13b}
LLAVA_HF=${LLAVA_HF:-$MROOT/llava-1.5-13b-hf}

NAME=run_llava13b_textvqa
SAVE_ROOT=./out/$NAME
mkdir -p "$SAVE_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.4}

echo "==== [$(date '+%F %T')] llava13b full (TextVQA only) start ===="
PYTHONPATH="./src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/main.py \
    --config ./config/llava13b.yaml \
    --model_tag llava \
    --save_root "$SAVE_ROOT" \
    --llava_template "$LLAVA_LIU" \
    --llava_template_hf "$LLAVA_HF" \
    --save_per_epoch
echo "==== [$(date '+%F %T')] llava13b full done ===="
