# SAGE-D: Accurate, Inference-efficient, and Tunable Delta Compression for Task-specific Fine-tuned Foundation Models

This project is a PyTorch implementation of 'SAGE-D: Accurate, Inference-efficient, and Tunable Delta Compression for Task-specific Fine-tuned Foundation Models'.
We provide executable source code with adjustable arguments and per-tag configuration files used in the paper.
The pipeline covers 7 task-specific fine-tuned foundation models spanning math, code, image captioning, and multi-modal reasoning.

## Prerequisites

- Python 3.10+
- [PyTorch](https://pytorch.org/) (with CUDA)
- [HuggingFace transformers / datasets / tokenizers](https://huggingface.co/docs/transformers/)
- [vLLM](https://github.com/vllm-project/vllm) (math / code / multi-modal eval)
- [EvalPlus](https://github.com/evalplus/evalplus) (MBPP / HumanEval scoring)
- [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) (BEiT-3 captioning BLEU / ROUGE / METEOR / CIDEr)
- [torchscale](https://github.com/microsoft/torchscale) `==0.2.0` and `timm==0.4.12` (BEiT-3 backbone)

The BEiT-3 tag additionally requires a small set of files from Microsoft's
[`unilm/beit3`](https://github.com/microsoft/unilm/tree/master/beit3) research
repository — see [BEiT-3 vendor setup](#beit-3-vendor-setup) below.

## Usage

There are 3 folders at the project root:
- `src`: source code (`main.py`, `model.py`, `eval.py`, `utils.py`).
- `config`: per-tag YAML files with the paper's exact hyperparameters.
- `bash`: ready-to-run launchers — one per Table 2 row.
- `data`: place datasets, models, and the COCO corpus under this directory.

You can run a launcher script to reproduce a single model's table row.
For example, `./bash/run_llava7b.sh 0` reproduces the LLaVA-V1.5-7B TextVQA
row of Table 2 on GPU 0:
```
==== [2025-06-12 12:34:56] llava7b full (TextVQA only) start ====
[stage 1] SVD + alpha-budget rank allocation
[stage 2] 4-bit + Hadamard compression
[stage 3] post-quant tuning
[step2 post-quant] epoch 7/7: 100%|██████████| 1078/1078 [29:18<00:00,  1.63s/it]
[stage 4] merge_llava + convert_llava_to_hf
[stage 5] vLLM TextVQA inference
EVAL RESULTS — model_tag=llava
  textvqa:
    {
      "n": 5000,
      "acc": 58.51
    }
```

Each launcher takes the GPU index as a single optional argument
(defaults to `0`):
```
./bash/run_wm7b.sh    [GPU]   # WizardMath-7B,         GSM8K
./bash/run_wm13b.sh   [GPU]   # WizardMath-13B,        GSM8K
./bash/run_magic7b.sh [GPU]   # Magicoder-S-CL-7B,     MBPP
./bash/run_wc13b.sh   [GPU]   # WizardCoder-13B,       MBPP
./bash/run_beit3.sh   [GPU]   # BEiT-3,                COCO Karpathy
./bash/run_llava7b.sh [GPU]   # LLaVA-V1.5-7B,         TextVQA
./bash/run_llava13b.sh[GPU]   # LLaVA-V1.5-13B,        TextVQA
```

You can control the following arguments (forwarded to `src/main.py`):
- `--config` (path to a YAML in `./config/`): per-tag hyperparameters.
- `--model_tag` (`wm`, `mc`, `llava`, `beit3`, ...): selects the eval suite.
- `--save_root` (any string): output directory for delta / compressed /
  tuned checkpoints and merged model.
- `--alpha` (default `0.03125` = 1/32): target compression ratio.
- `--bits` (default `4`): uniform quantization bit-width.
- `--group_size` (default `128`): rank-axis quantization tile size.
- `--save_per_epoch`: snapshot tuned weights after every epoch.
- `--eval_limit N`: cap eval set to N items (sanity check).
- `--llava_template`, `--llava_template_hf`: required when `model_tag=llava`
  (paths to the liuhaotian LLaVA shards and the llava-hf reference dir).

For example, you can train and evaluate Magicoder-7B with the paper
configuration on GPU 1:
```
./bash/run_magic7b.sh 1
```

You can override paths via environment variables when your data layout
differs from the default `./data/...`:
```
MROOT=/your/model/root DATA_ROOT=/your/dataset/root ./bash/run_wc13b.sh 0
```

## Datasets

Place the following under `./data/` (or set `MROOT` / `DATA_ROOT`):

| Tag | Base model | Fine-tuned model | Eval dataset |
| --- | --- | --- | --- |
| wm  | Llama-2-13b-hf | WizardMath-13B-V1.0 | GSM8K |
| mc  | CodeLlama-13b-Python-hf | WizardCoder-Python-13B-V1.0 | MBPP (EvalPlus) |
| llava (7B)  | vicuna-7b-v1.5  | llava-lm-only | TextVQA |
| llava (13B) | vicuna-13b-v1.5 | llava-lm-only-13b | TextVQA |
| beit3 | beit3-base (224 pretrained) | beit3-base (480 captioning) | COCO Karpathy test |

The base / fine-tuned models are available from:
- Llama-2 / CodeLlama: https://huggingface.co/meta-llama
- WizardMath / WizardCoder: https://huggingface.co/WizardLMTeam
- Vicuna: https://huggingface.co/lmsys
- LLaVA (liuhaotian / llava-hf): https://huggingface.co/liuhaotian, https://huggingface.co/llava-hf
- BEiT-3: https://github.com/microsoft/unilm/tree/master/beit3#download-checkpoints
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- TextVQA: https://huggingface.co/datasets/lmms-lab/textvqa
- COCO captioning (Karpathy split): https://cs.stanford.edu/people/karpathy/deepimagesent/

## BEiT-3 vendor setup

The `beit3` tag depends on five files from Microsoft's
[`unilm/beit3`](https://github.com/microsoft/unilm/tree/master/beit3) repo.
They are not redistributed here. Place them under `src/beit3_vendor/`:
```
git clone --depth 1 https://github.com/microsoft/unilm.git /tmp/unilm
mkdir -p src/beit3_vendor
cp /tmp/unilm/beit3/{modeling_finetune,modeling_utils,datasets,randaug,glossary}.py \
   src/beit3_vendor/
```
The original `utils.py` in that folder is **intentionally skipped** — it
imports `torchmetrics` / `tensorboardX` at top level. Our
`model.py::_import_beit3_vendor()` substitutes an in-memory stub so the
remaining files load without those packages.