# Sage-D

Sage-D: Accurate, Inference-efficient, and Tunable Delta Compression for Task-specific Fine-tuned Foundation Models

## Pipeline

```
       finetuned − base
              ↓
       SVD + rank alloc
              ↓
beta-absorb + Hadamard + quant
              ↓
   post-quant tuning (A, B)
              ↓
         merge & eval
```
---

### Repository layout
- `main.py`: pipeline entry point
- `model.py`: functions for SVD, compression, post-quant training, and merge
- `eval.py`: per-model, per-task evaluation functions
- `utils.py`: functions for packed-v1 I/O, calibration loaders, training-data prep, and CLI/YAML helpers
- `run.sh`: launcher
- `config/`: example configuration YAMLs

---

### Requirements
- Python 3.10+
- PyTorch (with CUDA)
- Hugging Face transformers, datasets, tokenizers
- vLLM (used by `eval.py` for math/code generation)
- EvalPlus (`mc` HumanEval + MBPP evaluation)
- lm-eval (optional; used for `chat` / TruthfulQA)

---

### Quick start
1) Place base and fine-tuned checkpoints under `model_root` (set in the YAML):
```
<model_root>/base_model/<base_model_name>/
<model_root>/finetuned_model/<finetuned_name>/
```

2) Run the pipeline:
```bash
./run.sh [model_tag] [GPU]
```
`model_tag` is the top-level key in each YAML (e.g. `wizardcoder13b.yaml` → `mc`, `wizardmath13b.yaml` → `wm`).

---

### Configuration (YAML)

- `model_root`: directory holding `base_model/` and `finetuned_model/` subtrees
- `<model_tag>:` 
  - `base_model`, `finetuned`: directory names under `model_root`
  - `calib_dataset`: calibration corpus for sensitivity scoring
  - `rank_min`: minimum per-layer rank, in `group_size` units
  - `num_train_samples`: training set size
  - `quant_train_epoch`: training epochs
  - `train_data_source`: traninig dataset
  - `learning_rate`, `weight_decay`, `warmup_ratio`, `max_grad_norm`
  - `train_batch_size`, `grad_accum`
  - `lm_loss_weight`: LM cross-entropy weight
  - `layer_recon_loss_weight`: reconstruction loss weight
  - `logit_distill_weight`: KD loss weight

CLI overrides (passed to `main.py`):
- `--alpha`: target compression ratio
- `--bits`: uniform quantization bits (default `4`)
- `--group_size`: rank-axis quantization tile (default `128`)
- `--context_length`: training sequence length (default `2048`)
- `--save_per_epoch`: snapshot tuned weights after every epoch

*Configs for other models will be added later.*
