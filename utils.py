"""Packed-v1 I/O, calibration loaders, training-data prep, CLI/YAML helpers."""

from __future__ import annotations

import argparse
import os
import yaml
import torch
from tqdm import tqdm


PACKED_FIELDS = (
    "U_q", "U_scale", "U_zero",
    "V_T_q", "V_T_scale", "V_T_zero",
    "A", "B",
)


def load_packed(path: str) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    """
    Load a packed-v1 quantized delta file into a per-layer dict plus meta.

    Parameters:
        path: Path to a packed-v1 .pt file (output of save_packed).

    Returns:
        per_layer: Dict mapping layer prefix to the 8-field packed dict.
        meta: Side metadata dict (bits, group_size, betas, ...).
    """
    flat = torch.load(path, map_location="cpu")
    meta = flat.pop("__meta__", None)
    if not isinstance(meta, dict) or meta.get("format") != "packed_v1":
        raise ValueError(
            f"{path} is not packed_v1 format (got meta={meta!r})."
        )

    per_layer: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in flat.items():
        prefix, field = k.rsplit(".", 1)
        per_layer.setdefault(prefix, {})[field] = v

    for prefix, packed in per_layer.items():
        missing = [f for f in PACKED_FIELDS if f not in packed]
        if missing:
            raise ValueError(f"{prefix} missing fields {missing} in {path}")

    return per_layer, meta


def save_packed(path: str,
                per_layer: dict[str, dict[str, torch.Tensor]],
                meta: dict) -> None:
    """
    Persist a per-layer packed dict + meta to disk in packed-v1 format.

    Parameters:
        path: Destination .pt file.
        per_layer: Dict mapping layer prefix to the 8-field packed dict.
        meta: Side metadata dict; copied under the "__meta__" key.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    flat: dict = {"__meta__": dict(meta)}
    for prefix, packed in per_layer.items():
        for field in PACKED_FIELDS:
            if field not in packed:
                raise ValueError(f"{prefix} packed dict missing field {field!r}")
            flat[f"{prefix}.{field}"] = packed[field]
    torch.save(flat, path)


_MAGICODER_PROMPT = (
    "You are an exceptionally intelligent coding assistant that consistently "
    "delivers accurate and reliable responses to user instructions.\n\n"
    "@@ Instruction\n{q}\n\n@@ Response\n{r}"
)

_METAMATH_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{q}\n\n### Response: Let's think step by step.\n{r}"
)


def _pack_to_seqlen(tokenizer, texts, n_samples: int, seqlen: int):
    """
    Concatenate texts, tokenize, and reshape into a (n_samples, seqlen) tensor.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        texts: Iterable of strings to concatenate.
        n_samples: Number of calibration sequences to return.
        seqlen: Length of each calibration sequence.

    Returns:
        Int tensor of shape (n_samples, seqlen) with the concatenated token ids.
    """
    big_text = "\n\n".join(texts)
    enc = tokenizer(big_text, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]
    need = n_samples * seqlen
    if ids.shape[0] < need:
        raise RuntimeError(f"only {ids.shape[0]} tokens; need {need}.")
    return ids[:need].reshape(n_samples, seqlen)


def get_c4_calibration(tokenizer, n_samples=128, seqlen=2048, seed=0):
    """
    Build calibration sequences from streaming allenai/c4 (en).

    Parameters:
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of calibration sequences to return.
        seqlen: Length of each calibration sequence.
        seed: Unused; kept for dispatcher compatibility.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts, tok_budget = [], 0
    target = int(n_samples * seqlen * 1.3)
    for ex in ds:
        texts.append(ex["text"])
        tok_budget += len(ex["text"]) // 4
        if tok_budget >= target:
            break
    return _pack_to_seqlen(tokenizer, texts, n_samples, seqlen)


def get_metamath_calibration(tokenizer, n_samples=128, seqlen=2048, seed=0):
    """
    Build calibration sequences from meta-math/MetaMathQA in MetaMath prompt format.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of calibration sequences to return.
        seqlen: Length of each calibration sequence.
        seed: Shuffle seed for the dataset.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    from datasets import load_dataset
    ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=seed)
    texts, tok_budget = [], 0
    target = int(n_samples * seqlen * 1.3)
    for ex in ds:
        texts.append(_METAMATH_PROMPT.format(q=ex["query"], r=ex["response"]))
        tok_budget += (len(ex["query"]) + len(ex["response"])) // 4
        if tok_budget >= target:
            break
    return _pack_to_seqlen(tokenizer, texts, n_samples, seqlen)


def get_mbpp_calibration(tokenizer, n_samples=128, seqlen=2048, seed=0):
    """
    Build calibration sequences from google-research-datasets/mbpp (sanitized) using the Magicoder prompt.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of calibration sequences to return.
        seqlen: Length of each calibration sequence.
        seed: Shuffle seed for the dataset.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    from datasets import load_dataset


    ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                      split="train").shuffle(seed=seed)
    sample = ds[0]
    q_field = "prompt" if "prompt" in sample else "text"
    r_field = "code"

    examples = list(ds)
    if not examples:
        raise RuntimeError("MBPP train split is empty.")
    texts, tok_budget = [], 0
    target = int(n_samples * seqlen * 1.3)
    i = 0
    while tok_budget < target:
        ex = examples[i % len(examples)]
        q = ex[q_field]
        r = ex[r_field]
        texts.append(_MAGICODER_PROMPT.format(q=q, r=r))
        tok_budget += (len(q) + len(r)) // 4
        i += 1
    return _pack_to_seqlen(tokenizer, texts, n_samples, seqlen)


def get_alpaca_calibration(tokenizer, n_samples=128, seqlen=2048, seed=0,
                           use_chat_template=True):
    """
    Build calibration sequences from yahma/alpaca-cleaned, optionally via the tokenizer's chat template.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of calibration sequences to return.
        seqlen: Length of each calibration sequence.
        seed: Shuffle seed for the dataset.
        use_chat_template: If True and the tokenizer has a chat template, apply it.

    Returns:
        Int tensor of shape (n_samples, seqlen).
    """
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=seed)
    has_chat = (use_chat_template
                and getattr(tokenizer, "chat_template", None) is not None)

    texts, tok_budget = [], 0
    target = int(n_samples * seqlen * 1.3)
    for ex in ds:
        instr = ex["instruction"]
        inp = ex.get("input") or ""
        out = ex["output"]
        user = (instr + "\n" + inp).strip() if inp else instr

        if has_chat:
            try:
                t = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user},
                     {"role": "assistant", "content": out}],
                    tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                has_chat = False
                continue
        else:
            t = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{user}\n\n### Response: {out}"
            )

        texts.append(t)
        tok_budget += (len(user) + len(out)) // 4
        if tok_budget >= target:
            break

    return _pack_to_seqlen(tokenizer, texts, n_samples, seqlen)


def get_calibration(name: str, tokenizer, n_samples=128, seqlen=2048, seed=0):
    """
    Dispatch to the calibration loader matching the given dataset name.

    Parameters:
        name: One of "c4" | "metamath" | "mbpp" | "alpaca".
        tokenizer: Hugging Face tokenizer.
        n_samples: Number of calibration sequences to return.
        seqlen: Length of each calibration sequence.
        seed: Shuffle seed forwarded to the loader.

    Returns:
        Int tensor of shape (n_samples, seqlen) from the selected loader.
    """
    table = {
        "c4": get_c4_calibration,
        "metamath": get_metamath_calibration,
        "mbpp": get_mbpp_calibration,
        "alpaca": get_alpaca_calibration,
    }
    if name not in table:
        raise ValueError(f"unknown calib_dataset={name!r}; choose from {list(table)}")
    return table[name](tokenizer, n_samples=n_samples, seqlen=seqlen, seed=seed)


import random as _random


_GSM8K_MATH_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response: Let's think step by step.\n{response}"
)

_MAGICODER_TRAIN_PROMPT = (
    "You are an exceptionally intelligent coding assistant that consistently "
    "delivers accurate and reliable responses to user instructions.\n\n"
    "@@ Instruction\n{instruction}\n\n@@ Response\n{response}"
)


def _pack_ids(ids_list, context_length: int) -> torch.Tensor:
    """
    Concatenate per-example token id tensors and reshape into fixed-length chunks.

    Parameters:
        ids_list: Iterable of int tensors of shape (1, T_i).
        context_length: Length of each output chunk; the trailing partial chunk is dropped.

    Returns:
        Int tensor of shape (n, context_length).
    """
    ids = torch.cat(ids_list, dim=1)
    trunc = ids.size(1) - (ids.size(1) % context_length)
    return ids[:, :trunc].view(-1, context_length)


def _pack_ids_pair(ids_list, labels_list,
                   context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack paired (input_ids, labels) tensors into fixed-length chunks, preserving token alignment.

    Parameters:
        ids_list: Iterable of int tensors of shape (1, T_i).
        labels_list: Iterable of int tensors of shape (1, T_i) aligned with ids_list.
        context_length: Length of each output chunk; the trailing partial chunk is dropped.

    Returns:
        ids: Int tensor of shape (n, context_length).
        labels: Int tensor of shape (n, context_length).
    """
    ids = torch.cat(ids_list, dim=1)
    labels = torch.cat(labels_list, dim=1)
    trunc = ids.size(1) - (ids.size(1) % context_length)
    return (ids[:, :trunc].view(-1, context_length),
            labels[:, :trunc].view(-1, context_length))


def prep_magicoder(tokenizer, num_samples: int,
                   context_length: int = 2048, seed: int = 0,
                   ) -> torch.Tensor:
    """
    Build Magicoder training data from OSS-Instruct-75K + Evol-Instruct-110K using the Magicoder prompt.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        num_samples: Half the size of the sampling pool (target ~ 2 * num_samples examples).
        context_length: Output chunk length.
        seed: RNG seed for sampling.

    Returns:
        ids: Int tensor of shape (n, context_length).
        labels: Clone of ids (no prompt masking).
    """
    from datasets import load_dataset
    rng = _random.Random(seed)

    oss = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    evol = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")

    def _norm(ex):
        """
        Extract (instruction, response) pair from a Magicoder example with mixed key names.

        Parameters:
            ex: Source example dict.

        Returns:
            Tuple (instruction, response) of strings (possibly empty).
        """
        instr = ex.get("problem") or ex.get("instruction") or ""
        resp = ex.get("solution") or ex.get("response") or ""
        return instr, resp

    pool = []
    for ex in oss:
        i, r = _norm(ex)
        if i and r:
            pool.append((i, r))
    for ex in evol:
        i, r = _norm(ex)
        if i and r:
            pool.append((i, r))

    target = min(2 * num_samples, len(pool))
    picks = rng.sample(pool, target)
    rng.shuffle(picks)

    texts = [_MAGICODER_TRAIN_PROMPT.format(instruction=i, response=r)
             for i, r in picks]
    eos = tokenizer.eos_token or ""
    ids_list = [tokenizer(t + eos, return_tensors="pt", truncation=False)["input_ids"]
                for t in texts]
    print(f"[mc-train] sampled {target}/{len(pool)} (2*num_samples={2*num_samples}) "
          f"from OSS-Instruct-75K + Evol-Instruct-110K -> {len(texts)} examples")
    ids = _pack_ids(ids_list, context_length)
    return ids, ids.clone()


def prep_alpaca_train(tokenizer, num_samples: int,
                      context_length: int = 2048, seed: int = 0,
                      use_chat_template: bool = True) -> torch.Tensor:
    """
    Build training data from yahma/alpaca-cleaned, optionally via the tokenizer's chat template.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        num_samples: Half the size of the sampling pool (target ~ 2 * num_samples examples).
        context_length: Output chunk length.
        seed: RNG seed for sampling and shuffling.
        use_chat_template: If True and the tokenizer has a chat template, apply it; falls back to Alpaca text otherwise.

    Returns:
        ids: Int tensor of shape (n, context_length).
        labels: Clone of ids (no prompt masking).
    """
    from datasets import load_dataset
    rng = _random.Random(seed)

    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=seed)
    has_chat = (use_chat_template
                and getattr(tokenizer, "chat_template", None) is not None)

    target = min(2 * num_samples, len(ds))
    indices = rng.sample(range(len(ds)), target)

    texts = []
    for i in indices:
        ex = ds[i]
        instr = ex["instruction"]
        inp = ex.get("input") or ""
        out = ex["output"]
        user = (instr + "\n" + inp).strip() if inp else instr

        if has_chat:
            try:
                t = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user},
                     {"role": "assistant", "content": out}],
                    tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                has_chat = False
                t = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{user}\n\n### Response: {out}"
                )
        else:
            t = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{user}\n\n### Response: {out}"
            )
        texts.append(t)
    rng.shuffle(texts)

    eos = tokenizer.eos_token or ""
    ids_list = [tokenizer(t + eos, return_tensors="pt", truncation=False)["input_ids"]
                for t in texts]
    print(f"[alpaca-train] sampled {target}/{len(ds)} (2*num_samples={2*num_samples}) "
          f"from yahma/alpaca-cleaned -> {len(texts)} examples")
    ids = _pack_ids(ids_list, context_length)
    return ids, ids.clone()


def prep_metamath_train(tokenizer, num_samples: int,
                        context_length: int = 2048, seed: int = 0
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build MetaMathQA training data with prompt-prefix masking in labels.

    Parameters:
        tokenizer: Hugging Face tokenizer.
        num_samples: Number of examples to sample from MetaMathQA.
        context_length: Output chunk length.
        seed: RNG seed for sampling and shuffling.

    Returns:
        ids: Int tensor of shape (n, context_length).
        labels: Int tensor of shape (n, context_length), with prompt prefix tokens set to -100.
    """
    from datasets import load_dataset
    rng = _random.Random(seed)

    ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=seed)
    target = min(num_samples, len(ds))
    indices = rng.sample(range(len(ds)), target)

    prompt_prefix_tpl = _GSM8K_MATH_PROMPT.split("{response}")[0]

    examples = []
    for i in indices:
        ex = ds[i]
        examples.append((ex["query"], ex["response"]))
    rng.shuffle(examples)

    eos = tokenizer.eos_token or ""
    ids_list, labels_list = [], []
    for query, response in examples:
        prompt = prompt_prefix_tpl.format(instruction=query)
        full = prompt + response + eos
        prompt_ids = tokenizer(prompt, return_tensors="pt",
                               truncation=False)["input_ids"]
        full_ids = tokenizer(full, return_tensors="pt",
                             truncation=False)["input_ids"]
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        ids_list.append(full_ids)
        labels_list.append(labels)

    print(f"[metamath-train] sampled {target}/{len(ds)} from MetaMathQA "
          f"-> {len(ids_list)} examples (prompt prefix masked)")
    return _pack_ids_pair(ids_list, labels_list, context_length)


def prep_train_data(model_tag: str, tokenizer, num_samples: int,
                    context_length: int = 2048, seed: int = 0,
                    train_data_source: str | None = None,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch to the training-data builder for the given model_tag or explicit data source.

    Default source per model_tag:
      wm -> metamath, mc -> magicoder, chat -> alpaca, llava -> alpaca.

    Parameters:
        model_tag: Tag used to pick the default training corpus.
        tokenizer: Hugging Face tokenizer.
        num_samples: Number of samples requested from the chosen builder.
        context_length: Output chunk length.
        seed: RNG seed forwarded to the builder.
        train_data_source: Optional explicit source: "metamath" | "magicoder" | "alpaca".

    Returns:
        ids: Int tensor of shape (n, context_length).
        labels: Int tensor of shape (n, context_length); -100 on prompt tokens for metamath, else a clone of ids.
    """
    src = train_data_source or {
        "wm": "metamath", "mc": "magicoder",
        "chat": "alpaca", "llava": "alpaca",
    }.get(model_tag)
    if src is None:
        raise ValueError(f"unknown model_tag={model_tag!r}")

    if src == "metamath":
        return prep_metamath_train(tokenizer, num_samples=num_samples,
                                   context_length=context_length, seed=seed)
    if src == "magicoder":
        return prep_magicoder(tokenizer, num_samples=num_samples,
                              context_length=context_length, seed=seed)
    if src == "alpaca":
        return prep_alpaca_train(tokenizer, num_samples=num_samples,
                                 context_length=context_length, seed=seed)
    raise ValueError(f"unknown train_data_source={src!r}")


def get_args(argv=None):
    """
    Parse the Sage-D pipeline CLI arguments.

    Parameters:
        argv: Optional argv list to parse; defaults to sys.argv[1:].

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sage-D delta-compression pipeline "
                    "(delta calc -> SVD -> compress -> tune -> merge -> eval)"
    )
    parser.add_argument("--config", type=str, default="./config.yaml",
                        help="Path to config file")
    parser.add_argument("--model_tag", type=str, required=True,
                        choices=["wm", "mc", "chat", "llava"],
                        help="Which model_tag block of config.yaml to use")
    parser.add_argument("--save_root", type=str, required=True,
                        help="Output dir; delta/compressed/tuned go inside")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")


    parser.add_argument("--alpha", type=float, default=0.03125,
                        help="compression ratio (1/32 = 0.03125)")
    parser.add_argument("--group_size", type=int, default=128,
                        help="quantization group size (rank quantum)")
    parser.add_argument("--bits", type=int, default=4,
                        help="quantization bit-width")
    parser.add_argument("--context_length", type=int, default=2048,
                        help="training / tuning context length")


    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_seqlen", type=int, default=2048)
    parser.add_argument("--calib_chunk", type=int, default=4)
    parser.add_argument("--beta_steps", type=int, default=11)


    parser.add_argument("--save_per_epoch", action="store_true", default=False)
    parser.add_argument("--llava_template", type=str, default=None,
                        help="(llava + train_data_source=llava_multimodal) "
                             "path to llava-1.5-7b-hf reference dir")
    return parser.parse_args(argv)


def load_yaml_config(yaml_path):
    """
    Load a YAML config file into a Python dict.

    Parameters:
        yaml_path: Path to the YAML file.

    Returns:
        Parsed config object (typically a dict).
    """
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError:
            raise ValueError("Yaml error - check yaml file")


def print_pipeline_info(args, cfg_tag, paths):
    """
    Print a banner summarizing the model paths and output locations for one pipeline run.

    Parameters:
        args: Parsed CLI namespace (uses model_tag, config, save_root).
        cfg_tag: Resolved per-tag config block (uses base_model, finetuned).
        paths: Dict of named output paths to display.
    """
    print("=" * 60)
    print(f"SAGE-D BUILD PIPELINE — model_tag={args.model_tag}")
    print(f"  config:    {args.config}")
    print(f"  base:      {cfg_tag.base_model}")
    print(f"  finetuned: {cfg_tag.finetuned}")
    print(f"  save_root: {args.save_root}")
    for k, v in paths.items():
        print(f"  {k:11s}: {v}")
    print("=" * 60)


def _print_table(title, d, indent=2, max_val=80):
    """
    Print a key/value dict as a titled, indented two-column table.

    Parameters:
        title: Section title shown above the table.
        d: Dict to render; keys and values are stringified.
        indent: Leading spaces per row.
        max_val: Maximum value width; longer values are truncated with an ellipsis.
    """
    print(title)
    print("-" * max(len(title), 40))
    if not d:
        print(f"{' ' * indent}(empty)")
        print()
        return
    width = max(len(str(k)) for k in d.keys())
    for k, v in d.items():
        s = str(v)
        if len(s) > max_val:
            s = s[:max_val - 3] + "..."
        print(f"{' ' * indent}{str(k):<{width}}  {s}")
    print()
