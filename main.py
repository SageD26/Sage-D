"""Sage-D pipeline entry point: SVD -> compress -> tune -> merge -> eval."""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import sys
import time
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import (
    merge,
    merge_llava,

    collect_activation_covariance,
    r_probe_from_alpha,
    compute_sensitivity,
    total_rank_budget,
    allocate_ranks_proportional,

    LINEAR_SUBMODULE_NAMES,
    compress_layer,
    reconstruct_delta,

    DeltaSVDLinear_QUANT,
    _build_delta_svd_modules_from_delta,
    _clear_memory,
    _restore_beta_absorb_and_hadamard,
    absorb_sigma_into_uv,
    change_model_quant,
    _load_quant_state_from_compressed,
    _TuneArgs,
    train_post_quant,
    _export_module_to_packed,
)
from utils import (
    load_packed, save_packed, get_calibration, prep_train_data,
    get_args, load_yaml_config, _print_table,
)
from eval import evaluate


def derive_paths(save_root: str, model_tag: str) -> dict:
    """
    Build the standard pipeline-artifact paths under save_root, keyed by model_tag.

    Parameters:
        save_root: Output root directory for this run.
        model_tag: Short model identifier (e.g. "wm", "mc") used as a filename prefix.

    Returns:
        paths: Dict with keys "delta", "compressed", "tuned" mapping to absolute file paths.
    """
    return {
        "delta":      os.path.join(save_root, f"{model_tag}_delta.pt"),
        "compressed": os.path.join(save_root, f"{model_tag}_compressed.pt"),
        "tuned":      os.path.join(save_root, f"{model_tag}_tuned.pt"),
    }


if __name__ == "__main__":


    if len(sys.argv) > 1:
        args = get_args(sys.argv[1:])
    else:
        argv = [
            "--config",     "./config/wizardcoder13b.yaml",
            "--model_tag",  "mc",
            "--save_root",  "./out",
            "--save_per_epoch",
        ]
        args = get_args(argv)
    config = load_yaml_config(args.config)
    cfg_tag = config[args.model_tag]

    os.makedirs(args.save_root, exist_ok=True)

    paths = derive_paths(args.save_root, args.model_tag)
    paths["data_root"]  = "'your_path'"
    paths["merged"]     = os.path.join(args.save_root, ".result", "merged", args.model_tag)
    paths["eval_out"]   = os.path.join(args.save_root, ".result", "logs",   args.model_tag)
    os.makedirs(os.path.dirname(paths["merged"]),   exist_ok=True)
    os.makedirs(os.path.dirname(paths["eval_out"]), exist_ok=True)


    print()
    _print_table("ARGS (CLI)", vars(args))

    print(f"CONFIG  (loaded from {args.config})")
    print("-" * 40)
    print(f"  available model tags : {', '.join(config.keys())}")
    print()

    _print_table(f"cfg_tag[{args.model_tag!r}]", cfg_tag)
    _print_table("PATHS", paths)


    base_mod = AutoModelForCausalLM.from_pretrained(
        os.path.join(config["model_root"], "base_model", cfg_tag["base_model"]),
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    ft_mod = AutoModelForCausalLM.from_pretrained(
        os.path.join(config["model_root"], "finetuned_model", cfg_tag["finetuned"]),
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    tok = AutoTokenizer.from_pretrained(
        os.path.join(config["model_root"], "finetuned_model", cfg_tag["finetuned"])
    )


    original_bit = next(iter(base_mod.parameters())).element_size() * 8
    calib_ids = get_calibration(cfg_tag["calib_dataset"], tok,
                                n_samples=args.calib_samples,
                                seqlen=args.calib_seqlen)
    print(f"original_bit={original_bit}  "
          f"calib_dataset={cfg_tag['calib_dataset']}")
    print(f"loading {cfg_tag['calib_dataset']} calibration...")
    print(f"calib ids shape: {tuple(calib_ids.shape)}")

    covariances = collect_activation_covariance(
        ft_mod, calib_ids, device=args.device, chunk_size=args.calib_chunk
    )
    print(f"covariances for {len(covariances)} Linear layers.")

    cache = {}
    layer_specs = []
    sensitivity_scores = {}
    ft_sd = ft_mod.state_dict()
    for k, v in tqdm(base_mod.state_dict().items(), desc="SVD pass1"):
        if not (("self_attn" in k or "mlp" in k) and k.endswith(".weight")):
            continue
        delta = ft_sd[k] - v
        out_dim, in_dim = v.shape
        full_name = k.replace(".weight", "")
        cov = covariances[full_name]

        delta_gpu = delta.to(device=args.device, dtype=torch.float32)
        U, S, V = torch.svd(delta_gpu)
        del delta_gpu
        S_cpu = S.detach().cpu().to(torch.float32)

        r_probe = r_probe_from_alpha(in_dim, out_dim, args.alpha,
                                     original_bit, args.bits)
        s_l = compute_sensitivity(
            S, V, cov, W_base=v, delta=delta, r_probe=r_probe, device=args.device,
        )
        sensitivity_scores[k] = s_l

        cache[k] = (U.detach().cpu().to(torch.float16),
                    S_cpu,
                    V.detach().cpu().to(torch.float16),
                    v.detach().cpu())
        layer_specs.append({"name": k, "in_dim": in_dim, "out_dim": out_dim})
        del U, S, V, delta
        torch.cuda.empty_cache()

    del ft_sd, base_mod, ft_mod
    torch.cuda.empty_cache()
    gc.collect()


    R_total = total_rank_budget(layer_specs, args.alpha,
                                original_bit=original_bit, quant_bit=args.bits,
                                group_size=args.group_size)
    r_min = args.group_size * cfg_tag["rank_min"]
    print(f"[svd] R_total={R_total}  group_size={args.group_size}  "
          f"rank_min={cfg_tag['rank_min']}  r_min={r_min}")

    ranks = allocate_ranks_proportional(
        sensitivity_scores, layer_specs, R_total,
        r_min=r_min, group_size=args.group_size,
    )
    r_vals = list(ranks.values())
    print(f"[svd] rank min/mean/max={min(r_vals)}/"
          f"{sum(r_vals)/len(r_vals):.1f}/{max(r_vals)}  sum={sum(r_vals)}")
    assert all(r % args.group_size == 0 for r in r_vals), \
        "every layer rank must be a multiple of group_size"

    param_dict = {}
    for k, (U, S, V, base) in cache.items():
        r = ranks[k]
        kk = k.replace(".weight", "")
        param_dict[kk + ".base"] = base
        param_dict[kk + ".U"] = U[:, :r].contiguous().to(torch.bfloat16)
        param_dict[kk + ".S"] = S[:r].contiguous().to(torch.bfloat16)
        param_dict[kk + ".V"] = V[:, :r].contiguous().to(torch.bfloat16)
    torch.save(param_dict, paths["delta"])
    print(f"[svd] wrote {paths['delta']}")


    from collections import defaultdict

    delta_sd = torch.load(paths["delta"], map_location="cpu")
    layer_ids = set()
    for k in delta_sd:
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            layer_ids.add(int(parts[2]))
    layer_ids = sorted(layer_ids)


    generator = torch.Generator().manual_seed(args.seed)
    out_sd = {}
    total_err = 0.0
    count = 0
    beta_records = {}

    for li in tqdm(layer_ids, desc="Compressing"):
        for proj in LINEAR_SUBMODULE_NAMES:
            prefix = f"model.layers.{li}.{proj}"
            if f"{prefix}.base" not in delta_sd:
                continue
            U = delta_sd[f"{prefix}.U"].to(args.device)
            S = delta_sd[f"{prefix}.S"].to(args.device)
            V = delta_sd[f"{prefix}.V"].to(args.device)

            packed, beta_list = compress_layer(
                U, S, V,
                bits=args.bits,
                group_size=args.group_size,
                generator=generator,
            )

            with torch.no_grad():
                delta_full = U.float() @ torch.diag(S.float()) @ V.float().T
                delta_q = reconstruct_delta(packed, args.group_size).to(delta_full.device)
                err = ((delta_q - delta_full).norm() / delta_full.norm().clamp_min(1e-8)).item()
                total_err += err
                count += 1

            for field, tensor in packed.items():
                out_sd[f"{prefix}.{field}"] = tensor
            beta_records[f"layers.{li}.{proj}"] = beta_list
            del U, S, V, packed, delta_q, delta_full
            if args.device == "cuda":
                torch.cuda.empty_cache()


    flat_betas = [b for vals in beta_records.values() for b in vals]
    if flat_betas:
        beta_t = torch.tensor(flat_betas)
        print(f"Beta stats (per-group)  mean={beta_t.mean():.3f}  "
              f"min={beta_t.min():.3f}  max={beta_t.max():.3f}  "
              f"std={beta_t.std():.3f}  n={len(flat_betas)}")


        hist_lo, hist_hi, n_bins = 0.4, 0.7, 31
        bins = torch.linspace(hist_lo, hist_hi, n_bins)
        counts = [0] * n_bins
        for b in flat_betas:
            idx = min(range(n_bins), key=lambda i: abs(bins[i].item() - b))
            counts[idx] += 1
        print(f"Beta histogram (range [{hist_lo}, {hist_hi}], {n_bins} bins):")
        for b, c in zip(bins.tolist(), counts):
            print(f"  {b:.3f}: {c:4d} {'#' * (c // 2)}")
        by_proj = defaultdict(list)
        for key, vals in beta_records.items():
            by_proj[key.split(".", 2)[-1]].extend(vals)
        print("Beta by proj (mean over all layers × groups):")
        for proj, vals in sorted(by_proj.items()):
            t = torch.tensor(vals)
            print(f"  {proj:20s}  mean={t.mean():.3f}  std={t.std():.3f}  "
                  f"min={t.min():.3f}  max={t.max():.3f}")

    meta = {
        "betas": beta_records,
        "bits": args.bits,
        "group_size": args.group_size,
        "format": "packed_v1",
    }


    per_layer_packed: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in out_sd.items():
        prefix, field = k.rsplit(".", 1)
        per_layer_packed.setdefault(prefix, {})[field] = v
    out_sd = None
    gc.collect()

    save_packed(paths["compressed"], per_layer_packed, meta)
    print(f"[stage 2] wrote {paths['compressed']}  "
          f"size={os.path.getsize(paths['compressed']) / 1024**3:.2f} GB")


    print(f"packed layers : {len(per_layer_packed)}")
    print(f"meta keys     : {list(meta)}")
    print(f"  bits        = {meta['bits']}")
    print(f"  group_size  = {meta['group_size']}")


    print(f"\n[STAGE 3/3] {time.strftime('%H:%M:%S')} post-quant tuning")
    t0 = time.time()

    train_data_source = cfg_tag.get("train_data_source")
    base_model_path = os.path.join(
        config["model_root"], "base_model", cfg_tag["base_model"]
    )
    ft_model_path = os.path.join(
        config["model_root"], "finetuned_model", cfg_tag["finetuned"]
    )


    if not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)


    group_size = int(meta["group_size"])
    print(f"[in-memory] {len(per_layer_packed)} layers; meta keys={list(meta)}")

    print(f"[load] delta (un-quant): {paths['delta']}")
    delta_sd = torch.load(paths["delta"], map_location="cpu")

    print(f"[load] base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
    )
    print(f"[load] finetuned model (frozen ref): {ft_model_path}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path, torch_dtype=torch.bfloat16,
    )

    base_model = base_model.to(args.device)
    ft_model = ft_model.to(args.device)

    base_model.set_input_embeddings(ft_model.get_input_embeddings())
    base_model.set_output_embeddings(ft_model.get_output_embeddings())
    base_model.config.vocab_size = ft_model.config.vocab_size
    base_model.config.pad_token_id = ft_model.config.pad_token_id


    _build_delta_svd_modules_from_delta(base_model, delta_sd, group_size, args.device)
    del delta_sd
    _clear_memory()


    _restore_beta_absorb_and_hadamard(base_model, meta, group_size, seed=args.seed)

    absorb_sigma_into_uv(base_model)


    bits = int(meta.get("bits"))
    change_model_quant(base_model, quant_bit=bits, group_size=group_size,
                       param_dtype=torch.bfloat16)
    _clear_memory()


    _load_quant_state_from_compressed(base_model, per_layer_packed)


    print(f"[data] preparing train data for model_tag={args.model_tag!r}, "
          f"num_train_samples={cfg_tag['num_train_samples']}")
    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids, labels = prep_train_data(
        args.model_tag, tokenizer, num_samples=cfg_tag["num_train_samples"],
        context_length=args.context_length, seed=args.seed,
        train_data_source=train_data_source,
    )
    print(f"[data] sequences={ids.shape[0]}, ctx={args.context_length}, "
          f"train_data_source={train_data_source or '(default per model_tag)'}")
    train_loader = DataLoader(
        TensorDataset(ids, labels),
        batch_size=cfg_tag["train_batch_size"], shuffle=False,
        num_workers=32, pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
    )


    for p in ft_model.parameters():
        p.requires_grad = False
    ft_model.eval()


    args_ns = _TuneArgs(
        learning_rate=cfg_tag["learning_rate"],
        weight_decay=cfg_tag["weight_decay"],
        warmup_ratio=cfg_tag["warmup_ratio"],
        max_grad_norm=cfg_tag["max_grad_norm"],
        lm_loss_weight=cfg_tag["lm_loss_weight"],
        layer_recon_loss_weight=cfg_tag["layer_recon_loss_weight"],
        logit_distill_weight=cfg_tag["logit_distill_weight"],
        grad_accum=cfg_tag["grad_accum"],
        quant_train_epoch=cfg_tag["quant_train_epoch"],
    )
    print("=" * 60)
    print(f"STEP 2: post-quant training (model_tag={args.model_tag}) "
          f"quant_train_epochs={cfg_tag['quant_train_epoch']}")
    print("=" * 60)


    if args.save_per_epoch:
        def _epoch_ckpt_cb(epoch_num, model_):
            """
            Snapshot every DeltaSVDLinear_QUANT into a packed_v1 file after one epoch.

            Parameters:
                epoch_num: 1-indexed epoch number reported by the training loop.
                model_: Student model whose quantized SVD modules are exported.
            """
            decoder_ = model_.model
            ep_per_layer: dict[str, dict] = {}
            for li_ in range(len(decoder_.layers)):
                for sn_ in LINEAR_SUBMODULE_NAMES:
                    try:
                        mod_ = decoder_.layers[li_].get_submodule(sn_)
                    except AttributeError:
                        continue
                    if not isinstance(mod_, DeltaSVDLinear_QUANT):
                        continue
                    prefix_ = f"model.layers.{li_}.{sn_}"
                    ep_per_layer[prefix_] = _export_module_to_packed(mod_)
            ep_path = paths["tuned"].replace(".pt", f".epoch{epoch_num}.pt")
            save_packed(ep_path, ep_per_layer, meta)
            print(f"[ckpt] epoch {epoch_num} -> {ep_path}")
        cb = _epoch_ckpt_cb
    else:
        cb = None

    train_post_quant(base_model, ft_model, train_loader, args_ns, args.device,
                     epoch_callback=cb)

    del ft_model
    _clear_memory()


    decoder = base_model.model
    out_per_layer: dict[str, dict] = {}
    for li in tqdm(range(len(decoder.layers)), desc="Export tuned"):
        for sn in LINEAR_SUBMODULE_NAMES:
            try:
                mod = decoder.layers[li].get_submodule(sn)
            except AttributeError:
                continue
            if not isinstance(mod, DeltaSVDLinear_QUANT):
                continue
            prefix = f"model.layers.{li}.{sn}"
            out_per_layer[prefix] = _export_module_to_packed(mod)

    save_packed(paths["tuned"], out_per_layer, meta)
    print(f"[done] trained {len(out_per_layer)} modules in {time.time() - t0:.1f}s")
    print(f"[done] wrote {paths['tuned']}")


    per_layer_t, meta_t = load_packed(paths["tuned"])
    print(f"tuned.pt layers: {len(per_layer_t)}")
    print(f"meta keys      : {list(meta_t)}")
    print(f"file size      : {os.path.getsize(paths['tuned']) / 1024**3:.2f} GB")


    for _name in ("base_model", "decoder", "out_per_layer", "tokenizer",
                  "train_loader", "ids", "labels"):
        globals().pop(_name, None)
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    if args.model_tag == "llava":
        if not args.llava_template:
            raise ValueError("--llava_template required for model_tag=llava "
                             "(e.g. /path/to/llava-v1.5-7b)")
        merge_llava(
            template_dir=args.llava_template,
            base_model=base_model_path,
            delta_path=paths["tuned"],
            save_path=paths["merged"],
        )
    else:
        merge(
            finetuned_model=ft_model_path,
            base_model=base_model_path,
            delta_path=paths["tuned"],
            save_path=paths["merged"],
            device=args.device,
        )
    print(f"\n[stage 4] wall = {time.time() - t0:.1f}s")


    assert os.path.isdir(paths["merged"]), paths["merged"]
    files = sorted(os.listdir(paths["merged"]))
    print(f"merged dir : {paths['merged']}")
    print(f"entries    : {len(files)}")
    for f in files:
        p = os.path.join(paths["merged"], f)
        sz = os.path.getsize(p) / 1024**2
        print(f"  {f:40s} {sz:8.1f} MB")
    have_cfg = os.path.exists(os.path.join(paths["merged"], "config.json"))
    print(f"config.json present: {have_cfg}")


    for _name in ("base_model", "ft_model", "model", "tokenizer", "base_sd"):
        globals().pop(_name, None)
    gc.collect()
    torch.cuda.empty_cache()

    os.makedirs(paths["eval_out"], exist_ok=True)
    t0 = time.time()
    eval_results = evaluate(
        model_tag=args.model_tag,
        model_dir=paths["merged"],
        data_root=paths["data_root"],
        output_dir=paths["eval_out"],
    )
    print(f"\n[stage 5] wall = {time.time() - t0:.1f}s")


    import json
    print("=" * 60)
    print(f"EVAL RESULTS — model_tag={args.model_tag}")
    print("=" * 60)
    for k, v in eval_results.items():
        if k in ("model_tag", "model_dir"):
            print(f"  {k:13s}: {v}")
            continue
        print(f"  {k}:")
        print("    " + json.dumps(v, indent=2, default=str).replace("\n", "\n    "))
    summary_path = os.path.join(paths["eval_out"], "summary.json")

    with open(summary_path, "w") as _f:
        json.dump(eval_results, _f, indent=2, default=str)
    print(f"\nsummary.json: {summary_path}")
