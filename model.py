"""SVD, compression, DeltaSVDLinear training, and merge utilities."""

from __future__ import annotations

import os
import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, SequentialLR, LinearLR as WarmupLR,
)
from transformers import AutoTokenizer, AutoModelForCausalLM


LINEAR_SUBMODULE_NAMES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def _is_pow2(n):
    """
    Check whether an integer is a positive power of two.

    Parameters:
        n: Integer to test.

    Returns:
        True if n is a positive power of two, else False.
    """
    return n > 0 and (n & (n - 1)) == 0


def _sylvester_hadamard(k, dtype=torch.float32):
    """
    Build a k x k orthonormal Hadamard matrix via Sylvester construction.

    Parameters:
        k: Side length of the Hadamard matrix (must be a power of two).
        dtype: Torch dtype for the returned matrix.

    Returns:
        A k x k orthonormal Hadamard matrix scaled by 1/sqrt(k).
    """
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < k:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(k)


def _random_orthogonal(k, dtype=torch.float32, generator=None):
    """
    Sample a k x k random orthogonal matrix via QR decomposition.

    Parameters:
        k: Side length of the orthogonal matrix.
        dtype: Torch dtype for sampling.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        A k x k orthogonal matrix.
    """
    A = torch.randn(k, k, dtype=dtype, generator=generator)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def build_block_hadamard(rank, n_blocks, generator=None):
    """
    Build a block-diagonal Hadamard (or random orthogonal) rotation matrix.

    Parameters:
        rank: Total rank dimension to rotate.
        n_blocks: Number of equal-sized diagonal blocks.
        generator: Optional torch.Generator for the random-orthogonal fallback.

    Returns:
        A (rank, rank) block-diagonal orthogonal matrix.
    """
    bs = rank // n_blocks
    blocks = []
    for _ in range(n_blocks):
        if _is_pow2(bs):
            blocks.append(_sylvester_hadamard(bs))
        else:
            blocks.append(_random_orthogonal(bs, generator=generator))
    return torch.block_diag(*blocks)


def quantize_U(U, bits, group_size):
    """
    Asymmetrically quantize U (out_f, rank) per group along the rank axis.

    Parameters:
        U: Float tensor of shape (out_f, rank).
        bits: Quantization bit-width.
        group_size: Group size along the rank axis.

    Returns:
        q: uint8 quantized tensor of shape (out_f, rank).
        scale: bfloat16 per-group scale of shape (out_f, n_groups).
        zero: bfloat16 per-group zero-point of shape (out_f, n_groups).
    """
    maxq = 2 ** bits - 1
    out_f, rank = U.shape
    assert rank % group_size == 0
    n_groups = rank // group_size
    U_r = U.reshape(out_f, n_groups, group_size)
    xmin = U_r.amin(dim=-1, keepdim=True)
    xmax = U_r.amax(dim=-1, keepdim=True)
    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(both_zero, torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin).clamp(min=1e-5) / maxq
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(U_r / scale) + zero, 0, maxq).to(torch.uint8)
    return (q.reshape(out_f, rank),
            scale.squeeze(-1).to(torch.bfloat16),
            zero.squeeze(-1).to(torch.bfloat16))


def dequantize_U(U_q, U_scale, U_zero, group_size):
    """
    Dequantize U (out_f, rank), snapping the zero-point to the nearest integer.

    Parameters:
        U_q: uint8 quantized U of shape (out_f, rank).
        U_scale: Per-group scale of shape (out_f, n_groups).
        U_zero: Per-group zero-point of shape (out_f, n_groups).
        group_size: Group size along the rank axis.

    Returns:
        Float32 dequantized U of shape (out_f, rank).
    """
    out_f, rank = U_q.shape
    n_groups = rank // group_size
    q = U_q.reshape(out_f, n_groups, group_size).to(torch.float32)
    s = U_scale.to(torch.float32).unsqueeze(-1)
    z = U_zero.to(torch.float32).round().unsqueeze(-1)
    return ((q - z) * s).reshape(out_f, rank)


def quantize_VT(V_T, bits, group_size):
    """
    Asymmetrically quantize V_T (rank, in_f) per group along the rank axis.

    Parameters:
        V_T: Float tensor of shape (rank, in_f).
        bits: Quantization bit-width.
        group_size: Group size along the rank axis.

    Returns:
        q: uint8 quantized tensor of shape (rank, in_f).
        scale: bfloat16 per-group scale of shape (n_groups, in_f).
        zero: bfloat16 per-group zero-point of shape (n_groups, in_f).
    """
    maxq = 2 ** bits - 1
    rank, in_f = V_T.shape
    assert rank % group_size == 0
    n_groups = rank // group_size
    V_r = V_T.reshape(n_groups, group_size, in_f)
    xmin = V_r.amin(dim=1, keepdim=True)
    xmax = V_r.amax(dim=1, keepdim=True)
    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(both_zero, torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin).clamp(min=1e-5) / maxq
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(V_r / scale) + zero, 0, maxq).to(torch.uint8)
    return (q.reshape(rank, in_f),
            scale.squeeze(1).to(torch.bfloat16),
            zero.squeeze(1).to(torch.bfloat16))


def dequantize_VT(V_T_q, V_T_scale, V_T_zero, group_size):
    """
    Dequantize V_T (rank, in_f), snapping the zero-point to the nearest integer.

    Snaps z to nearest int to match training-time STE.

    Parameters:
        V_T_q: uint8 quantized V_T of shape (rank, in_f).
        V_T_scale: Per-group scale of shape (n_groups, in_f).
        V_T_zero: Per-group zero-point of shape (n_groups, in_f).
        group_size: Group size along the rank axis.

    Returns:
        Float32 dequantized V_T of shape (rank, in_f).
    """
    rank, in_f = V_T_q.shape
    n_groups = rank // group_size
    q = V_T_q.reshape(n_groups, group_size, in_f).to(torch.float32)
    s = V_T_scale.to(torch.float32).unsqueeze(1)
    z = V_T_zero.to(torch.float32).round().unsqueeze(1)
    return ((q - z) * s).reshape(rank, in_f)


def _group_error(beta, U_g, V_g, S_g):
    """
    Compute the sum of squared per-column/row dynamic ranges after sigma absorption.

    Parameters:
        beta: Absorption exponent in [0, 1] applied to S on the U side.
        U_g: Group slice of U, shape (out_f, group_size).
        V_g: Group slice of V_T, shape (group_size, in_f).
        S_g: Group slice of singular values, shape (group_size,).

    Returns:
        Scalar float quantifying total quantization-relevant range.
    """
    U_s = U_g * S_g.pow(beta).unsqueeze(0)
    V_s = S_g.pow(1.0 - beta).unsqueeze(1) * V_g
    u_range = U_s.amax(dim=1) - U_s.amin(dim=1)
    v_range = V_s.amax(dim=0) - V_s.amin(dim=0)
    return (u_range.pow(2).sum() + v_range.pow(2).sum()).item()


def _golden_section(f, lo=0.0, hi=1.0, tol=1e-3, max_iter=30):
    """
    Minimize a 1-D unimodal function over [lo, hi] via golden-section search.

    Parameters:
        f: Scalar function to minimize.
        lo: Lower bound of the search interval.
        hi: Upper bound of the search interval.
        tol: Tolerance on the bracket width for early stopping.
        max_iter: Maximum number of iterations.

    Returns:
        Approximate minimizer (midpoint of the final bracket).
    """
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def find_optimal_betas_per_group(U, S, V_T, group_size, tol=1e-4,
                                 lo=0.4, hi=0.7):
    """
    Search optimal sigma-absorption exponents beta per rank-axis group.

    Parameters:
        U: Float tensor of shape (out_f, rank).
        S: Singular values of shape (rank,).
        V_T: Float tensor of shape (rank, in_f).
        group_size: Group size along the rank axis.
        tol: Tolerance for the inner golden-section search.
        lo: Lower bound for beta in each group.
        hi: Upper bound for beta in each group.

    Returns:
        Float32 tensor of shape (n_groups,) with the optimal beta per group.
    """
    out_f, rank = U.shape
    n_groups = rank // group_size
    betas = torch.zeros(n_groups, dtype=torch.float32)
    for g in range(n_groups):
        cs = slice(g * group_size, (g + 1) * group_size)
        U_g = U[:, cs]
        V_g = V_T[cs, :]
        S_g = S[cs]
        betas[g] = _golden_section(
            lambda a: _group_error(a, U_g, V_g, S_g),
            lo=lo, hi=hi, tol=tol,
        )
    return betas


def compress_layer(U, S, V, bits, group_size, generator):
    """
    Compress one projection's SVD factors into a packed quantized layer dict.

    Assumes the SVD step has already produced ranks that are multiples of
    `group_size` (via group_size-aligned allocation), so per-group reshape
    is clean without padding. Uses a single full-rank Hadamard (n_blocks=1),
    falling back to random orthogonal if rank is not a power of 2.

    Parameters:
        U: Float tensor of shape (out_f, rank) from SVD.
        S: Float singular values of shape (rank,).
        V: Float tensor of shape (in_f, rank) from SVD.
        bits: Uniform quantization bit-width.
        group_size: Group size along the rank axis.
        generator: torch.Generator for the random-orthogonal fallback.

    Returns:
        packed: Dict with 8-field quantized layer state (U_q/scale/zero, V_T_q/scale/zero, A, B).
        betas_list: List of per-group beta values used for sigma absorption.
    """
    U = U.float()
    S = S.float().clamp(min=1e-8)
    V_T = V.transpose(-1, -2).contiguous().float()

    rank = S.shape[0]
    if rank % group_size != 0:
        raise ValueError(
            f"compress_layer: rank={rank} is not a multiple of "
            f"group_size={group_size}. Re-run svd with --group_size "
            f"{group_size} (and matching --k_min).")

    betas = find_optimal_betas_per_group(U, S, V_T, group_size=group_size)
    n_groups = rank // group_size
    for g in range(n_groups):
        cs = slice(g * group_size, (g + 1) * group_size)
        a = float(betas[g].item())
        U[:, cs] = U[:, cs] * S[cs].pow(a).unsqueeze(0)
        V_T[cs, :] = S[cs].pow(1.0 - a).unsqueeze(1) * V_T[cs, :]

    H = build_block_hadamard(rank, n_blocks=1, generator=generator).to(U.device)
    U = U @ H
    V_T = H.T @ V_T

    U_q, U_scale, U_zero = quantize_U(U, bits, group_size)
    V_T_q, V_T_scale, V_T_zero = quantize_VT(V_T, bits, group_size)

    out_f = U.shape[0]
    in_f = V_T.shape[1]

    packed = {
        "U_q": U_q.cpu(),
        "U_scale": U_scale.cpu(),
        "U_zero": U_zero.cpu(),
        "V_T_q": V_T_q.cpu(),
        "V_T_scale": V_T_scale.cpu(),
        "V_T_zero": V_T_zero.cpu(),
        "A": torch.ones(in_f, dtype=torch.bfloat16),
        "B": torch.ones(out_f, dtype=torch.bfloat16),
    }
    return packed, betas.tolist()


def reconstruct_delta(packed, group_size):
    """
    Dequantize a packed layer and apply A/B gains to return the fp32 delta.

    Reverse of compress_layer. A (in_f,) scales V_T's right (in_f dim);
    B (out_f,) scales U's left (out_f dim). Effective delta is
    diag(B) @ U @ V_T @ diag(A).

    Parameters:
        packed: Dict with 8-field quantized layer state.
        group_size: Group size along the rank axis used at quantize time.

    Returns:
        Float32 reconstructed delta tensor of shape (out_f, in_f).
    """
    U_dq = dequantize_U(packed["U_q"], packed["U_scale"], packed["U_zero"], group_size)
    V_T_dq = dequantize_VT(packed["V_T_q"], packed["V_T_scale"], packed["V_T_zero"], group_size)
    A = packed["A"].to(torch.float32)
    B = packed["B"].to(torch.float32)
    V_T_dq = V_T_dq * A.unsqueeze(0)
    U_dq = U_dq * B.unsqueeze(-1)
    return U_dq @ V_T_dq


def _diag_v_cov_v(V: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """
    Compute the diagonal of V^T @ cov @ V column-wise.

    Parameters:
        V: Tensor whose columns are evaluated, shape (d, rank).
        cov: Covariance matrix of shape (d, d).

    Returns:
        Non-negative tensor of shape (rank,) holding (V^T cov V)_{ii}.
    """
    cov_v = cov @ V
    return (V * cov_v).sum(dim=0).clamp(min=0)


def r_probe_from_alpha(in_dim: int, out_dim: int, alpha: float,
                       original_bit: int = 16, quant_bit: int = 4) -> int:
    """
    Derive a probe rank for a layer from a target compression ratio alpha.

    Parameters:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        alpha: Target compression ratio.
        original_bit: Bit-width of the original dense weights.
        quant_bit: Bit-width of the quantized low-rank factors.

    Returns:
        Probe rank clamped to [1, min(in_dim, out_dim)].
    """
    r = int(original_bit * in_dim * out_dim * alpha
            / (quant_bit * (in_dim + out_dim)))
    return max(1, min(r, min(in_dim, out_dim)))


@torch.no_grad()
def compute_sensitivity(sigma: torch.Tensor,
                           V: torch.Tensor,
                           cov: torch.Tensor,
                           W_base: torch.Tensor,
                           delta: torch.Tensor,
                           r_probe: int,
                           device: str = "cuda") -> float:
    """
    Compute the sensitivity score for a single layer at a given probe rank.

    Parameters:
        sigma: Singular values of the delta SVD, shape (rank,).
        V: Right singular vectors V (not V^T), shape (in_dim, rank).
        cov: Input activation covariance, shape (in_dim, in_dim).
        W_base: Base weight matrix.
        delta: Delta weight matrix added to W_base.
        r_probe: Probe rank at which to evaluate sensitivity.
        device: Device on which to run the computation.

    Returns:
        Scalar sensitivity score (weighted-tail energy normalized by full-weight energy).
    """
    full_rank = sigma.shape[0]
    r = max(0, min(int(r_probe), full_rank))
    if r >= full_rank:
        return 0.0

    sigma = sigma.to(device=device, dtype=torch.float32)
    sigma2 = sigma.pow(2)
    V_d = V.to(device=device, dtype=torch.float32)
    cov_d = cov.to(device=device, dtype=torch.float32)
    diag = _diag_v_cov_v(V_d, cov_d)
    weighted_tail = (sigma2 * diag)[r:].sum()

    W_ft = (W_base.to(device=device, dtype=torch.float32)
            + delta.to(device=device, dtype=torch.float32))
    cov_W_ft_T = cov_d @ W_ft.T
    denom = (W_ft * cov_W_ft_T.T).sum().clamp(min=1e-30)
    del W_ft, cov_W_ft_T
    return float((weighted_tail / denom).item())


@torch.no_grad()
def allocate_ranks_proportional(scores: dict[str, float],
                                layer_specs: list[dict],
                                R_total: int,
                                r_min: int = 1,
                                group_size: int = 1,
                                ) -> dict[str, int]:
    """
    Distribute R_total across layers proportionally to sensitivity scores.

    Distribute R_total across layers proportionally to scores in units of
    `group_size`; every returned rank is exactly a multiple of group_size, and
    water-fill cleanup preserves the multiple-of-group_size invariant.
    R_total and r_min must both be multiples of group_size.

    Parameters:
        scores: Mapping from layer name to non-negative sensitivity score.
        layer_specs: List of dicts with 'name', 'in_dim', 'out_dim' per layer.
        R_total: Total rank budget to distribute (multiple of group_size).
        r_min: Minimum rank per layer (multiple of group_size).
        group_size: Allocation unit; every assigned rank is a multiple of this.

    Returns:
        Mapping from layer name to allocated rank summing to R_total.
    """
    if group_size > 1:
        if R_total % group_size != 0:
            raise ValueError(
                f"R_total={R_total} must be a multiple of group_size={group_size}")

    if group_size > 1 and r_min % group_size != 0:
        raise ValueError(
            f"r_min={r_min} must be a multiple of group_size={group_size}")

    n = len(layer_specs)
    names = [s['name'] for s in layer_specs]

    unit = group_size
    B = R_total // unit
    u_min = r_min // unit
    u_max = [min(s['in_dim'], s['out_dim']) // unit for s in layer_specs]

    floor_total = u_min * n
    cap_total = sum(u_max)
    if B < floor_total:
        raise ValueError(
            f"R_total/{unit}={B} < floor_total={floor_total} "
            f"(r_min={r_min}, n_layers={n})")
    if B > cap_total:
        raise ValueError(
            f"R_total/{unit}={B} > cap_total={cap_total}")

    s_list = [max(0.0, float(scores[name])) for name in names]
    s_sum = sum(s_list)
    if s_sum <= 0:
        s_list = [1.0] * n
        s_sum = float(n)

    units = []
    for i in range(n):
        u_raw = B * s_list[i] / s_sum
        units.append(max(u_min, min(int(round(u_raw)), u_max[i])))

    diff = B - sum(units)
    if diff != 0:
        idx_by_score = sorted(range(n), key=lambda i: s_list[i], reverse=True)
        order = idx_by_score if diff > 0 else list(reversed(idx_by_score))
        i = 0
        while diff != 0 and i < 100 * n:
            li = order[i % n]
            if diff > 0 and units[li] < u_max[li]:
                units[li] += 1
                diff -= 1
            elif diff < 0 and units[li] > u_min:
                units[li] -= 1
                diff += 1
            i += 1
        if diff != 0:
            raise RuntimeError(f"water-fill could not converge, residual={diff}")

    assert sum(units) == B
    return {names[i]: units[i] * unit for i in range(n)}


def total_rank_budget(layer_specs: list[dict], alpha: float,
                      original_bit: int = 16, quant_bit: int = 4,
                      group_size: int = 1) -> int:
    """
    Compute the total rank budget across layers from a target compression ratio.

    Floors the sum to a multiple of group_size so allocate_ranks_proportional
    can work in units of group_size (group_size=1 is a no-op).

    Parameters:
        layer_specs: List of dicts with 'in_dim' / 'out_dim' per layer.
        alpha: Target compression ratio.
        original_bit: Bit-width of original dense weights.
        quant_bit: Bit-width of quantized low-rank factors.
        group_size: Multiple to which the total is floored.

    Returns:
        Total rank budget as a multiple of group_size.
    """
    raw = sum(
        r_probe_from_alpha(s['in_dim'], s['out_dim'], alpha,
                           original_bit=original_bit, quant_bit=quant_bit)
        for s in layer_specs
    )


    return (raw // group_size) * group_size


@torch.no_grad()
def collect_activation_covariance(model, calib_ids, device, chunk_size=4):
    """
    Run calibration data layer-by-layer and accumulate per-Linear input covariances.

    Parameters:
        model: HF causal LM whose Linear layers' input covariances are collected.
        calib_ids: Tokenized calibration inputs of shape (n_samples, seq_len).
        device: Device on which to run the forward passes.
        chunk_size: Mini-batch size for chunked forwarding.

    Returns:
        Mapping from full module name to (in, in) covariance tensor on CPU fp32.
    """
    model.eval()
    base = model.model

    all_hidden = []
    for i in range(0, calib_ids.shape[0], chunk_size):
        ids = calib_ids[i:i + chunk_size].to(device)
        all_hidden.append(base.embed_tokens(ids).cpu())
    all_hidden = torch.cat(all_hidden, dim=0)

    seq_len = calib_ids.shape[1]
    dtype = next(base.parameters()).dtype
    mask_val = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((seq_len, seq_len), mask_val, device=device, dtype=dtype),
        diagonal=1,
    )[None, None, :, :]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    rotary_emb = getattr(base, 'rotary_emb', None)

    covariances = {}
    for layer_idx, layer in enumerate(tqdm(base.layers, desc="Calib")):
        layer_covs, layer_counts = {}, {}
        hooks = []

        def make_hook(full_name):
            """
            Create a forward hook that accumulates input covariance for one Linear.

            Parameters:
                full_name: Fully qualified module name used as the storage key.

            Returns:
                Forward hook closure registered on a torch.nn.Linear module.
            """
            def fn(module, inp, out):
                """
                Forward hook: accumulate x^T x and a sample count for this Linear.

                Parameters:
                    module: The torch.nn.Linear the hook is attached to.
                    inp: Tuple of positional inputs to the module.
                    out: Module output (unused).
                """
                x = inp[0].detach().to(torch.float32).reshape(-1, inp[0].shape[-1])
                xtx = x.T @ x
                if full_name not in layer_covs:
                    layer_covs[full_name] = torch.zeros_like(xtx)
                    layer_counts[full_name] = 0
                layer_covs[full_name] += xtx
                layer_counts[full_name] += x.shape[0]
            return fn

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full = f"model.layers.{layer_idx}.{name}"
                hooks.append(module.register_forward_hook(make_hook(full)))

        next_hidden = []
        for i in range(0, all_hidden.shape[0], chunk_size):
            chunk = all_hidden[i:i + chunk_size].to(device)
            bs = chunk.shape[0]
            pos = position_ids.expand(bs, -1)
            kwargs = dict(attention_mask=causal.expand(bs, -1, -1, -1),
                          position_ids=pos)
            if rotary_emb is not None:
                kwargs['position_embeddings'] = rotary_emb(chunk, pos)
            out = layer(chunk, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            next_hidden.append(out.cpu())

        for h in hooks:
            h.remove()
        for full in layer_covs:
            covariances[full] = (layer_covs[full] / layer_counts[full]).cpu()
        del layer_covs, layer_counts

        all_hidden = torch.cat(next_hidden, dim=0)
        del next_hidden
        gc.collect()
        torch.cuda.empty_cache()

    return covariances


def _clear_memory():
    """
    Run Python GC and (if available) empty the CUDA caching allocator.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DeltaSVDLinear(nn.Module):
    """
    Linear layer implementing y = x @ W_base^T + diag(B) @ U @ diag(S) @ V_T @ diag(A).

    A (shape (in_features,)) scales V_T on its right (in_f axis); B (shape
    (out_features,)) scales U on its left (out_f axis). Both initialize to
    ones (= identity diagonal). S is a learnable diagonal that
    `absorb_sigma_into_uv()` later folds symmetrically into U and V_T.
    """

    def __init__(self, base: torch.Tensor, U: torch.Tensor, sigma: torch.Tensor,
                 V_T: torch.Tensor, in_features: int, out_features: int,
                 param_dtype=torch.bfloat16):
        """
        Initialize a DeltaSVDLinear module from precomputed SVD factors.

        Parameters:
            base: Base weight tensor of shape (out_features, in_features).
            U: Left singular vectors of shape (out_features, rank).
            sigma: Singular values of shape (rank,).
            V_T: Right singular vectors transposed, shape (rank, in_features).
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            param_dtype: Storage dtype for U, V_T, S, and the base buffer.
        """
        super().__init__()
        self.register_buffer("base", base.to(param_dtype))
        self.U = nn.Parameter(U.to(param_dtype), requires_grad=False)
        self.V_T = nn.Parameter(V_T.to(param_dtype), requires_grad=False)
        self.S = nn.Parameter(sigma.detach().to(param_dtype).clone())


        self.A = nn.Parameter(torch.ones(in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.ones(out_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning y = x @ W_base^T + the SVD delta path output.

        Parameters:
            x: Input activations of shape (..., in_features).

        Returns:
            Output activations of shape (..., out_features).
        """
        y_base = F.linear(x, self.base)
        pdt = self.U.dtype
        A = self.A.to(pdt)
        B = self.B.to(pdt)

        V_T_eff = self.V_T * A.unsqueeze(0)
        out = torch.matmul(x, V_T_eff.T)

        if hasattr(self, "S"):
            out = out * self.S

        out = torch.matmul(out, self.U.T) * B

        return y_base + out


@torch.no_grad()
def absorb_sigma_into_uv(model):
    """
    Symmetrically fold sqrt(S) into U and V_T on every DeltaSVDLinear, then delete S.

    Parameters:
        model: nn.Module containing DeltaSVDLinear submodules to be transformed in place.
    """
    count = 0
    for mod in model.modules():
        if not isinstance(mod, DeltaSVDLinear):
            continue
        if not hasattr(mod, "S"):
            continue
        S_data = mod.S.data.float().clamp(min=1e-8)
        sqrt_S = torch.sqrt(S_data)
        U_f = mod.U.data.float()
        V_T_f = mod.V_T.data.float()
        mod.U.data.copy_((U_f * sqrt_S.unsqueeze(0)).to(mod.U.dtype))
        mod.V_T.data.copy_((sqrt_S.unsqueeze(1) * V_T_f).to(mod.V_T.dtype))
        del mod.S
        count += 1
    print(f"[absorb_sigma_into_uv] folded sqrt(S) into U,V_T on {count} modules.")


def _get_minq_maxq(bits, sym=False):
    """
    Return (minq, maxq) bounds for symmetric or asymmetric `bits`-bit quantization.

    Parameters:
        bits: Quantization bit-width.
        sym: If True, use signed symmetric range; else use unsigned asymmetric range.

    Returns:
        minq: Minimum representable quantized value.
        maxq: Maximum representable quantized value.
    """
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2 ** bits - 1)
        minq = 0
    return minq, maxq


class DeltaSVDLinear_QUANT(nn.Module):
    """
    Quantized DeltaSVDLinear with single learnable A / B vectors.

    A (shape (in_features,)) scales V_T on its right (in_f axis); B (shape
    (out_features,)) scales U on its left (out_f axis). Both initialize to
    ones (= identity diagonal). Quantization groups along the rank axis are
    controlled by `group_size`, with uniform `bits` per group.
    """

    def __init__(self, svd_module: DeltaSVDLinear, group_size: int,
                 bits: int = 4, param_dtype=torch.bfloat16):
        """
        Build a quantized DeltaSVDLinear from a fitted DeltaSVDLinear.

        Uses uniform `bits` for every per-group quantization tile.

        Parameters:
            svd_module: Source DeltaSVDLinear providing base / U / V_T / S / A / B.
            group_size: Group size along the rank axis for quantization tiles.
            bits: Uniform quantization bit-width.
            param_dtype: Storage dtype for the dequantized factors and base buffer.
        """
        super().__init__()
        self.bits = bits
        self.param_dtype = param_dtype
        self.group_size = group_size

        self.register_buffer("base", svd_module.base.data.clone().to(param_dtype))

        U_data = svd_module.U.data.clone().float()
        V_T_data = svd_module.V_T.data.clone().float()
        if hasattr(svd_module, "S"):
            S_data = svd_module.S.data.clone().float().clamp(min=1e-8)
            sqrt_S = torch.sqrt(S_data)
            U_data = U_data * sqrt_S.unsqueeze(0)
            V_T_data = sqrt_S.unsqueeze(1) * V_T_data

        rank = U_data.shape[1]
        if rank % group_size != 0:
            raise ValueError(
                f"DeltaSVDLinear_QUANT: rank={rank} not a multiple of "
                f"group_size={group_size}.")
        n_groups = rank // group_size
        self._n_groups = n_groups


        self.A = nn.Parameter(svd_module.A.data.clone().to(torch.float32))
        self.B = nn.Parameter(svd_module.B.data.clone().to(torch.float32))

        with torch.no_grad():
            U_int_parts, U_scale_parts, U_zero_parts = [], [], []
            for g in range(n_groups):
                cs = slice(g * group_size, (g + 1) * group_size)
                qi, qs, qz = self._quantize(U_data[:, cs], dim=1)
                U_int_parts.append(qi); U_scale_parts.append(qs); U_zero_parts.append(qz)
            self.register_buffer("U_int", torch.cat(U_int_parts, dim=1))
            self.register_buffer("U_scale", torch.stack(U_scale_parts, dim=1))
            self.register_buffer("U_zero", torch.stack(U_zero_parts, dim=1))

            V_int_parts, V_scale_parts, V_zero_parts = [], [], []
            for g in range(n_groups):
                rs = slice(g * group_size, (g + 1) * group_size)
                qi, qs, qz = self._quantize(V_T_data[rs, :], dim=0)
                V_int_parts.append(qi); V_scale_parts.append(qs); V_zero_parts.append(qz)
            self.register_buffer("V_T_int", torch.cat(V_int_parts, dim=0))
            self.register_buffer("V_T_scale", torch.stack(V_scale_parts, dim=0))
            self.register_buffer("V_T_zero", torch.stack(V_zero_parts, dim=0))


        self.register_buffer("U_unquant", U_data.clone().to(param_dtype),
                             persistent=False)
        self.register_buffer("V_T_unquant", V_T_data.clone().to(param_dtype),
                             persistent=False)

    def _quantize(self, x, dim=-1):
        """
        Asymmetrically quantize a tensor along one axis to uint8 with fp scale/zero.

        Parameters:
            x: Float tensor to quantize.
            dim: Axis along which min/max are taken (collapsed to scalars per slice).

        Returns:
            q: uint8 quantized tensor matching x's shape.
            scale: Per-slice scale in param_dtype with `dim` squeezed out.
            zero: Per-slice zero-point in param_dtype with `dim` squeezed out.
        """
        x_fp32 = x.float()
        _, maxq = _get_minq_maxq(self.bits, sym=False)
        maxq = maxq.to(x_fp32.device)
        xmax = torch.amax(x_fp32, dim=dim, keepdim=True)
        xmin = torch.amin(x_fp32, dim=dim, keepdim=True)
        tmp = (xmin == 0) & (xmax == 0)
        xmin = xmin.clone(); xmax = xmax.clone()
        xmin[tmp] = -1; xmax[tmp] = +1
        scale = (xmax - xmin).clamp(min=1e-5) / maxq
        zero = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(x_fp32 / scale) + zero, 0, maxq)
        sq_dim = dim if dim >= 0 else dim + len(x.shape)
        return (q.to(torch.uint8),
                scale.squeeze(sq_dim).to(self.param_dtype),
                zero.squeeze(sq_dim).to(self.param_dtype))

    def _dequantize_U(self):
        """
        Dequantize the stored U_int into a full (out_f, rank) tensor.

        Returns:
            Dequantized U as a param_dtype tensor of shape (out_f, rank).
        """
        gs = self.group_size
        n_groups = self._n_groups
        parts = []
        for g in range(n_groups):
            cs = slice(g * gs, (g + 1) * gs)
            q = self.U_int[:, cs].to(self.param_dtype)
            s = self.U_scale[:, g].to(self.param_dtype)
            z = self.U_zero[:, g].to(self.param_dtype)
            parts.append((q - z.unsqueeze(1)) * s.unsqueeze(1))
        return torch.cat(parts, dim=1)

    def _dequantize_V_T(self):
        """
        Dequantize the stored V_T_int into a full (rank, in_f) tensor.

        Returns:
            Dequantized V_T as a param_dtype tensor of shape (rank, in_f).
        """
        gs = self.group_size
        n_groups = self._n_groups
        parts = []
        for g in range(n_groups):
            rs = slice(g * gs, (g + 1) * gs)
            q = self.V_T_int[rs, :].to(self.param_dtype)
            s = self.V_T_scale[g, :].to(self.param_dtype)
            z = self.V_T_zero[g, :].to(self.param_dtype)
            parts.append((q - z.unsqueeze(0)) * s.unsqueeze(0))
        return torch.cat(parts, dim=0)

    def _student_svd_path(self, x, return_intermediate=False):
        """
        Compute the SVD delta path using the live (quantized) student factors.

        Parameters:
            x: Input activations of shape (..., in_features).
            return_intermediate: If True, also return the intermediate after V_T.

        Returns:
            When return_intermediate is True: (intermediate, out) tuple where
            intermediate is x @ (V_T diag(A))^T and out is the final delta path.
            Otherwise: just out, the SVD delta path output.
        """
        pdt = self.param_dtype
        U_dq = self._dequantize_U()
        V_dq = self._dequantize_V_T()
        A = self.A.to(pdt)
        B = self.B.to(pdt)


        intermediate = torch.matmul(x, (V_dq * A.unsqueeze(0)).T)
        out = torch.matmul(intermediate, U_dq.T) * B

        if return_intermediate:
            return intermediate, out
        return out

    def layer_recon_components(self, x):
        """
        Compute the two-stage layer reconstruction losses against the un-quantized teacher.

        Parameters:
            x: Input activations of shape (..., in_features).

        Returns:
            None when teacher buffers are absent. Otherwise (l1, l2) where l1 is
            the MSE on the intermediate (x V_T^T) and l2 is the MSE on the full
            delta output (x V_T^T U^T).
        """
        if not hasattr(self, "U_unquant"):
            return None
        pdt = self.param_dtype
        x = x.to(pdt)


        with torch.no_grad():
            t_int = torch.matmul(x, self.V_T_unquant.T)
            t_full = torch.matmul(t_int, self.U_unquant.T)
        s_int, s_full = self._student_svd_path(x, return_intermediate=True)
        l1 = F.mse_loss(s_int, t_int)
        l2 = F.mse_loss(s_full, t_full)
        return l1, l2

    def forward(self, x):
        """
        Forward pass returning y = x @ W_base^T + quantized SVD delta path output.

        Parameters:
            x: Input activations of shape (..., in_features).

        Returns:
            Output activations of shape (..., out_features).
        """
        x = x.to(self.param_dtype)
        y_base = F.linear(x, self.base)
        out = self._student_svd_path(x)
        return y_base + out


def change_model_quant(model, quant_bit, group_size,
                       param_dtype=torch.bfloat16):
    """
    Replace every DeltaSVDLinear in the model with DeltaSVDLinear_QUANT in place.

    `group_size` controls the rank-axis quantization grouping (each module has
    rank // group_size groups), all quantized to uniform `quant_bit`.

    Parameters:
        model: HF causal LM whose decoder layers host DeltaSVDLinear modules.
        quant_bit: Uniform quantization bit-width.
        group_size: Group size along the rank axis.
        param_dtype: Storage dtype for the new quantized modules.

    Returns:
        The input model with modules swapped in place.
    """
    def replace(mod):
        """
        Recursively walk a module subtree and swap DeltaSVDLinear children for quantized ones.

        Parameters:
            mod: Module whose subtree is rewritten in place.
        """
        for name, child in list(mod.named_children()):
            if isinstance(child, DeltaSVDLinear):
                setattr(mod, name, DeltaSVDLinear_QUANT(
                    child, group_size=group_size, bits=quant_bit,
                    param_dtype=param_dtype))
            else:
                replace(child)
    for layer in tqdm(model.model.layers, desc="Quantizing"):
        replace(layer)
    return model


def _build_delta_svd_modules_from_delta(base_model, delta_sd: dict,
                                        group_size: int,
                                        device: str):
    """
    Replace base_model's targeted Linears with DeltaSVDLinear modules from a delta state dict.

    Initialized from delta.pt's (U, S, V, base) per layer/proj. The rank per
    layer is whatever delta.pt stored (already group_size-aligned by the SVD
    step). Quantization grouping along rank is decided later by
    DeltaSVDLinear_QUANT using `group_size`.

    Parameters:
        base_model: HF causal LM whose Linears are replaced in place.
        delta_sd: Dict mapping prefixed keys (.U/.S/.V/.base) to tensors.
        group_size: Group size along the rank axis; ranks must be multiples.
        device: Device to move the model to after building.

    Returns:
        The same `base_model` instance with DeltaSVDLinear modules attached.
    """
    param_d = torch.bfloat16
    decoder = base_model.model
    num_layers = len(decoder.layers)
    n_built = 0
    for li in tqdm(range(num_layers), desc="Build DeltaSVDLinear"):
        for sn in LINEAR_SUBMODULE_NAMES:
            prefix = f"model.layers.{li}.{sn}"
            U_key = f"{prefix}.U"
            if U_key not in delta_sd:
                continue
            U = delta_sd[f"{prefix}.U"].float()
            S = delta_sd[f"{prefix}.S"].float()
            V = delta_sd[f"{prefix}.V"].float()
            W_base = delta_sd[f"{prefix}.base"]
            out_f, rank = U.shape
            in_f = V.shape[0]
            if rank % group_size != 0:
                raise ValueError(
                    f"layer {prefix} rank={rank} is not a multiple of "
                    f"group_size={group_size} (was the SVD step run with "
                    f"--group_size {group_size}?)")
            V_T = V.transpose(-1, -2).contiguous()
            new_mod = DeltaSVDLinear(
                base=W_base.clone(), U=U, sigma=S, V_T=V_T,
                in_features=in_f, out_features=out_f, param_dtype=param_d,
            )
            parts = sn.split(".")
            parent = decoder.layers[li].get_submodule(".".join(parts[:-1]))
            setattr(parent, parts[-1], new_mod)
            n_built += 1
    base_model.to(device)
    print(f"[build] DeltaSVDLinear modules built: {n_built}")
    return base_model


def _restore_beta_absorb_and_hadamard(model, meta: dict, group_size: int,
                                        seed: int = 0):
    """
    Replay per-group beta absorption and Hadamard rotation on every DeltaSVDLinear.

    This makes the un-quantized U' / V' inside DeltaSVDLinear match what
    compressed.pt was quantized from, so DeltaSVDLinear_QUANT.U_unquant
    teacher is consistent.

    Parameters:
        model: HF causal LM containing DeltaSVDLinear modules to update in place.
        meta: Metadata dict from compressed.pt; must contain 'betas' mapping.
        group_size: Group size along the rank axis used during compression.
        seed: Seed for the torch.Generator driving the random-orthogonal fallback.
    """
    betas_map = meta.get("betas", {})
    generator = torch.Generator().manual_seed(seed)

    decoder = model.model
    n_done = 0
    for li, layer in enumerate(decoder.layers):
        for sn in LINEAR_SUBMODULE_NAMES:
            try:
                mod = layer.get_submodule(sn)
            except AttributeError:
                continue
            if not isinstance(mod, DeltaSVDLinear):
                continue

            key = f"layers.{li}.{sn}"
            beta_list = betas_map.get(key)
            if beta_list is None:
                raise KeyError(f"compressed.pt meta has no betas for {key}")

            U_f = mod.U.data.float()
            V_T_f = mod.V_T.data.float()
            S_f = mod.S.data.float().clamp(min=1e-8)
            rank = U_f.shape[1]
            n_groups = rank // group_size
            if len(beta_list) != n_groups:
                raise ValueError(
                    f"{key}: meta betas len={len(beta_list)} != n_groups={n_groups}")
            for g in range(n_groups):
                cs = slice(g * group_size, (g + 1) * group_size)
                a = float(beta_list[g])
                U_f[:, cs] = U_f[:, cs] * S_f[cs].pow(a).unsqueeze(0)
                V_T_f[cs, :] = S_f[cs].pow(1.0 - a).unsqueeze(1) * V_T_f[cs, :]


            H = build_block_hadamard(rank, n_blocks=1, generator=generator)
            H = H.to(U_f.device)
            U_f = U_f @ H
            V_T_f = H.T @ V_T_f

            mod.U.data.copy_(U_f.to(mod.U.dtype))
            mod.V_T.data.copy_(V_T_f.to(mod.V_T.dtype))


            mod.S.data.copy_(torch.ones_like(mod.S.data))
            n_done += 1
    print(f"[replay] beta-absorb + hadamard on {n_done} modules.")


def _load_quant_state_from_compressed(model, per_layer_packed: dict):
    """
    Overwrite each DeltaSVDLinear_QUANT's quantized state with values from compressed.pt.

    After change_model_quant() built DeltaSVDLinear_QUANT from the replayed
    un-quantized U/V_T, this guarantees the live student matches what merge
    would consume.

    Parameters:
        model: HF causal LM hosting DeltaSVDLinear_QUANT modules.
        per_layer_packed: Mapping from layer prefix to the 8-field packed dict.
    """
    decoder = model.model
    n_done = 0
    for li, layer in enumerate(decoder.layers):
        for sn in LINEAR_SUBMODULE_NAMES:
            try:
                mod = layer.get_submodule(sn)
            except AttributeError:
                continue
            if not isinstance(mod, DeltaSVDLinear_QUANT):
                continue
            prefix = f"model.layers.{li}.{sn}"
            packed = per_layer_packed[prefix]
            mod.U_int.data.copy_(packed["U_q"].to(mod.U_int.device))
            mod.V_T_int.data.copy_(packed["V_T_q"].to(mod.V_T_int.device))
            mod.U_scale.data.copy_(packed["U_scale"].to(
                device=mod.U_scale.device, dtype=mod.U_scale.dtype))
            mod.U_zero.data.copy_(packed["U_zero"].to(
                device=mod.U_zero.device, dtype=mod.U_zero.dtype))
            mod.V_T_scale.data.copy_(packed["V_T_scale"].to(
                device=mod.V_T_scale.device, dtype=mod.V_T_scale.dtype))
            mod.V_T_zero.data.copy_(packed["V_T_zero"].to(
                device=mod.V_T_zero.device, dtype=mod.V_T_zero.dtype))


            n_done += 1
    print(f"[init] copied quant state from compressed.pt into {n_done} modules.")


def _set_requires_grad(model, predicate):
    """
    Toggle requires_grad on each named parameter according to a predicate.

    Parameters:
        model: nn.Module whose parameters' requires_grad will be set.
        predicate: Callable(name) -> bool deciding which parameters to train.

    Returns:
        List of parameters that were enabled for gradient computation.
    """
    trainable = []
    for n, p in model.named_parameters():
        if predicate(n):
            p.requires_grad = True
            trainable.append(p)
        else:
            p.requires_grad = False
            p.grad = None
    return trainable


def install_layer_recon_hooks(model):
    """
    Install forward pre-hooks on every DeltaSVDLinear_QUANT to accumulate two-stage recon loss.

    Parameters:
        model: nn.Module containing DeltaSVDLinear_QUANT submodules.

    Returns:
        reset: Callable that clears the accumulated loss buffer for the next step.
        total: Callable returning the mean loss across hit modules (or None if zero).
        remove: Callable that removes all registered hook handles.
    """
    state = {"sum": None, "count": 0}
    handles = []

    def make_hook(mod):
        """
        Create a forward pre-hook that accumulates this module's recon loss into the shared state.

        Parameters:
            mod: The DeltaSVDLinear_QUANT instance whose teacher loss is collected.

        Returns:
            A forward pre-hook closure to register on the module.
        """
        def pre_hook(_module, inputs):
            """
            Forward pre-hook: add this module's two-stage recon loss to the shared running sum.

            Parameters:
                _module: Unused; the module the hook fires on.
                inputs: Tuple of inputs to the module's forward.
            """
            comps = mod.layer_recon_components(inputs[0])
            if comps is None:
                return
            l1, l2 = comps
            l = l1 + l2
            state["sum"] = l if state["sum"] is None else state["sum"] + l
            state["count"] += 1
        return pre_hook

    for mod in model.modules():
        if isinstance(mod, DeltaSVDLinear_QUANT) and hasattr(mod, "U_unquant"):
            handles.append(mod.register_forward_pre_hook(make_hook(mod)))

    def reset():
        """
        Clear the accumulated layer-recon sum and counter for the next training step.
        """
        state["sum"] = None
        state["count"] = 0

    def total():
        """
        Return the mean per-module two-stage recon loss accumulated since the last reset.

        Returns:
            Mean loss tensor, or None if no module contributed since reset.
        """
        if state["count"] == 0 or state["sum"] is None:
            return None
        return state["sum"] / state["count"]

    def remove():
        """
        Remove every forward pre-hook handle registered by install_layer_recon_hooks.
        """
        for h in handles:
            h.remove()

    return reset, total, remove


@torch.no_grad()
def clear_layer_recon_teachers(model):
    """
    Delete the non-persistent U_unquant / V_T_unquant teacher buffers from every quant module.

    Parameters:
        model: nn.Module containing DeltaSVDLinear_QUANT submodules.
    """
    cleared = 0
    for mod in model.modules():
        if not isinstance(mod, DeltaSVDLinear_QUANT):
            continue
        if hasattr(mod, "U_unquant"):
            del mod.U_unquant
            cleared += 1
        if hasattr(mod, "V_T_unquant"):
            del mod.V_T_unquant
    if cleared:
        print(f"[layer_recon] cleared teacher buffers on {cleared} modules.")


def _build_sched(optimizer, total_steps, warmup_steps, base_lr):
    """
    Build a linear-warmup + cosine-annealing learning rate schedule.

    Parameters:
        optimizer: torch optimizer whose learning rate is scheduled.
        total_steps: Total number of optimizer steps.
        warmup_steps: Number of linear-warmup steps.
        base_lr: Base learning rate used to set the cosine minimum.

    Returns:
        torch SequentialLR combining the warmup and cosine schedules.
    """
    warm = WarmupLR(optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=warmup_steps)
    cos = CosineAnnealingLR(optimizer,
                            T_max=max(1, total_steps - warmup_steps),
                            eta_min=base_lr * 0.01)
    return SequentialLR(optimizer, [warm, cos], milestones=[warmup_steps])


def _train_loop(model, ref_model, train_loader, args_ns, device,
                num_epochs: int, trainable_predicate, phase_name: str,
                epoch_callback=None):
    """
    Run a post-quant training loop combining LM CE, layer recon MSE, and KL distillation.

    Loss = lm_w * LM_CE + recon_w * mean per-module two-stage recon
    + distill_w * KL(ft || student). Reference is the FROZEN FINETUNED model.
    Recon collapses to 0 if no DeltaSVDLinear_QUANT modules are present
    (e.g. step 1 with un-quantized DeltaSVDLinear).

    Parameters:
        model: Student model to train.
        ref_model: Frozen fine-tuned teacher used for KL distillation logits.
        train_loader: Iterable yielding (input_ids, labels) batches.
        args_ns: Namespace with learning_rate, weight_decay, warmup_ratio,
            grad_accum, max_grad_norm, lm_loss_weight, layer_recon_loss_weight,
            logit_distill_weight.
        device: Device on which to move batches.
        num_epochs: Number of epochs to train.
        trainable_predicate: Callable(param_name) -> bool selecting trainable params.
        phase_name: Human-readable phase label used in logs and progress bars.
        epoch_callback: Optional fn(epoch_num, model) invoked after each epoch (1-indexed).
    """
    params = _set_requires_grad(model, trainable_predicate)
    if len(params) == 0:
        print(f"[{phase_name}] no trainable params; skipping.")
        return

    optimizer = torch.optim.AdamW(
        params, lr=args_ns.learning_rate, weight_decay=args_ns.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    total_steps = max(1, len(train_loader) * num_epochs // args_ns.grad_accum)
    warmup_steps = max(1, int(total_steps * args_ns.warmup_ratio))
    scheduler = _build_sched(optimizer, total_steps, warmup_steps,
                             args_ns.learning_rate)

    autocast = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)

    model.config.use_cache = False
    ref_model.config.use_cache = False

    recon_reset, recon_total, recon_remove = install_layer_recon_hooks(model)

    print(f"[{phase_name}] trainable tensors={len(params)}, "
          f"total_steps={total_steps}, warmup={warmup_steps}, lr={args_ns.learning_rate}")

    try:
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"[{phase_name}] epoch {epoch+1}/{num_epochs}")
            optimizer.zero_grad()

            for bi, batch in enumerate(pbar):
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)

                with torch.no_grad(), autocast():
                    ref_out = ref_model(input_ids, use_cache=False)
                ref_logits = ref_out.logits.detach()
                del ref_out

                with autocast():
                    recon_reset()
                    out = model(input_ids, labels=labels, use_cache=False)
                    lm = out.loss
                    recon = recon_total()
                    if recon is None:
                        recon = torch.zeros((), device=input_ids.device,
                                            dtype=torch.float32)
                    vmin = min(out.logits.shape[-1], ref_logits.shape[-1])
                    log_p = F.log_softmax(out.logits[..., :vmin].float(), dim=-1)
                    p_ref = F.softmax(ref_logits[..., :vmin].float(), dim=-1)
                    distill = F.kl_div(log_p, p_ref, reduction="batchmean")

                    lm_w = args_ns.lm_loss_weight * lm
                    recon_w = args_ns.layer_recon_loss_weight * recon
                    distill_w = args_ns.logit_distill_weight * distill
                    loss = lm_w + distill_w + recon_w

                (loss / args_ns.grad_accum).backward()

                if (bi + 1) % args_ns.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, args_ns.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix({
                    "lm": f"{lm.item():.3f}({lm_w.item():.2f})",
                    "rec": f"{recon.item():.3f}({recon_w.item():.2f})",
                    "kd": f"{distill.item():.3f}({distill_w.item():.2f})",
                    "tot": f"{loss.item():.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                del out, log_p, p_ref


            if epoch_callback is not None:
                epoch_callback(epoch + 1, model)
    finally:
        recon_remove()
    model.eval()


def train_post_quant(model, ref_model, train_loader, args_ns, device,
                    epoch_callback=None):
    """
    Drive the post-quant tuning phase that trains only the per-module A and B vectors.

    Parameters:
        model: Student model containing DeltaSVDLinear_QUANT modules.
        ref_model: Frozen fine-tuned teacher used for KL distillation.
        train_loader: Iterable yielding training batches.
        args_ns: Namespace of training hyper-parameters (see _train_loop).
        device: Device used for batches.
        epoch_callback: Optional fn(epoch_num, model) called after each epoch.
    """
    def pred(n):
        """
        Trainable-parameter predicate selecting only the per-module A and B vectors.

        Parameters:
            n: Parameter name from model.named_parameters().

        Returns:
            True if `n` ends with '.A' or '.B'.
        """
        return n.endswith(".A") or n.endswith(".B")
    _train_loop(model, ref_model, train_loader, args_ns, device,
                num_epochs=args_ns.quant_train_epoch,
                trainable_predicate=pred,
                phase_name="step2 post-quant",
                epoch_callback=epoch_callback)
    clear_layer_recon_teachers(model)


@torch.no_grad()
def _export_module_to_packed(mod: DeltaSVDLinear_QUANT) -> dict:
    """
    Export a trained DeltaSVDLinear_QUANT into the packed_v1 8-field dict.

    A: (in_f,) scales V_T's right; B: (out_f,) scales U's left.

    Parameters:
        mod: Trained DeltaSVDLinear_QUANT to serialize.

    Returns:
        Dict with CPU tensors for U_q, U_scale, U_zero, V_T_q, V_T_scale,
        V_T_zero, A, B suitable for io.save_packed.
    """
    return {
        "U_q": mod.U_int.detach().cpu(),
        "U_scale": mod.U_scale.detach().cpu(),
        "U_zero": mod.U_zero.detach().cpu(),
        "V_T_q": mod.V_T_int.detach().cpu(),
        "V_T_scale": mod.V_T_scale.detach().cpu(),
        "V_T_zero": mod.V_T_zero.detach().cpu(),
        "A": mod.A.detach().to(torch.bfloat16).cpu(),
        "B": mod.B.detach().to(torch.bfloat16).cpu(),
    }


class _TuneArgs:
    """
    Minimal namespace consumed by _train_loop's args_ns positional.
    """
    def __init__(self, **kwargs):
        """
        Initialize the namespace by setting each keyword argument as an attribute.

        Parameters:
            kwargs: Arbitrary keyword arguments stored as instance attributes.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)


def merge(finetuned_model: str, base_model: str, delta_path: str,
          save_path: str, device: str = "cuda"):
    """
    Dequantize a packed delta, add it onto the base weights, and save a HF model directory.

    Parameters:
        finetuned_model: HF path/dir whose tokenizer and shell model are loaded.
        base_model: HF path/dir whose dense base weights receive the delta.
        delta_path: Path to a packed_v1 .pt produced by compress / tune.
        save_path: Destination directory for the merged HF model.
        device: Device on which the merged model is materialized.
    """
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model)
    model = AutoModelForCausalLM.from_pretrained(
        finetuned_model, torch_dtype=torch.bfloat16
    ).to(device)
    delta = torch.load(delta_path, map_location="cpu")

    meta = delta.pop("__meta__", None)
    if not (isinstance(meta, dict) and meta.get("format") == "packed_v1"):
        raise ValueError(
            f"merge expects packed_v1 delta (output of compress / tune); "
            f"got meta={meta!r}.")
    group_size = meta["group_size"]

    base_sd = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).state_dict()

    prefixes = sorted({k.rsplit(".", 1)[0] for k in delta})
    merged = {}
    for prefix in tqdm(prefixes, desc="Dequant+merge"):
        packed = {
            "U_q": delta[f"{prefix}.U_q"],
            "U_scale": delta[f"{prefix}.U_scale"],
            "U_zero": delta[f"{prefix}.U_zero"],
            "V_T_q": delta[f"{prefix}.V_T_q"],
            "V_T_scale": delta[f"{prefix}.V_T_scale"],
            "V_T_zero": delta[f"{prefix}.V_T_zero"],
            "A": delta[f"{prefix}.A"],
            "B": delta[f"{prefix}.B"],
        }
        delta_fp = reconstruct_delta(packed, group_size)
        base_key = f"{prefix}.weight"
        base_w = base_sd[base_key].to(torch.float32)
        merged[base_key] = (base_w + delta_fp).to(torch.bfloat16)

    model.load_state_dict(merged, strict=False)
    del base_sd, merged


    gc_cfg = getattr(model, "generation_config", None)
    if gc_cfg is not None:


        gc_cfg.do_sample = True
        for _attr in ("temperature", "top_p", "top_k"):
            if getattr(gc_cfg, _attr, None) is not None:
                setattr(gc_cfg, _attr, None)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


def _compute_dense_lm_weights(delta_path: str, base_model: str
                              ) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Load a packed delta and base weights, returning dense bf16 LM weights and meta.

    For each LM linear, dense = base + dequant(delta).

    Parameters:
        delta_path: Path to the packed_v1 .pt delta file.
        base_model: HF path/dir whose dense base weights are loaded.

    Returns:
        dense: Dict mapping LM weight key -> dense bf16 tensor.
        meta: Metadata dict from the packed delta file.
    """
    delta = torch.load(delta_path, map_location="cpu")
    meta = delta.pop("__meta__", None)
    if not (isinstance(meta, dict) and meta.get("format") == "packed_v1"):
        raise ValueError(f"expected packed_v1 delta; got meta={meta!r}")
    group_size = meta["group_size"]

    base_sd = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).state_dict()

    prefixes = sorted({k.rsplit(".", 1)[0] for k in delta})
    dense: dict[str, torch.Tensor] = {}
    for prefix in tqdm(prefixes, desc="Dequant LM"):
        packed = {
            "U_q": delta[f"{prefix}.U_q"],
            "U_scale": delta[f"{prefix}.U_scale"],
            "U_zero": delta[f"{prefix}.U_zero"],
            "V_T_q": delta[f"{prefix}.V_T_q"],
            "V_T_scale": delta[f"{prefix}.V_T_scale"],
            "V_T_zero": delta[f"{prefix}.V_T_zero"],
            "A": delta[f"{prefix}.A"],
            "B": delta[f"{prefix}.B"],
        }
        delta_fp = reconstruct_delta(packed, group_size)
        base_key = f"{prefix}.weight"
        base_w = base_sd[base_key].to(torch.float32)
        dense[base_key] = (base_w + delta_fp).to(torch.bfloat16)
    del base_sd
    return dense, meta


def merge_llava(template_dir: str, base_model: str,
                delta_path: str, save_path: str):
    """
    LLaVA-specific merge: overlay compressed LM weights onto a LLaVA template directory.

    Writes a directory in liuhaotian (LlavaLlamaForCausalLM) format that is
    ready for `convert_llava_to_hf()`:
      - Aux files (config.json, tokenizer*, etc.) copied from template_dir.
      - Each LLaVA shard rewritten: LM weights (model.layers.*, model.norm,
        embed_tokens, lm_head) overwritten with base + dequant(delta);
        vision_tower / mm_projector keys passed through verbatim.
      - pytorch_model.bin.index.json copied unchanged.

    Parameters:
        template_dir: Path to llava-v1.5-7b (provides LLaVA shards + aux).
        base_model: Path to vicuna-7b-v1.5 (LLaVA's LLM backbone).
        delta_path: Tuned packed-v1 .pt (output of `tune` step).
        save_path: Output liuhaotian-format directory.
    """
    import glob
    import shutil

    print(f"[merge_llava] computing dense LM weights from {delta_path}")
    dense, _meta = _compute_dense_lm_weights(delta_path, base_model)
    print(f"[merge_llava] dense LM keys: {len(dense)}")

    os.makedirs(save_path, exist_ok=True)


    for f in os.listdir(template_dir):
        s = os.path.join(template_dir, f)
        d = os.path.join(save_path, f)
        if os.path.isdir(s) or f.endswith(('.bin', '.safetensors')):
            continue
        if not os.path.exists(d):
            shutil.copy2(s, d)


    shards = sorted(glob.glob(os.path.join(template_dir, 'pytorch_model-*.bin')))
    if not shards:

        shards = sorted(glob.glob(os.path.join(template_dir, '*.safetensors')))
        if not shards:
            raise RuntimeError(f"No LLaVA shards found under {template_dir}")
    print(f"[merge_llava] rewriting {len(shards)} shard(s)")
    overwritten = 0
    for shard in shards:
        if shard.endswith('.safetensors'):
            from safetensors.torch import load_file, save_file
            sd = load_file(shard)
        else:
            sd = torch.load(shard, map_location='cpu', weights_only=False)
        new_sd = {}
        for k, v in sd.items():
            if k in dense:
                new_sd[k] = dense[k].to(v.dtype)
                overwritten += 1
            else:
                new_sd[k] = v
        out = os.path.join(save_path, os.path.basename(shard))
        if shard.endswith('.safetensors'):
            new_sd = {k: (vv.contiguous() if hasattr(vv, "contiguous") else vv)
                      for k, vv in new_sd.items()}
            save_file(new_sd, out)
        else:
            torch.save(new_sd, out)
        print(f"  wrote {out}")
    print(f"[merge_llava] overwritten {overwritten} LM weights "
          f"(of {len(dense)} dense candidates)")


    for f in glob.glob(os.path.join(template_dir, 'pytorch_model.bin.index.json')):
        shutil.copy2(f, save_path)
    for f in glob.glob(os.path.join(template_dir, 'model.safetensors.index.json')):
        shutil.copy2(f, save_path)
