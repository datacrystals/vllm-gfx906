# SPDX-License-Identifier: Apache-2.0
# gfx906 MoE wna16 PREFILL path: dequantize int4 experts to fp16 and run
# Tensile GEMMs, instead of the Triton fused_moe_kernel_gptq_awq.
#
# Why: the Triton MoE kernel tops out at ~3.5 TFLOP/s on gfx906 (tl.dot
# lowers to FMA; no MFMA on this arch) and is 89% of prefill GPU time.
# Tensile fp16 GEMM sustains ~7 TFLOP/s at per-expert M>=256, so
# dequant-to-fp16 + per-expert torch.mm is ~1.5x end-to-end prefill once
# per-step M is large (LPT>=2048 makes single-request chunks 2048 tokens).
#
# Numerics: dequant in fp32 -> cast fp16 (one rounding, more accurate than
# the legacy hsub2/hmul2 fp16 chain); GEMMs with reduced-precision-reduction
# disabled (fp32 accumulate); topk combine in fp32 via index_add (same
# contract as the moe_wna16 fp32-atomic quality fix).
#
# Engaged only when VLLM_GFX906_MOE_DQMM=1 (default off) and
# tokens*topk/local_experts >= threshold; caller falls back to fused_experts.

import os

import torch
import triton
import triton.language as tl

_ENABLED = None
_MIN_PER_EXPERT = 256


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("VLLM_GFX906_MOE_DQMM", "0") == "1"
        if _ENABLED:
            # fp32 accumulation in fp16 GEMMs (same motivation as the
            # fp32-atomic moe_wna16 quality fix).
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return _ENABLED


def min_per_expert() -> int:
    return _MIN_PER_EXPERT


@triton.jit
def _dq4_kernel(
    w_ptr, z_ptr, s_ptr, o_ptr,
    K, g,
    stride_wn, stride_zn, stride_sn, stride_on,
    HAS_ZP: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    # w: [N, K/2] uint8 (even k low nibble); z: [N/2, K/g] uint8 (even n low
    # nibble); s: [N, K/g] fp16; o: [N, K] fp16. All 2D views of one expert.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = pid_k * BK + tl.arange(0, BK)

    w = tl.load(w_ptr + rn[:, None] * stride_wn + (rk[None, :] // 2))
    w32 = w.to(tl.int32)
    nib = (rk[None, :] % 2) * 4
    wq = (w32 >> nib) & 0xF

    kg = rk // g
    s = tl.load(s_ptr + rn[:, None] * stride_sn + kg[None, :]).to(tl.float32)
    if HAS_ZP:
        z = tl.load(z_ptr + (rn[:, None] // 2) * stride_zn + kg[None, :])
        z32 = z.to(tl.int32)
        znib = (rn[:, None] % 2) * 4
        zq = ((z32 >> znib) & 0xF).to(tl.float32)
    else:
        zq = tl.full((BN, 1), 8.0, tl.float32)

    o = (wq.to(tl.float32) - zq) * s
    tl.store(o_ptr + rn[:, None] * stride_on + rk[None, :], o.to(tl.float16))


def _dequant(w, s, z, out, group_size):
    # w [N, K/2] uint8 view; s [N, K/g] fp16; z [N/2, K/g] uint8 or None
    N = w.shape[0]
    K = out.shape[1]
    BN, BK = 64, 256
    grid = (N // BN, K // BK)
    _dq4_kernel[grid](
        w, z if z is not None else w, s, out,
        K, group_size,
        w.stride(0), z.stride(0) if z is not None else 0,
        s.stride(0), out.stride(0),
        HAS_ZP=z is not None, BN=BN, BK=BK,
    )
    return out


# Reusable workspace buffers. With PYTORCH_ALLOC_CONF expandable_segments:False
# (standing config — ES:True wedged the GPUs in A/B), per-call allocation of
# the big workspaces (47 MoE layers x per-chunk) fragments VRAM until the HSA
# runtime dies with "OUT_OF_RESOURCES, free mem: 8 MB". Grow-only module-level
# cache: allocated once at max size, reused every layer/chunk thereafter.
_BUF_CACHE: dict = {}


def _buf(name: str, shape, dtype, dev) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    key = (name, dev)
    buf = _BUF_CACHE.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, device=dev)
        _BUF_CACHE[key] = buf
    return buf[:numel].view(shape)


def moe_dqmm_forward(
    x: torch.Tensor,               # [M, K] fp16 hidden states (this step)
    w13_w: torch.Tensor,           # [E, 2*inter, K/2] uint8
    w13_s: torch.Tensor,           # [E, 2*inter, K/g] fp16
    w13_z: torch.Tensor | None,    # [E, inter? no: (2*inter)/2, K/g] uint8
    w2_w: torch.Tensor,            # [E, hidden, inter/2] uint8
    w2_s: torch.Tensor,            # [E, hidden, inter/g] fp16
    w2_z: torch.Tensor | None,
    topk_weights: torch.Tensor,    # [M, topk] fp32/fp16
    topk_ids: torch.Tensor,        # [M, topk] int32/int64 (global expert ids)
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    group_size: int,
    activation: str,
) -> torch.Tensor:
    M, K = x.shape
    E, N13 = w13_w.shape[0], w13_w.shape[1]
    inter = N13 // 2
    N2 = w2_w.shape[1]
    topk = topk_ids.shape[1]
    dev = x.device

    flat_ids = topk_ids.reshape(-1)
    local = expert_map[flat_ids] if expert_map is not None else flat_ids
    valid = local >= 0
    pair_idx = torch.nonzero(valid, as_tuple=True)[0]
    local_v = local[valid]
    order = torch.argsort(local_v, stable=True)
    pair_sorted = pair_idx[order]
    e_sorted = local_v[order]
    counts = torch.bincount(e_sorted, minlength=E).cpu().tolist()  # 1 sync

    topk_w_flat = topk_weights.reshape(-1)
    out32 = _buf("out32", (M, N2), torch.float32, dev)
    out32.zero_()

    W13 = _buf("W13", (N13, K), torch.float16, dev)
    W2 = _buf("W2", (N2, inter), torch.float16, dev)

    # layer.activation is MoEActivation (enum) in vllm; normalize to str
    act = str(getattr(activation, "name", activation)).lower()

    # Bound transient GEMM/accum buffers: hot experts can draw >10k pairs
    # (a fp32 [cnt, hidden] buffer per slice otherwise OOMs at steady state).
    PAIR_CAP = 4096

    start = 0
    for e in range(E):
        cnt = counts[e]
        if cnt == 0:
            continue
        seg_all = pair_sorted[start:start + cnt]
        start += cnt

        _dequant(w13_w[e], w13_s[e],
                 w13_z[e] if w13_z is not None else None, W13, group_size)
        _dequant(w2_w[e], w2_s[e],
                 w2_z[e] if w2_z is not None else None, W2, group_size)

        for s0 in range(0, cnt, PAIR_CAP):
            seg = seg_all[s0:s0 + PAIR_CAP]
            tok = seg // topk
            A = x.index_select(0, tok)                 # [n, K] fp16
            w_e = topk_w_flat[seg].to(torch.float32)
            if apply_router_weight_on_input:
                A = (A.to(torch.float32) * w_e[:, None]).to(x.dtype)

            H = A @ W13.t()                            # [n, 2*inter]
            del A
            if act == "silu":
                H = torch.nn.functional.silu(H[:, :inter]) * H[:, inter:]
            elif act == "gelu":
                H = torch.nn.functional.gelu(H[:, :inter]) * H[:, inter:]
            else:
                raise NotImplementedError(f"dqmm activation {activation}")

            O = H @ W2.t()                             # [n, hidden]
            del H
            O32 = O.to(torch.float32)
            del O
            if not apply_router_weight_on_input:
                O32 *= w_e[:, None]
            out32.index_add_(0, tok, O32)
            del O32

    return out32.to(x.dtype)
