# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash track: decode-shape-gated fused Triton mHC kernels for gfx906.
#
# Background (trace attribution in /data/vllm-gfx906-dsv4/KDA_TAIL_PLAN.md):
# the profiled GLM-5.3 decode step runs the pure-PyTorch `_mhc_pre_fallback`
# / `_mhc_post_fallback` from vllm/model_executor/layers/mhc.py INSIDE the
# FULL cudagraph replay. With hc_sinkhorn_iterations=20 that is ~144 aten
# launches per mhc_pre call x90 calls/step (~13k tiny fp32 elementwise/reduce
# kernels, ~52 ms/step) plus a 1.5 MB fp32 rocBLAS SGEMM x90 (~22 ms/step).
# The pre-existing vllm/model_executor/layers/mhc_triton.py fixes the same
# problem for DSVd4 but (a) is not shape-gated and wedged workers on the
# first real PREFILL after a clean-cache boot (see EXPERIMENTS.md
# 2026-09-11 "mHC-Triton TPM chain diagnosis"), so the whole flag is parked;
# (b) uses BLOCK_K=512/num_warps=8 which put 24x512 fp32 fn tiles in
# registers on a 128-lane gfx906 wavefront budget.
#
# This module:
#   * keeps one program per token, fp32 accumulation in registers, no tl.dot,
#     no PDL/TMA, block dims <= 1024, num_warps=4 (gfx906-safe);
#   * BLOCK_K=256 for the mix GEMM phase (24x256 fp32 tile = 6k regs/program);
#   * fuses the per-layer RMSNorm + cast-to-model-dtype that model.py applies
#     to layer_input (WITH_NORM=1), eliminating the native fp32 RMSNorm chain
#     (~5 kernels + cast per call);
#   * DEFAULTS TO DECODE-SHAPED BATCHES ONLY (num_tokens <=
#     VLLM_GLM53_MHC_FUSED_MAX_TOKENS, default 256) so the un-root-caused
#     prefill hang of the DSV4 triton path cannot be triggered, while decode
#     (the storm) is fully covered.
#
# Env gates:
#   VLLM_GLM53_MHC_FUSED=1            decode-shaped tokens only (default max 256)
#   VLLM_GLM53_MHC_FUSED=2 / "all"    all token counts (ONLY after the prefill
#                                     hang root cause is understood)
#   VLLM_GLM53_MHC_FUSED_MAX_TOKENS   decode-shape threshold (default 256)
# Anything else / unset -> callers must use the torch fallback.
#
# Math replicates _mhc_pre_fallback/_mhc_post_fallback bit-for-bit in op
# order (fp32), plus the vLLM RMSNorm-native formula when norm_weight is
# given. Offsets are int64-safe. Tested shapes: HC=4, H=4096 (GLM-5.3),
# sinkhorn_repeat=20, post_mult=2.0.

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False

_ENV_GATE = "VLLM_GLM53_MHC_FUSED"
_ENV_MAX_TOK = "VLLM_GLM53_MHC_FUSED_MAX_TOKENS"
_MAX_TOKENS_DEFAULT = 256


def _gate_value() -> str:
    return os.environ.get(_ENV_GATE, "0").strip().lower()


def mhc_fused_max_tokens() -> int:
    try:
        return int(os.environ.get(_ENV_MAX_TOK, str(_MAX_TOKENS_DEFAULT)))
    except ValueError:
        return _MAX_TOKENS_DEFAULT


def mhc_fused_enabled(num_tokens: int | None = None) -> bool:
    """True when the fused kernels should run for a call of `num_tokens`.

    num_tokens=None means "no shape info" -> only allowed in 'all' mode.
    """
    if not HAS_TRITON:
        return False
    g = _gate_value()
    if g in ("0", "", "false", "off"):
        return False
    if g in ("2", "all"):
        return True
    # default: decode-only shapes
    if num_tokens is None:
        return False
    return num_tokens <= mhc_fused_max_tokens()


if HAS_TRITON:

    @triton.jit
    def _glm53_mhc_pre_kernel(
        residual_ptr,  # [T, HC*H] residual dtype (flattened streams)
        fn_ptr,  # [HC3, HC*H] fp32
        hc_scale_ptr,  # [3] fp32
        hc_base_ptr,  # [HC3] fp32
        norm_w_ptr,  # [H] any dtype (WITH_NORM only)
        post_mix_ptr,  # [T, HC] fp32 (out)
        comb_mix_ptr,  # [T, HC*HC] fp32 (out)
        layer_input_ptr,  # [T, H] out dtype
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        norm_eps,
        sinkhorn_repeat,
        HC: tl.constexpr,
        H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
        WITH_NORM: tl.constexpr,
    ):
        t = tl.program_id(0)
        HC2: tl.constexpr = HC * HC
        KH: tl.constexpr = HC * H
        t64 = t.to(tl.int64)

        offs_hc = tl.arange(0, HC)
        offs_hc2 = tl.arange(0, HC2)

        # ---------------- Phase A: mixes GEMM + sqrsum, fp32 in-register --
        x_base = residual_ptr + t64 * KH
        acc_pre = tl.zeros((HC, ), dtype=tl.float32)
        acc_post = tl.zeros((HC, ), dtype=tl.float32)
        acc_comb = tl.zeros((HC2, ), dtype=tl.float32)
        acc_sq = 0.0

        for k0 in range(0, KH, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < KH
            x = tl.load(x_base + offs_k, mask=kmask, other=0.0).to(tl.float32)
            acc_sq += tl.sum(x * x, axis=0)

            fn_pre = tl.load(
                fn_ptr + offs_hc[:, None].to(tl.int64) * KH + offs_k[None, :],
                mask=kmask[None, :],
                other=0.0,
            )
            acc_pre += tl.sum(fn_pre * x[None, :], axis=1)

            fn_post = tl.load(
                fn_ptr + (HC + offs_hc)[:, None].to(tl.int64) * KH
                + offs_k[None, :],
                mask=kmask[None, :],
                other=0.0,
            )
            acc_post += tl.sum(fn_post * x[None, :], axis=1)

            fn_comb = tl.load(
                fn_ptr + (2 * HC + offs_hc2)[:, None].to(tl.int64) * KH
                + offs_k[None, :],
                mask=kmask[None, :],
                other=0.0,
            )
            acc_comb += tl.sum(fn_comb * x[None, :], axis=1)

        # ----------- Phase B: rms scale, gates, softmax + Sinkhorn --------
        rms = tl.math.rsqrt(acc_sq / KH + rms_eps)

        s0 = tl.load(hc_scale_ptr + 0)
        s1 = tl.load(hc_scale_ptr + 1)
        s2 = tl.load(hc_scale_ptr + 2)

        pre_mix = tl.sigmoid(acc_pre * rms * s0
                             + tl.load(hc_base_ptr + offs_hc)) + hc_pre_eps

        post_mix = (tl.sigmoid(acc_post * rms * s1
                               + tl.load(hc_base_ptr + HC + offs_hc))
                    * hc_post_mult_value)
        tl.store(post_mix_ptr + t64 * HC + offs_hc, post_mix)

        comb = tl.reshape(
            acc_comb * rms * s2
            + tl.load(hc_base_ptr + 2 * HC + offs_hc2), (HC, HC))

        # row softmax (+eps), then (repeat-1) x {row-norm, col-norm}
        row_max = tl.max(comb, axis=1)
        comb = tl.exp(comb - row_max[:, None])
        comb = comb / tl.sum(comb, axis=1)[:, None] + hc_sinkhorn_eps
        for _ in range(0, sinkhorn_repeat - 1):
            comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_sinkhorn_eps)
            comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_sinkhorn_eps)
        tl.store(comb_mix_ptr + t64 * HC2 + offs_hc2,
                 tl.reshape(comb, (HC2, )))

        # -------- Phase C: layer_input = sum_s pre_mix[s] * r[s, :] -------
        # WITH_NORM: two passes over the row (recompute per block); pass 1
        # gets the sum of squares, pass 2 writes (li * invrms * w) cast to
        # the output/model dtype. WITHOUT_NORM: single pass, raw store.
        res2d = residual_ptr + t64 * KH
        out_ty = layer_input_ptr.dtype.element_ty
        if WITH_NORM:
            sq = 0.0
            for h0 in range(0, H, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                r = tl.load(res2d + offs_hc[:, None].to(tl.int64) * H
                            + offs_h[None, :]).to(tl.float32)
                li = tl.sum(r * pre_mix[:, None], axis=0)
                sq += tl.sum(li * li, axis=0)
            inv = tl.math.rsqrt(sq / H + norm_eps)
            for h0 in range(0, H, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                r = tl.load(res2d + offs_hc[:, None].to(tl.int64) * H
                            + offs_h[None, :]).to(tl.float32)
                li = tl.sum(r * pre_mix[:, None], axis=0)
                w = tl.load(norm_w_ptr + offs_h).to(tl.float32)
                tl.store(layer_input_ptr + t64 * H + offs_h,
                         (li * inv * w).to(out_ty))
        else:
            for h0 in range(0, H, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                r = tl.load(res2d + offs_hc[:, None].to(tl.int64) * H
                            + offs_h[None, :]).to(tl.float32)
                li = tl.sum(r * pre_mix[:, None], axis=0)
                tl.store(layer_input_ptr + t64 * H + offs_h, li.to(out_ty))

    @triton.jit
    def _glm53_mhc_post_kernel(
        comb_ptr,  # [T, HC, HC] fp32 (a[i, o])
        res_ptr,  # [T, HC, H] residual dtype
        post_ptr,  # [T, HC] fp32 (c)
        x_ptr,  # [T, H] x dtype
        out_ptr,  # [T, HC, H] out dtype = residual dtype
        HC: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        t = tl.program_id(0)
        t64 = t.to(tl.int64)
        offs_hc = tl.arange(0, HC)

        # a_t[o, i] = comb[i, o]
        a_t = tl.load(comb_ptr + t64 * HC * HC
                      + offs_hc[None, :].to(tl.int64) * HC + offs_hc[:, None])
        c = tl.load(post_ptr + t64 * HC + offs_hc)  # [HC] fp32

        out_ty = out_ptr.dtype.element_ty
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            x = tl.load(x_ptr + t64 * H + offs_h).to(tl.float32)
            r = tl.load(res_ptr + t64 * HC * H
                        + offs_hc[:, None].to(tl.int64) * H
                        + offs_h[None, :]).to(tl.float32)  # [HC(i), BH]
            # out[o, h] = c[o] * x[h] + sum_i a[i, o] * r[i, h]
            ol = c[:, None] * x[None, :]
            for i in range(HC):
                a_col_i = tl.sum(tl.where(offs_hc[None, :] == i, a_t, 0.0),
                                 axis=1)  # [HC(o)]
                r_i = tl.sum(tl.where(offs_hc[:, None] == i, r, 0.0),
                             axis=0)  # [BH]
                ol += a_col_i[:, None] * r_i[None, :]
            tl.store(out_ptr + t64 * HC * H
                     + offs_hc[:, None].to(tl.int64) * H + offs_h[None, :],
                     ol.to(out_ty))


def _check_common(residual, fn, hc_scale, hc_base):
    if not HAS_TRITON:
        raise RuntimeError("glm53_mhc_fused requires triton")
    assert residual.is_cuda
    assert residual.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32


def mhc_pre_fused(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-5,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mhc_pre (+ optional RMSNorm + cast).

    Same (post_mix, comb_mix, layer_input) contract as
    vllm/model_executor/layers/mhc.py::_mhc_pre_fallback. When
    `norm_weight` is given, layer_input is returned already RMS-normed
    (vLLM native RMSNorm formula, fp32 accum) and cast to `out_dtype`
    (default: residual dtype), which lets the caller skip its
    input_layernorm + `.to(model_dtype)` — that fp32 native-norm chain is
    ~6 aten launches per call on gfx906.
    """
    _check_common(residual, fn, hc_scale, hc_base)

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3, )
    assert hc_base.shape == (hc_mult3, )
    assert hc_mult in (1, 2, 4, 8)  # tl.arange dims must be powers of two
    assert hidden_size % 1024 == 0
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size, )
        if out_dtype is None:
            out_dtype = residual.dtype
    else:
        out_dtype = residual.dtype

    outer_shape = residual.shape[:-2]
    res_flat = residual.contiguous().view(-1, hc_mult, hidden_size)
    num_tokens = res_flat.shape[0]
    device = residual.device

    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32,
                           device=device)
    comb_mix = torch.empty(num_tokens, hc_mult2, dtype=torch.float32,
                           device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=out_dtype,
                              device=device)

    # norm weight ptr may be real or dummy; triton needs a tensor arg
    nw = norm_weight.contiguous() if norm_weight is not None else hc_scale

    _glm53_mhc_pre_kernel[(num_tokens, )](
        res_flat,
        fn.contiguous(),
        hc_scale,
        hc_base,
        nw,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        norm_eps,
        sinkhorn_repeat,
        HC=hc_mult,
        H=hidden_size,
        BLOCK_K=256,
        BLOCK_H=1024,
        WITH_NORM=norm_weight is not None,
        num_warps=4,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def mhc_post_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Fused mhc_post; same contract as mhc.py::_mhc_post_fallback."""
    if not HAS_TRITON:
        raise RuntimeError("glm53_mhc_fused requires triton")

    assert residual.is_cuda
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    assert hc_mult in (1, 2, 4, 8)
    assert hidden_size % 1024 == 0

    outer_shape = residual.shape[:-2]
    res_flat = residual.contiguous().view(-1, hc_mult, hidden_size)
    num_tokens = res_flat.shape[0]

    x_flat = x.contiguous().view(num_tokens, hidden_size)
    post_flat = post_layer_mix.float().contiguous().view(num_tokens, hc_mult)
    comb_flat = (comb_res_mix.float().contiguous().view(
        num_tokens, hc_mult, hc_mult))

    out = torch.empty_like(res_flat)
    _glm53_mhc_post_kernel[(num_tokens, )](
        comb_flat,
        res_flat,
        post_flat,
        x_flat,
        out,
        HC=hc_mult,
        H=hidden_size,
        BLOCK_H=1024,
        num_warps=4,
    )
    return out.view(*outer_shape, hc_mult, hidden_size)
