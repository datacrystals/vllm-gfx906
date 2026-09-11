# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for the DeepSeek-V4 mHC (hyper-connection) blocks.

gfx906 / triton-gfx906 3.5.0 constraints honored here:
- no launch_pdl / PDL kwargs
- no tl.dot (no MMA); all math is plain fp32 loads, broadcast multiplies and
  tl.sum reductions, so there is no input_precision concern at all
- no TMA / warp-specialization / epilogue-fusion APIs
- every block dimension is <= 1024
- one program per token; the 20-iteration Sinkhorn runs entirely in registers

The math replicates `_mhc_pre_fallback` / `_mhc_post_fallback` in mhc.py
exactly (same op order, same fp32 streams):
    pre:   gemm (x @ fn.T) + sqrsum -> rms-norm -> [pre|post|comb] mixes ->
           softmax(+eps) -> (sinkhorn_repeat-1) x {row-div, col-div} ->
           layer_input accumulate in fp32, cast to residual dtype
    post:  out[o, h] = post[o] * x[h] + sum_i comb[i, o] * residual[i, h]

This is a numerics-preserving launch-count optimization: ~90 eager launches
per mhc_pre call down to 1 kernel, and ~6 per mhc_post call down to 1.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only environments (tests import lazily)
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _mhc_pre_fused_kernel(
        residual_ptr,  # [T, HC*H] fp32 (flattened streams)
        fn_ptr,  # [HC3, HC*H] fp32
        hc_scale_ptr,  # [3] fp32
        hc_base_ptr,  # [HC3] fp32
        post_mix_ptr,  # [T, HC] fp32 (out)
        comb_mix_ptr,  # [T, HC*HC] fp32 (out)
        layer_input_ptr,  # [T, H] out, dtype = residual dtype
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        HC: tl.constexpr,
        H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        t = tl.program_id(0)
        HC2: tl.constexpr = HC * HC
        KH: tl.constexpr = HC * H  # flattened stream length, e.g. 16384

        # ------------------------------------------------------------------
        # Phase A: gemm_out_mul = x @ fn.T and gemm_out_sqrsum = (x*x).sum(),
        # accumulated in fp32 over BLOCK_K-sized k-slices, split by output
        # row group: pre rows [0, HC), post rows [HC, 2*HC),
        # comb rows [2*HC, 2*HC + HC*HC).
        # ------------------------------------------------------------------
        x_base = residual_ptr + t.to(tl.int64) * KH
        fn_base = fn_ptr

        offs4 = tl.arange(0, HC)
        offs16 = tl.arange(0, HC2)

        acc_pre = tl.zeros((HC, ), dtype=tl.float32)
        acc_post = tl.zeros((HC, ), dtype=tl.float32)
        acc_comb = tl.zeros((HC2, ), dtype=tl.float32)
        acc_sq = 0.0

        for k0 in range(0, KH, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < KH
            x = tl.load(x_base + offs_k, mask=kmask, other=0.0)  # fp32
            acc_sq += tl.sum(x * x, axis=0)

            fn_pre = tl.load(
                fn_base + offs4[:, None] * KH + offs_k[None, :],
                mask=kmask[None, :],
                other=0.0,
            )
            acc_pre += tl.sum(fn_pre * x[None, :], axis=1)

            fn_post = tl.load(
                fn_base + (HC + offs4)[:, None] * KH + offs_k[None, :],
                mask=kmask[None, :],
                other=0.0,
            )
            acc_post += tl.sum(fn_post * x[None, :], axis=1)

            fn_comb = tl.load(
                fn_base + (2 * HC + offs16)[:, None] * KH + offs_k[None, :],
                mask=kmask[None, :],
                other=0.0,
            )
            acc_comb += tl.sum(fn_comb * x[None, :], axis=1)

        # ------------------------------------------------------------------
        # Phase B: rms-normalized mixes, sigmoid gates, softmax + Sinkhorn.
        # ------------------------------------------------------------------
        rms = tl.math.rsqrt(acc_sq / KH + rms_eps)

        s0 = tl.load(hc_scale_ptr + 0)
        s1 = tl.load(hc_scale_ptr + 1)
        s2 = tl.load(hc_scale_ptr + 2)

        # pre_mix = sigmoid(mixes[:HC] * s0 + base[:HC]) + hc_pre_eps
        mix_pre = acc_pre * rms
        base_pre = tl.load(hc_base_ptr + offs4)
        pre_mix = tl.sigmoid(mix_pre * s0 + base_pre) + hc_pre_eps

        # post_mix = sigmoid(mixes[HC:2HC] * s1 + base[HC:2HC]) * post_mult
        mix_post = acc_post * rms
        base_post = tl.load(hc_base_ptr + HC + offs4)
        post_mix = tl.sigmoid(mix_post * s1 + base_post) * hc_post_mult_value
        tl.store(post_mix_ptr + t.to(tl.int64) * HC + offs4, post_mix)

        # comb = mixes[2HC:] * s2 + base[2HC:]  ->  [HC, HC]
        mix_comb = acc_comb * rms
        base_comb = tl.load(hc_base_ptr + 2 * HC + offs16)
        comb = tl.reshape(mix_comb * s2 + base_comb, (HC, HC))

        # comb = softmax(comb, dim=-1) + eps   (row softmax)
        row_max = tl.max(comb, axis=1)
        comb = tl.exp(comb - row_max[:, None])
        row_sum = tl.sum(comb, axis=1)
        comb = comb / row_sum[:, None] + hc_sinkhorn_eps

        # repeat-1 iterations of {row normalize, col normalize}
        for _ in range(0, sinkhorn_repeat - 1):
            row_sum = tl.sum(comb, axis=1)
            comb = comb / (row_sum[:, None] + hc_sinkhorn_eps)
            col_sum = tl.sum(comb, axis=0)
            comb = comb / (col_sum[None, :] + hc_sinkhorn_eps)

        tl.store(
            comb_mix_ptr + t.to(tl.int64) * HC2 + offs16,
            tl.reshape(comb, (HC2, )),
        )

        # ------------------------------------------------------------------
        # Phase C: layer_input = sum_s pre_mix[s] * residual[s, :]
        # (fp32 accumulate, cast to residual dtype on store)
        # ------------------------------------------------------------------
        res2d = residual_ptr + t.to(tl.int64) * KH  # viewed [HC, H] rows
        out_ty = layer_input_ptr.dtype.element_ty
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            r = tl.load(res2d + offs4[:, None] * H + offs_h[None, :])
            ol = tl.sum(r * pre_mix[:, None], axis=0)
            tl.store(layer_input_ptr + t.to(tl.int64) * H + offs_h,
                     ol.to(out_ty))

    @triton.jit
    def _mhc_post_fused_kernel(
        comb_ptr,  # [T, HC, HC] fp32 (a)
        res_ptr,  # [T, HC, H] residual dtype
        post_ptr,  # [T, HC] fp32 (c)
        x_ptr,  # [T, H] x dtype (model dtype)
        out_ptr,  # [T, HC, H] out, dtype = residual dtype
        HC: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        t = tl.program_id(0)
        offs4 = tl.arange(0, HC)

        # a = comb (row-major [i, o]), c = post
        a = tl.load(comb_ptr + t.to(tl.int64) * HC * HC +
                    offs4[:, None] * HC + offs4[None, :])  # [HC(i), HC(o)]
        c = tl.load(post_ptr + t.to(tl.int64) * HC + offs4)  # [HC]

        aT = tl.trans(a)  # [HC(o), HC(i)]
        out_ty = out_ptr.dtype.element_ty
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            # x_i: x block, broadcast over output streams o
            x = tl.load(x_ptr + t.to(tl.int64) * H + offs_h).to(tl.float32)
            # r: residual streams [HC(i), BLOCK_H]
            r = tl.load(res_ptr + t.to(tl.int64) * HC * H +
                        offs4[:, None] * H + offs_h[None, :]).to(tl.float32)
            # out[o, h] = c[o] * x[h] + sum_i a[i, o] * r[i, h]
            ol = c[:, None] * x[None, :] + \
                tl.sum(r[None, :, :] * aT[:, :, None], axis=1)
            tl.store(
                out_ptr + t.to(tl.int64) * HC * H + offs4[:, None] * H +
                offs_h[None, :],
                ol.to(out_ty),
            )


def mhc_pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton fused equivalent of vllm.model_executor.layers.mhc._mhc_pre_fallback.

    Returns (post_mix, comb_mix, layer_input) with the same shapes/dtypes.
    """
    if not HAS_TRITON:
        raise RuntimeError("mhc_pre_triton requires triton (not importable)")

    assert residual.is_cuda
    assert residual.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3, )
    assert hc_base.shape == (hc_mult3, )
    # tl.arange dimensions must be powers of two
    assert hc_mult in (1, 2, 4, 8)
    assert hidden_size % 1024 == 0

    outer_shape = residual.shape[:-2]
    residual_flat = residual.contiguous().view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x_flat = residual_flat.view(num_tokens, hc_mult * hidden_size).float()
    fn_c = fn.contiguous()

    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32,
                           device=residual.device)
    comb_mix = torch.empty(num_tokens, hc_mult2, dtype=torch.float32,
                           device=residual.device)
    layer_input = torch.empty(num_tokens, hidden_size,
                              dtype=residual.dtype, device=residual.device)

    _mhc_pre_fused_kernel[(num_tokens, )](
        x_flat,
        fn_c,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        HC=hc_mult,
        H=hidden_size,
        BLOCK_K=512,
        BLOCK_H=1024,
        num_warps=8,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Triton fused equivalent of vllm.model_executor.layers.mhc._mhc_post_fallback."""
    if not HAS_TRITON:
        raise RuntimeError("mhc_post_triton requires triton (not importable)")

    assert residual.is_cuda
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    assert hc_mult in (1, 2, 4, 8)
    assert hidden_size % 1024 == 0

    outer_shape = residual.shape[:-2]
    num_tokens = residual.contiguous().view(-1, hc_mult, hidden_size).shape[0]

    res_flat = residual.contiguous().view(num_tokens, hc_mult, hidden_size)
    x_flat = x.contiguous().view(num_tokens, hidden_size)
    post_flat = post_layer_mix.float().contiguous().view(num_tokens, hc_mult)
    comb_flat = comb_res_mix.float().contiguous().view(num_tokens, hc_mult,
                                                       hc_mult)

    out = torch.empty_like(res_flat)
    _mhc_post_fused_kernel[(num_tokens, )](
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
