# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash track: fused MoE router kernel for gfx906
# (sigmoid scoring + noaux_tc e_score_correction_bias + top-k selection),
# specialized to the GLM-5.3 regime: num_expert_group == 1, E <= 1024,
# topk <= 32.
#
# Replaces the torch path in
# vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py
# ::grouped_topk (lines ~113-162), which at decode is ~14 aten launches per
# MoE layer (sigmoid, add-bias, 3 x torch.topk [bitonic/radix blocks],
# zeros_like, scatter_, masked_fill, gather, sum, div, mul, casts). In the
# GLM-5.3 rank0 decode-step trace that is ~600 launches and ~3.5-4 ms/step
# (aten::sbtopk::gatherTopK x126 = 2.44 ms alone is all MoE-router topk).
# With n_group == 1 the group stage is degenerate (group_mask is all-ones),
# so the whole routing collapses to: s=sigmoid(logits); sel=s+bias;
# iterative top-k on sel; weights=renorm(gather(s)); weights*=scale.
# One program per token, everything register-resident, no tl.dot. fp32
# selection with an explicit fp16 rounding of the sigmoid to reproduce the
# reference's dtype flow (torch computes sigmoid on fp16 in fp32 opmath and
# rounds back to fp16 before the fp32 bias add).
#
# Numerics: weights match the torch reference to within one fp16 rounding
# (renorm recip divider computed fp32 vs torch's fp16-out division; bench
# gate is allclose atol=rtol=1e-3). Tie-breaking: lowest expert index wins
# equal scores; torch.topk's CUDA ties are unspecified-order anyway. The
# bench compares sorted id sets per token and reports tie mismatches.
#
# Env gate: VLLM_GLM53_MOE_ROUTER_FUSED=1 (default off). Wire point:
# grouped_topk_router.py grouped_topk(), before the torch fallback branch.

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

_ENV = "VLLM_GLM53_MOE_ROUTER_FUSED"


def router_fused_enabled() -> bool:
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "on")


if HAS_TRITON:

    @triton.jit
    def _glm53_sigmoid_topk_kernel(
        logits_ptr,  # [T, E] fp16/bf16/fp32 (contiguous, row stride = stride_l_t)
        bias_ptr,  # [E] fp32 (contiguous)
        w_ptr,  # [T, K] fp32 (out)
        idx_ptr,  # [T, K] int32 (out)
        stride_l_t: tl.int64,
        stride_w_t: tl.int64,
        stride_i_t: tl.int64,
        scale,
        E,
        E_BLOCK: tl.constexpr,
        K: tl.constexpr,
        RENORM: tl.constexpr,
        ROUND_DT: tl.constexpr,  # 0 = no rounding (fp32 logits), 1 = fp16, 2 = bf16
    ):
        t = tl.program_id(0)
        t64 = t.to(tl.int64)
        offs = tl.arange(0, E_BLOCK)
        emask = offs < E

        x = tl.load(logits_ptr + t64 * stride_l_t + offs,
                    mask=emask, other=float("-inf")).to(tl.float32)
        # torch.opmath flow: sigmoid is computed in fp32 and stored back in
        # the input dtype (fp16/bf16 logits round; fp32 logits don't).
        s = tl.sigmoid(x)
        if ROUND_DT == 1:
            s = s.to(tl.float16).to(tl.float32)
        elif ROUND_DT == 2:
            s = s.to(tl.bfloat16).to(tl.float32)
        bias = tl.load(bias_ptr + offs, mask=emask, other=0.0)
        sel = tl.where(emask, s + bias, float("-inf"))

        wsum = 0.0
        wv = tl.zeros((K,), dtype=tl.float32)
        # K constexpr -> python-level unrolled selection loop
        for k in tl.static_range(K):
            m = tl.max(sel, axis=0)
            # lowest index among ties
            idx = tl.min(tl.where(sel == m, offs, E_BLOCK), axis=0)
            tl.store(idx_ptr + t64 * stride_i_t + k, idx.to(tl.int32))
            w_k = tl.sum(tl.where(offs == idx, s, 0.0), axis=0)
            wsum += w_k
            # stash pre-norm weight in lane k of a scratch vector
            wv = tl.where(tl.arange(0, K) == k, w_k, wv)
            sel = tl.where(offs == idx, float("-inf"), sel)

        if RENORM:
            # torch: topk_weights.sum(-1) stays in the logits dtype (fp32
            # accumulation, rounded back on store) — round the sum to the
            # logits dtype before dividing, then round the quotient.
            if ROUND_DT == 1:
                wsum = wsum.to(tl.float16).to(tl.float32)
            elif ROUND_DT == 2:
                wsum = wsum.to(tl.bfloat16).to(tl.float32)
            wv = wv / wsum
            if ROUND_DT == 1:
                wv = wv.to(tl.float16).to(tl.float32)
            elif ROUND_DT == 2:
                wv = wv.to(tl.bfloat16).to(tl.float32)
        wv = wv * scale
        # torch keeps weights in the logits dtype through renorm+scale and
        # only then casts to fp32 — replicate the per-dtype rounding (after
        # renorm AND after scale, matching where torch materializes).
        if ROUND_DT == 1:
            wv = wv.to(tl.float16).to(tl.float32)
        elif ROUND_DT == 2:
            wv = wv.to(tl.bfloat16).to(tl.float32)
        tl.store(w_ptr + t64 * stride_w_t + tl.arange(0, K), wv)


def router_fused_supported(
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
) -> bool:
    return (
        HAS_TRITON
        and gating_output.is_cuda
        and gating_output.dim() == 2
        and scoring_func == "sigmoid"
        and e_score_correction_bias is not None
        and e_score_correction_bias.numel() == gating_output.shape[1]
        and num_expert_group == 1
        and (topk_group in (None, 0, 1))  # degenerate single group
        and gating_output.shape[1] <= 1024
        and 0 < topk <= 32
        and (topk & (topk - 1)) == 0  # power of two for tl.arange(K)
    )


def fused_sigmoid_topk(
    gating_output: torch.Tensor,  # [T, E]
    e_score_correction_bias: torch.Tensor,  # [E] fp32
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused GLM-5.3 router: returns (topk_weights fp32, topk_ids int32)."""
    if not router_fused_supported(
        gating_output, e_score_correction_bias, topk, 1, 1, "sigmoid"
    ):
        raise ValueError("unsupported shapes for glm53 fused router")

    # Make contiguity/stripes unambiguous BEFORE asking for data_ptr/strides:
    # passing the *original* tensor's stride with the *contiguous copy's*
    # pointer (the previous code) walks out of the copy's allocation whenever
    # the caller hands in a strided view — the likely mechanism of the
    # capture-time memory fault (one rank's capture-time logits were a
    # strided view of a padded buffer).
    logits = gating_output.contiguous()
    bias = e_score_correction_bias.float().contiguous()
    num_tokens, num_experts = logits.shape
    e_block = 1 << (num_experts - 1).bit_length()
    device = logits.device
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32,
                               device=device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32,
                           device=device)

    if logits.dtype == torch.float16:
        round_dt = 1
    elif logits.dtype == torch.bfloat16:
        round_dt = 2
    else:
        round_dt = 0  # fp32 (or anything else): no intermediate rounding

    _glm53_sigmoid_topk_kernel[(num_tokens, )](
        logits,
        bias,
        topk_weights,
        topk_ids,
        logits.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        routed_scaling_factor,
        num_experts,
        E_BLOCK=e_block,
        K=topk,
        RENORM=renormalize,
        ROUND_DT=round_dt,
        num_warps=4,
    )
    return topk_weights, topk_ids


def grouped_topk_maybe_fused(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Drop-in for grouped_topk()'s torch body. Returns None when the fused
    path does not apply (caller falls through to the reference)."""
    if not router_fused_enabled():
        return None
    if not router_fused_supported(
        gating_output,
        e_score_correction_bias,
        topk,
        num_expert_group,
        topk_group,
        scoring_func,
    ):
        return None
    return fused_sigmoid_topk(
        gating_output,
        e_score_correction_bias.float(),
        topk,
        renormalize,
        routed_scaling_factor,
    )
