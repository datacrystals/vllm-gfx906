# SPDX-License-Identifier: Apache-2.0
# Track P1 — fused single-kernel GDN decode recurrence for gfx906.
#
# Replaces the ~30-op pure-torch eager recurrence of
# `gdn_gfx906_fallback.packed_decode_ref` / `sigmoid_gating_update_ref`
# with ONE triton kernel launch per layer per decode step.
#
# Math is identical to the reference (fp32 throughout, same op order):
#   q,k  : l2norm(x) = x * rsqrt(sum(x^2) + 1e-6); q *= scale
#   g,beta: x = a + dt_bias
#           sp = log1p(exp(min(x,20))) if x <= 20 else x
#           g = -exp(A_log) * sp ; beta = sigmoid(b)
#   S    : S <- S * exp(g); dv = (v - S @ k) * beta; S <- S + dv (x) k
#   o    : o = S^T? no — o[v] = sum_k S[v,k] * q[k]     (matches einsum "bhvk,bhk->bhv")
# Slot idx <= 0 (NULL sentinel): output row zeroed, state untouched
# (the reference writes slot-0 readback of itself via where(); equivalent).
#
# Differences vs the reference are only fp reduction-order (tl.sum tree vs
# BLAS accumulate) and libm rounding — expected |diff| ~1e-6 fp32.

import torch
import triton
import triton.language as tl

try:
    _log1p = tl.math.log1p  # noqa
    _HAS_LOG1P = True
except AttributeError:
    _HAS_LOG1P = False

EPS_L2NORM = tl.constexpr(1e-6)


@triton.jit
def _gdn_decode_step_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    state_ptr, out_ptr, indices_ptr,
    scale,
    s_q_b, s_k_b, s_v_b, s_a_b, s_b_b,
    s_state_slot, s_out_b, s_idx,
    H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, REP: tl.constexpr,
    THRESH: tl.constexpr, L2NORM: tl.constexpr, HAS_LOG1P: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // REP

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)  # host guarantees K is a power of 2
    mask_v = o_v < V

    p_out = out_ptr + i_n * s_out_b + i_hv * V + o_v
    idx = tl.load(indices_ptr + i_n * s_idx).to(tl.int64)
    if idx <= 0:
        zero = tl.zeros([BV], dtype=tl.float32)
        tl.store(p_out, zero.to(p_out.dtype.element_ty), mask=mask_v)
        return

    b_q = tl.load(q_ptr + i_n * s_q_b + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(k_ptr + i_n * s_k_b + i_h * K + o_k).to(tl.float32)
    if L2NORM:
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + EPS_L2NORM)
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + EPS_L2NORM)
    b_q = b_q * scale

    a_val = tl.load(a_ptr + i_n * s_a_b + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * s_b_b + i_hv).to(tl.float32)
    A_log = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    x = a_val + dt_bias
    xc = tl.minimum(x, THRESH)
    if HAS_LOG1P:
        sp = tl.where(x <= THRESH, tl.math.log1p(tl.exp(xc)), x)
    else:
        sp = tl.where(x <= THRESH, tl.log(1.0 + tl.exp(xc)), x)
    decay = tl.exp(-tl.exp(A_log) * sp)
    beta = tl.sigmoid(b_val)

    b_v = tl.load(v_ptr + i_n * s_v_b + i_hv * V + o_v,
                  mask=mask_v, other=0.0).to(tl.float32)

    p_s = (state_ptr + idx * s_state_slot + i_hv * V * K
           + o_v[:, None] * K + o_k[None, :])
    S = tl.load(p_s, mask=mask_v[:, None], other=0.0).to(tl.float32)
    S = S * decay
    dv = b_v - tl.sum(S * b_k[None, :], 1)
    dv = dv * beta
    S = S + dv[:, None] * b_k[None, :]
    b_o = tl.sum(S * b_q[None, :], 1)

    tl.store(p_s, S.to(p_s.dtype.element_ty), mask=mask_v[:, None])
    tl.store(p_out, b_o.to(p_out.dtype.element_ty), mask=mask_v)


# tuned on MI50 (gfx906) in bench_gdn_decode.py; fixed to keep launches
# graph-capturable and autotune-free on the hot path
def _pick_config(HV: int, V: int):
    bv = 64 if V % 64 == 0 else triton.next_power_of_2(V)
    return bv, 4  # (BV, num_warps)


def _launch(q, k, v, a, b, A_log, dt_bias, scale, state, out, indices,
            use_l2norm: bool, threshold: float):
    B, HV, V = v.shape
    H = q.shape[1]
    K = q.shape[2]
    if K != triton.next_power_of_2(K):
        raise ValueError(f"head_k_dim must be power of 2, got {K}")
    BV, num_warps = _pick_config(HV, V)
    grid = (triton.cdiv(V, BV), B * HV)
    _gdn_decode_step_kernel[grid](
        q, k, v, a, b, A_log, dt_bias,
        state, out, indices,
        scale,
        q.stride(0), k.stride(0), v.stride(0), a.stride(0), b.stride(0),
        state.stride(0), out.stride(0), indices.stride(0),
        H=H, HV=HV, K=K, V=V, BV=BV, REP=HV // H,
        THRESH=threshold, L2NORM=use_l2norm, HAS_LOG1P=_HAS_LOG1P,
        num_warps=num_warps, num_stages=1,
    )


def packed_decode_fused(*, mixed_qkv: torch.Tensor, a: torch.Tensor,
                        b: torch.Tensor, A_log: torch.Tensor,
                        dt_bias: torch.Tensor, scale: float,
                        initial_state: torch.Tensor, out: torch.Tensor,
                        ssm_state_indices: torch.Tensor,
                        use_qk_l2norm_in_kernel: bool = True):
    """Drop-in for fused_recurrent_gated_delta_rule_packed_decode()."""
    B, qkv_dim = mixed_qkv.shape
    HV, V, K = initial_state.shape[-3:]
    H = (qkv_dim - HV * V) // (2 * K)
    q = mixed_qkv[:, :H * K].view(B, H, K)
    k = mixed_qkv[:, H * K:2 * H * K].view(B, H, K)
    v = mixed_qkv[:, 2 * H * K:].view(B, HV, V)
    _launch(q, k, v, a, b, A_log, dt_bias, scale,
            initial_state, out.view(B, HV, V), ssm_state_indices,
            use_qk_l2norm_in_kernel, SOFTPLUS_THRESHOLD_CONST)
    return out, initial_state


SOFTPLUS_THRESHOLD_CONST = 20.0


def sigmoid_gating_update_fused(*, A_log: torch.Tensor, a: torch.Tensor,
                                b: torch.Tensor, dt_bias: torch.Tensor,
                                q: torch.Tensor, k: torch.Tensor,
                                v: torch.Tensor,
                                beta: float = 1.0, threshold: float = 20.0,
                                scale: float = None,
                                initial_state: torch.Tensor = None,
                                inplace_final_state: bool = True,
                                cu_seqlens: torch.Tensor | None = None,
                                ssm_state_indices: torch.Tensor | None = None,
                                num_accepted_tokens: torch.Tensor | None = None,
                                use_qk_l2norm_in_kernel: bool = False,
                                is_kda: bool = False, **_ignored):
    """Drop-in for fused_sigmoid_gating_delta_rule_update (1 tok/seq, no CPU
    sync — the reference fallback's cu_seqlens.tolist() is removed: the call
    site asserts all lens == 1 already, and the kernel needs no lengths)."""
    if num_accepted_tokens is not None:
        raise NotImplementedError("GDN gfx906 fused: spec decode not supported")
    if is_kda:
        raise NotImplementedError("GDN gfx906 fused: IS_KDA branch unused")
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    assert B == 1
    if scale is None:
        scale = K ** -0.5
    idx = ssm_state_indices
    if idx.ndim > 1:
        idx = idx[:, 0]
    out = torch.empty(T, 1, HV, V, dtype=initial_state.dtype,
                      device=q.device)
    _launch(q[0], k[0], v[0], a, b, A_log, dt_bias, scale,
            initial_state, out.view(T, HV, V), idx,
            use_qk_l2norm_in_kernel, threshold)
    o = out.view(1, T, HV, V)
    return o, initial_state if inplace_final_state else initial_state.clone()


def install_fused_decode() -> bool:
    """Swap the eager-torch GDN decode fallback for the fused triton kernel.
    Called from gdn_gfx906_fallback.install_gfx906_gdn_fallback() when
    VLLM_GDN_GFX906_FUSED_DECODE=1."""
    from vllm.model_executor.layers.mamba import gdn_linear_attn as gla
    gla.fused_recurrent_gated_delta_rule_packed_decode = packed_decode_fused
    gla.fused_sigmoid_gating_delta_rule_update = sigmoid_gating_update_fused
    print("[GDN gfx906] fused decode kernel ACTIVE "
          "(1 launch/layer instead of ~30 eager ops)", flush=True)
    return True
