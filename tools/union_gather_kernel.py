#!/usr/bin/env python3
"""union_gather_kernel.py — sparse-MLA prefill, union-of-indices gather.

GLM53-PREFILL step 2 (PREFILL_KERNEL_PLAN.md). Adjacent prefill tokens
retrieve heavily-overlapping topk rows (measured 52% pairwise); gathering
each UNIQUE row once per query tile cuts random DRAM reads 2-5x.

Math (matches reference_mla_sparse_prefill):
    S[t,h,j] = scale * dot(q[t,h], K[idx[t,j]])
    P[t,h,:] = softmax(S[t,h,:])
    out[t,h] = sum_j P[t,h,j] * V[idx[t,j], :D_V]     # V = first D_V dims

Kernel: flash-style online softmax with per-row indirection (INV) into the
union buffer KU. V-dims split across grid dim 2 so per-program accumulators
stay small. Offline self-test vs pure-torch reference below.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

D_QK = 576
D_V = 512


def union_prep(indices: torch.Tensor):
    T, TOPK = indices.shape
    u_idx, inv = torch.unique(indices.reshape(-1), return_inverse=True)
    inv = inv.view(T, TOPK).to(torch.int32).contiguous()
    return u_idx.contiguous(), inv, int(u_idx.numel())


@triton.jit
def _union_attn_kernel(
    Q, KU, INV, OUT, LSE,
    stride_q_t, stride_q_h,
    stride_ku_u,
    stride_inv_t,
    stride_out_t, stride_out_h,
    scale,
    T, TOPK,
    D_QK_R: tl.constexpr, D_QK_C: tl.constexpr,
    D_V_R: tl.constexpr,
    BLOCK_M: tl.constexpr, TILE_K: tl.constexpr,
    KD_CH: tl.constexpr, VD_CH: tl.constexpr,
):
    """Grid: (ceil(T/BLOCK_M), H, ceil(D_V/VD_CH))."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)

    toff = pid_t * BLOCK_M
    rows = toff + tl.arange(0, BLOCK_M)
    t_mask = rows < T
    dv0 = pid_v * VD_CH
    dv = dv0 + tl.arange(0, VD_CH)
    dv_ok = dv < D_V_R

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, VD_CH], dtype=tl.float32)

    for j0 in range(0, TOPK, TILE_K):
        j = j0 + tl.arange(0, TILE_K)
        inv = tl.load(INV + rows[:, None] * stride_inv_t + j[None, :],
                      mask=t_mask[:, None] & (j[None, :] < TOPK), other=-1)
        valid = inv >= 0

        # S[bm, tk] = scale * sum_d Q[bm,d] * KU[inv[bm,tk], d]
        S = tl.zeros([BLOCK_M, TILE_K], dtype=tl.float32)
        for d0 in range(0, D_QK_C, KD_CH):
            d = d0 + tl.arange(0, KD_CH)
            d_ok = d < D_QK_R
            qd = tl.load(Q + rows[:, None] * stride_q_t
                         + pid_h * stride_q_h + d[None, :],
                         mask=t_mask[:, None] & d_ok[None, :], other=0.0)
            k = tl.load(KU + inv[:, :, None] * stride_ku_u + d[None, None, :],
                        mask=valid[:, :, None] & d_ok[None, None, :],
                        other=0.0)
            S += tl.sum(qd.to(tl.float32)[:, None, :] * k.to(tl.float32),
                        axis=2)
        S *= scale
        S = tl.where(valid, S, float("-inf"))

        # online softmax (flash-attention style)
        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_safe)
        P = tl.exp(S - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(P, axis=1)
        acc = acc * alpha[:, None]

        # acc[bm, vch] += sum_tk P[bm,tk] * KU[inv[bm,tk], dv]
        v = tl.load(KU + inv[:, :, None] * stride_ku_u + dv[None, None, :],
                    mask=valid[:, :, None] & dv_ok[None, None, :], other=0.0)
        acc += tl.sum(P.to(tl.float32)[:, :, None] * v.to(tl.float32), axis=1)
        m_i = m_safe

    tl.store(OUT + rows[:, None] * stride_out_t
             + pid_h * stride_out_h + dv[None, :],
             (acc / l_i[:, None]).to(OUT.dtype.element_ty),
             mask=t_mask[:, None] & dv_ok[None, :])
    tl.store(LSE + pid_h * T + rows,
             (m_i + tl.log(l_i)).to(LSE.dtype.element_ty), mask=t_mask)


def union_prefill_fwd(q, kv, indices, scale=None):
    """q [T,H,D_QK] fp16/bf16; kv [S_KV,D_QK]; indices [T,TOPK].
    Returns out [T,H,D_V] in q.dtype."""
    T, H, _ = q.shape
    TOPK = indices.shape[1]
    if scale is None:
        scale = D_QK ** -0.5
    u_idx, inv, U = union_prep(indices)
    ku = kv.index_select(0, u_idx).contiguous()   # the ONE gather
    out = torch.empty(T, H, D_V, device=q.device, dtype=q.dtype)
    lse = torch.empty(H, T, device=q.device, dtype=torch.float32)
    BLOCK_M, VD_CH = 32, 64
    grid = (triton.cdiv(T, BLOCK_M), H, triton.cdiv(D_V, VD_CH))
    _union_attn_kernel[grid](
        q, ku, inv, out, lse,
        q.stride(0), q.stride(1),
        ku.stride(0),
        inv.stride(0),
        out.stride(0), out.stride(1),
        scale,
        T, TOPK,
        D_QK_R=D_QK, D_QK_C=triton.next_power_of_2(D_QK),
        D_V_R=D_V,
        BLOCK_M=BLOCK_M, TILE_K=32, KD_CH=64, VD_CH=VD_CH,
    )
    return out


def _reference(q, kv, indices, scale):
    g = kv.index_select(0, indices.reshape(-1)).view(*indices.shape, -1)
    s = torch.einsum("thd,tjd->thj", q.float(), g.float()) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("thj,tjd->thd", p, g[..., :D_V].float()).to(q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, H, TOPK, S_KV = 8, 2, 64, 512
    q = torch.randn(T, H, D_QK, device=dev, dtype=torch.float16)
    kv = torch.randn(S_KV, D_QK, device=dev, dtype=torch.float16)
    idx = torch.randint(0, S_KV, (T, TOPK), device=dev)
    scale = D_QK ** -0.5
    ref = _reference(q, kv, idx, scale)
    out = union_prefill_fwd(q, kv, idx, scale)
    err = (out.float() - ref.float()).abs().max().item()
    rel = err / (ref.float().abs().max().item() + 1e-6)
    print(f"self-test: max_abs_err={err:.4f} rel={rel:.3f}")
    assert rel < 0.05, "union kernel deviates from reference"
    print("UNION-KERNEL SELF-TEST PASS")
