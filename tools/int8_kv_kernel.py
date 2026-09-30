#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# GLM53-INT8KV: int8 MLA latent KV cache — standalone micro-kernels + self-test.
#
# Mission: halve the dominant MLA-latent term of the 256-token block cache
# (11 sparse layers x 576 dims x 256 tokens x fp16) by quantizing the
# per-token latent row to symmetric int8 with per-token (or per-token-group)
# fp32 scales.  The indexer kpool keys stay fp16 (retrieval is discrete-risk)
# — nothing in this file touches the indexer caches.
#
# Row layout (primary: G=1 or G=2, ROW=584 bytes per token per layer):
#     [0   :576] int8 data  : dims [0:512) kv_c latent | [512:576) k_pe rope
#     [576 :580] fp32 scale : group 0
#     [580 :584] fp32 scale : group 1 (zero pad when G=1)
#   ROW = align8(576 + 4*G).  This matches the fp8_ds_mla precedent
#   (kv_cache_interface.py: "448B NoPE + 128B RoPE + 8B scale = 584B/token")
#   where per-token scales are stored INLINE in the raw block bytes, so
#   block-granular copy/swap/prefix-caching carry them automatically.
#
# Consumers mirrored here (production file:line in INT8_KV_PLAN.md):
#   (a) int8 quantize-on-write   — concat_and_cache_mla replacement
#   (b) dequant-on-gather prefill — reference_mla_sparse_prefill replacement
#   (c) int8 triton decode kernels — dequant-load variant of
#       _mla_sparse_vec_kernel and _deepgemm_fp16_paged_mqa_logits_stage1
#       (tl.load int8 -> .to(tl.float32) * scale)
#
# Self-test:  python3 tools/int8_kv_kernel.py [--gpu]
#   CPU always (torch A/B, rel < 2e-2 gate).  --gpu also launches triton.
#
# No vLLM imports.  torch required; triton optional.

from __future__ import annotations

import argparse
import sys

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - triton absent on CPU-only boxes
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Layout constants.  d_qk = kv_lora_rank(512) + qk_rope_head_dim(64).
# ---------------------------------------------------------------------------
D_LATENT = 512
D_ROPE = 64
D_QK = D_LATENT + D_ROPE  # 576
D_V = D_LATENT            # 512 — value side is the latent part
ALIGN = 8
DATA_BYTES = D_QK  # 576 int8 data bytes per row
GS_SINGLE = (D_QK,)
GS_SPLIT = (D_LATENT, D_ROPE)


def _align8(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def num_groups(group_sizes: tuple[int, ...]) -> int:
    return len(group_sizes)


def row_bytes(group_sizes: tuple[int, ...]) -> int:
    """Storage bytes per token row: int8 data + fp32 scales, 8B-aligned."""
    return _align8(DATA_BYTES + 4 * num_groups(group_sizes))


# ---------------------------------------------------------------------------
# Dtype discipline.  The int32-scatter burn (2026-09-29: scatter_add with
# int32 indices corrupts silently on ROCm) taught us to force dtypes at every
# boundary instead of trusting callers.
# ---------------------------------------------------------------------------
def _check_flt(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name}: expected float16/bfloat16/float32, got {x.dtype}")
    return x


def _as_i64(idx: torch.Tensor, name: str) -> torch.Tensor:
    """int64 gather/scatter indices — NEVER trust incoming int32 (ROCm)."""
    return idx.to(torch.int64)


def _as_i8(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dtype != torch.int8:
        raise TypeError(f"{name}: expected torch.int8, got {x.dtype}")
    return x


def _as_f32(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dtype != torch.float32:
        raise TypeError(f"{name}: expected torch.float32, got {x.dtype}")
    return x


# ---------------------------------------------------------------------------
# (0) core quantization: symmetric int8, per-token or per-token-group fp32
# ---------------------------------------------------------------------------
def quantize_rows_int8(
    x: torch.Tensor,
    group_sizes: tuple[int, ...] = GS_SINGLE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group int8 quantization of [..., 576] float rows.

    q[..., d]     = clamp(round(x[..., d] / s[g]), -127, 127)
    s[g]          = amax(|x[group g]|) / 127   (1.0 for an all-zero group)

    Returns:
      q     : [..., 576] torch.int8
      scale : [..., G]   torch.float32
    """
    _check_flt(x, "x")
    assert x.shape[-1] == D_QK, f"x.shape[-1]={x.shape[-1]} != {D_QK}"
    assert sum(group_sizes) == D_QK, f"group_sizes {group_sizes} must sum to {D_QK}"
    xf = x.to(torch.float32)
    q = torch.empty_like(xf, dtype=torch.int8)
    scales = []
    off = 0
    for g in group_sizes:
        chunk = xf[..., off : off + g]
        amax = chunk.abs().amax(dim=-1, keepdim=True)  # [..., 1]
        scale = amax / 127.0
        # All-zero group -> scale 1.0 so dequant is exactly 0 and no NaN.
        scale = torch.where(amax > 0, scale, torch.ones_like(scale))
        q[..., off : off + g] = torch.clamp(
            torch.round(chunk / scale), -127.0, 127.0
        ).to(torch.int8)
        scales.append(scale.to(torch.float32))
        off += g
    return q, torch.cat(scales, dim=-1)  # [..., G] float32


def dequantize_rows_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_sizes: tuple[int, ...] = GS_SINGLE,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """x_hat[..., d] = float(q[..., d]) * scale[..., group(d)]"""
    _as_i8(q, "q")
    _as_f32(scale, "scale")
    assert q.shape[-1] == D_QK, q.shape
    G = num_groups(group_sizes)
    assert scale.shape[-1] == G, f"scale.shape[-1]={scale.shape[-1]} != G={G}"
    assert scale.shape[:-1] == q.shape[:-1], (scale.shape, q.shape)
    out = torch.empty(q.shape[:-1] + (D_QK,), dtype=torch.float32, device=q.device)
    off = 0
    for gi, g in enumerate(group_sizes):
        s = scale[..., gi : gi + 1]  # [..., 1]
        out[..., off : off + g] = q[..., off : off + g].to(torch.float32) * s
        off += g
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# Packed inline layout: raw [..., ROW] int8; scale bytes at row offset 576.
# ---------------------------------------------------------------------------
def pack_rows_int8(
    q: torch.Tensor, scale: torch.Tensor, group_sizes: tuple[int, ...] = GS_SINGLE
) -> torch.Tensor:
    """Pack (q int8 [...,576], scale f32 [...,G]) into raw int8 [..., ROW]."""
    _as_i8(q, "q")
    _as_f32(scale, "scale")
    ROW = row_bytes(group_sizes)
    G = num_groups(group_sizes)
    lead = q.shape[:-1]
    assert q.shape[-1] == D_QK, q.shape
    assert scale.shape == lead + (G,), (scale.shape, lead, G)
    raw = torch.zeros(lead + (ROW,), dtype=torch.int8, device=q.device)
    raw[..., :DATA_BYTES] = q
    scale_bytes = scale.contiguous().view(torch.uint8).reshape(lead + (4 * G,))
    raw[..., DATA_BYTES : DATA_BYTES + 4 * G] = scale_bytes.to(torch.int8)
    return raw


def unpack_rows_int8(
    raw: torch.Tensor, group_sizes: tuple[int, ...] = GS_SINGLE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_rows_int8 (bit-exact, including scales)."""
    _as_i8(raw, "raw")
    ROW = row_bytes(group_sizes)
    G = num_groups(group_sizes)
    assert raw.shape[-1] == ROW, f"raw.shape[-1]={raw.shape[-1]} != ROW={ROW}"
    lead = raw.shape[:-1]
    q = raw[..., :DATA_BYTES].contiguous()
    scale_bytes = raw[..., DATA_BYTES : DATA_BYTES + 4 * G].contiguous()
    scale = scale_bytes.view(torch.uint8).reshape(lead + (4 * G,)).view(torch.float32)
    return q, scale.reshape(lead + (G,)).contiguous()


def views_from_packed(
    raw: torch.Tensor, group_sizes: tuple[int, ...] = GS_SINGLE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-copy (data, scale) views over a packed raw cache tensor.

    Mirrors vllm/v1/attention/backends/triton_attn.py::_ensure_scale_caches:
    the scale is carved from inline row padding via a strided f32 storage
    view.  raw: contiguous int8 [..., ROW] (e.g. [num_blocks, block_size, ROW]
    or flattened [rows, ROW]).
    """
    _as_i8(raw, "raw")
    ROW = row_bytes(group_sizes)
    G = num_groups(group_sizes)
    assert raw.is_contiguous(), "views_from_packed requires contiguous raw"
    assert raw.shape[-1] == ROW, (raw.shape, ROW)
    assert DATA_BYTES % 4 == 0 and ROW % 4 == 0
    data = raw[..., :DATA_BYTES]
    lead = raw.shape[:-1]
    prod_lead = 1
    for s in lead:
        prod_lead *= s
    # f32 view: first scale sits at byte 576 within each row.  set_() wants
    # the storage_offset in elements of the VIEW dtype (float32), so the byte
    # offset must be divided by 4 — an int8-element offset here is the classic
    # silent-corruption footgun.
    scale_byte_off = raw.storage_offset() + DATA_BYTES  # int8 elems == bytes
    assert scale_byte_off % 4 == 0
    base_f32 = torch.tensor([], dtype=torch.float32, device=raw.device).set_(
        raw.untyped_storage(),
        storage_offset=scale_byte_off // 4,
        size=(prod_lead, G),
        stride=(ROW // 4, 1),
    )
    scale = base_f32.reshape(lead + (G,))
    return data, scale  # strided view, deliberately no copy


# ---------------------------------------------------------------------------
# (a) quantize-on-write: int8 replacement for ops.concat_and_cache_mla
#     production call site: vllm/v1/attention/backend.py:990/910
#       (SparseMLAAttentionImpl.do_kv_cache_update -> ops.concat_and_cache_mla)
#     C++ reference semantics: csrc/cache_kernels.cu concat_and_cache_mla_kernel
# ---------------------------------------------------------------------------
def int8_concat_and_cache_mla_torch(
    kv_c: torch.Tensor,          # [T, 512] float — kv_c_normed
    k_pe: torch.Tensor,          # [T, 64]  float — rope dims (already roped)
    kv_cache: torch.Tensor,      # [..., ROW] int8 packed (e.g. [blocks, bs, ROW])
    slot_mapping: torch.Tensor,  # [T] int32/int64; -1 = padded, must not write
    group_sizes: tuple[int, ...] = GS_SINGLE,
) -> None:
    """Int8 quantize-on-write.  Row content order [kv_c | k_pe] matches the
    C++ concat_and_cache_mla_kernel exactly; slot_mapping < 0 is a no-op."""
    _check_flt(kv_c, "kv_c")
    _check_flt(k_pe, "k_pe")
    _as_i8(kv_cache, "kv_cache")
    T = kv_c.shape[0]
    assert kv_c.shape == (T, D_LATENT), kv_c.shape
    assert k_pe.shape == (T, D_ROPE), k_pe.shape
    ROW = row_bytes(group_sizes)
    assert kv_cache.shape[-1] == ROW, kv_cache.shape
    assert kv_cache.is_contiguous()
    slots = _as_i64(slot_mapping, "slot_mapping").reshape(-1)
    assert slots.shape[0] == T, (slots.shape, T)

    rows = torch.cat([kv_c, k_pe], dim=-1)  # [T, 576] — same concat as C++
    q, scale = quantize_rows_int8(rows, group_sizes)
    packed = pack_rows_int8(q, scale, group_sizes)  # [T, ROW] int8

    valid = slots >= 0
    flat = kv_cache.view(-1, ROW)
    tgt = slots[valid]
    if tgt.numel():
        assert int(tgt.max()) < flat.shape[0], "slot out of range"
    flat[tgt] = packed[valid.to(torch.bool)]  # int64 index scatter (forced)


int8_concat_and_cache_mla = int8_concat_and_cache_mla_torch


# ---------------------------------------------------------------------------
# (b) dequant-on-gather prefill matmul
#     production call site: reference_mla_sparse_prefill
#       vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py:678-728
#       (gather at :707, matmuls at :710-723)
#     The union-GEMM variant union_gather_prefill (:624-676) gathers at :664
#     and gets the identical dequant-on-gather treatment.
# ---------------------------------------------------------------------------
def _mla_sparse_attn_core(
    q: torch.Tensor,        # [s_q, h_q, 576]
    kv_rows: torch.Tensor,  # [rows, 576] already in final (dequantized) dtype
    indices: torch.Tensor,  # [s_q, topk] int64
    sm_scale: float,
    d_v: int,
    chunk: int = 512,
) -> torch.Tensor:
    """Shared attention core — identical math for the fp16 baseline and the
    int8 path so the A/B isolates quantization error only."""
    s_q, h_q, d_qk = q.shape
    assert d_qk == D_QK, d_qk
    s_kv = kv_rows.shape[0]
    topk = indices.shape[1]
    out = torch.empty(s_q, h_q, d_v, device=q.device, dtype=kv_rows.dtype)
    for start in range(0, s_q, chunk):
        end = min(start + chunk, s_q)
        idx_chunk = indices[start:end]  # [cs, topk]
        invalid = (idx_chunk < 0) | (idx_chunk >= s_kv)
        idx_safe = idx_chunk.masked_fill(invalid, 0)
        gathered = kv_rows.index_select(0, idx_safe.reshape(-1)).reshape(
            end - start, topk, d_qk
        )
        if kv_rows.dtype == torch.float32:
            p = q[start:end].float() @ gathered.transpose(1, 2)
        else:
            p = (q[start:end] @ gathered.transpose(1, 2)).float()
        p.masked_fill_(invalid.unsqueeze(1), float("-inf"))
        p = p * sm_scale
        lse = torch.logsumexp(p, dim=-1)                 # [cs, h_q]
        s_for_o = torch.exp(p - lse.unsqueeze(-1))       # softmax already
        out[start:end] = (
            s_for_o.to(kv_rows.dtype) @ gathered[..., :d_v]
        ).to(out.dtype)
    return out


def fp16_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,       # [rows, 576] float16
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = D_V,
    chunk: int = 512,
) -> torch.Tensor:
    """fp16 baseline (mirrors reference_mla_sparse_prefill)."""
    _check_flt(kv, "kv")
    return _mla_sparse_attn_core(q, kv, indices, sm_scale, d_v, chunk)


def int8_mla_sparse_prefill(
    q: torch.Tensor,
    kv_data: torch.Tensor,   # [rows, 576] int8
    kv_scale: torch.Tensor,  # [rows, G] float32
    indices: torch.Tensor,   # [s_q, topk] any int
    sm_scale: float,
    d_v: int = D_V,
    group_sizes: tuple[int, ...] = GS_SINGLE,
    chunk: int = 512,
) -> torch.Tensor:
    """Dequant-on-gather prefill: gathered.int8 -> * scale -> matmul.

    Only the gathered chunk is dequantized (never the full cache), matching
    the production memory profile on MI50.
    """
    _as_i8(kv_data, "kv_data")
    _as_f32(kv_scale, "kv_scale")
    s_q, h_q, d_qk = q.shape
    assert d_qk == D_QK
    if indices.dim() == 3:
        indices = indices[:, 0, :]
    idx2 = _as_i64(indices, "indices").reshape(s_q, -1)
    out = torch.empty(s_q, h_q, d_v, device=q.device, dtype=q.dtype)
    s_kv = kv_data.shape[0]
    topk = idx2.shape[1]
    for start in range(0, s_q, chunk):
        end = min(start + chunk, s_q)
        idx_chunk = idx2[start:end]
        invalid = (idx_chunk < 0) | (idx_chunk >= s_kv)
        idx_safe = idx_chunk.masked_fill(invalid, 0).reshape(-1)
        gq = kv_data.index_select(0, idx_safe)   # [cs*topk, 576] int8
        gs = kv_scale.index_select(0, idx_safe)  # [cs*topk, G] f32
        g_rows = dequantize_rows_int8(gq, gs, group_sizes, out_dtype=q.dtype)
        g_rows = g_rows.reshape(end - start, topk, D_QK)
        if q.dtype == torch.float32:
            p = q[start:end] @ g_rows.transpose(1, 2)
        else:
            p = (q[start:end] @ g_rows.transpose(1, 2)).float()
        p.masked_fill_(invalid.unsqueeze(1), float("-inf"))
        p = p * sm_scale
        lse = torch.logsumexp(p, dim=-1)
        s_for_o = torch.exp(p - lse.unsqueeze(-1))
        out[start:end] = (s_for_o.to(q.dtype) @ g_rows[..., :d_v]).to(out.dtype)
    return out


# ---------------------------------------------------------------------------
# (c) decode kernels with int8 dequant-load
#     production call sites:
#       _mla_sparse_vec_kernel             (backends)  rocm_aiter_mla_sparse.py:405
#       _deepgemm_fp16_paged_mqa_logits_stage1 (ops)   rocm_aiter_mla_sparse.py:225
#     NOTE: the paged-MQA-logits family scores the INDEXER kpool cache, which
#     stays fp16 by design (retrieval is discrete-risk).  Its int8 variant is
#     provided here as the dequant-load pattern / A-B tool only; see plan.
# ---------------------------------------------------------------------------


def int8_sparse_vec_decode_torch(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = D_V,
    group_sizes: tuple[int, ...] = GS_SINGLE,
) -> torch.Tensor:
    """Torch reference of the int8 sparse-vec decode kernel (softmax over the
    topk set per token/head, PV over the latent part)."""
    return int8_mla_sparse_prefill(
        q, kv_data, kv_scale, indices, sm_scale, d_v, group_sizes, chunk=1 << 30
    )


def int8_paged_mqa_logits_torch(
    q: torch.Tensor,             # [B, next_n, H, D] float
    kv_raw: torch.Tensor,        # [num_blocks, block_size, ROW] int8 packed
    weights: torch.Tensor,       # [B*next_n, H] float32
    context_lens: torch.Tensor,  # [B] int32
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    max_model_len: int,
    group_sizes: tuple[int, ...] = GS_SINGLE,
) -> torch.Tensor:
    """fp8_paged_mqa_logits_torch semantics on an int8 packed cache:
    logits[b*n, pos] = sum_h w * relu(q . k_dequant) causal, -inf elsewhere."""
    _as_i8(kv_raw, "kv_raw")
    _as_f32(weights, "weights")
    B, next_n, H, dim = q.shape
    num_block, block_size, ROW = kv_raw.shape
    assert dim == D_QK, dim
    k_data, k_scale = views_from_packed(kv_raw, group_sizes)
    # Flat [rows, ...] views so the block-table gather is a single index_select
    # over global token rows (phys*block_size + t), same as production.
    k_data = k_data.reshape(num_block * block_size, dim)
    k_scale = k_scale.reshape(num_block * block_size, -1)
    logits = torch.full(
        [B * next_n, max_model_len], float("-inf"), device=q.device,
        dtype=torch.float32,
    )
    cl = context_lens.tolist()
    qf = q.float()
    for i in range(B):
        context_len = int(cl[i])
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        w_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        nb = (context_len + block_size - 1) // block_size
        rows_idx: list[int] = []
        for blk in range(nb):
            phys = int(block_tables[i, blk].item())
            rows_idx.extend(phys * block_size + t for t in range(block_size))
        rows_t = _as_i64(torch.tensor(rows_idx, device=q.device), "rows")
        gq = k_data.index_select(0, rows_t)   # [total, 576] int8
        gs = k_scale.index_select(0, rows_t)  # [total, G] f32
        kx = dequantize_rows_int8(gq, gs, group_sizes)  # [total, 576] f32
        total = kx.shape[0]
        k_offs = torch.arange(total, device=q.device)
        for n in range(next_n):
            s = qf[i, n] @ kx.transpose(0, 1)  # [H, total]
            mask = (k_offs < context_len) & (k_offs <= q_offsets[n])
            s = torch.where(mask[None, :], s, torch.full_like(s, float("-inf")))
            s = torch.relu(s) * w_slice[:, n : n + 1]  # [H,1] broadcast over total
            s = s.sum(dim=0)  # [total]
            logits[i * next_n + n, :total] = torch.where(
                k_offs <= q_offsets[n], s, torch.full_like(s, float("-inf"))
            )
    return logits


def _paged_logits_float_ref(
    q: torch.Tensor,
    kv_rows: torch.Tensor,  # [rows, 576] float32
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Unquantized reference with identical semantics."""
    B, next_n, H, _ = q.shape
    logits = torch.full(
        [B * next_n, max_model_len], float("-inf"), dtype=torch.float32
    )
    qf = q.float()
    cl = context_lens.tolist()
    for i in range(B):
        context_len = int(cl[i])
        q_offsets = torch.arange(context_len - next_n, context_len)
        w_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        nb = (context_len + block_size - 1) // block_size
        rows: list[int] = []
        for blk in range(nb):
            phys = int(block_tables[i, blk])
            rows.extend(phys * block_size + t for t in range(block_size))
        kx = kv_rows[torch.tensor(rows, dtype=torch.int64)]
        total = kx.shape[0]
        k_offs = torch.arange(total)
        for n in range(next_n):
            s = qf[i, n] @ kx.transpose(0, 1)
            mask = (k_offs < context_len) & (k_offs <= q_offsets[n])
            s = torch.where(mask[None, :], s, torch.full_like(s, float("-inf")))
            s = torch.relu(s) * w_slice[:, n : n + 1]
            s = s.sum(dim=0)
            logits[i * next_n + n, :total] = torch.where(
                k_offs <= q_offsets[n], s, torch.full_like(s, float("-inf"))
            )
    return logits


if HAS_TRITON:

    @triton.jit
    def _int8_sparse_vec_kernel(
        output_ptr, query_ptr, kv_data_ptr, kv_scale_ptr, topk_indices_ptr,
        stride_out_sq: tl.int64, stride_out_hq: tl.int64,
        stride_q_sq: tl.int64, stride_q_hq: tl.int64,
        stride_kv: tl.int64, stride_scale: tl.int64,
        stride_idx_sq: tl.int64,
        scale,                       # sm_scale
        s_kv: tl.int32,
        D_QK_C: tl.constexpr,
        D_V_C: tl.constexpr,
        TOPK: tl.constexpr,
        D_SCORE_CHUNK: tl.constexpr,
        D_V_CHUNK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        TILE_K: tl.constexpr,
        SPLIT: tl.constexpr,         # 0 => one per-token scale; else split dim
        G: tl.constexpr,
    ):
        """_mla_sparse_vec_kernel with int8 dequant-load:
        tl.load(int8) -> .to(tl.float32) * scale[row(,group)] -> dot."""
        pid_token = tl.program_id(0)
        pid_hb = tl.program_id(1)
        pid_dv = tl.program_id(2)

        head_start = pid_hb * BLOCK_M
        dv_start = pid_dv * D_V_CHUNK

        offs_m = tl.arange(0, BLOCK_M)
        offs_dk = tl.arange(0, D_SCORE_CHUNK)
        offs_dv = tl.arange(0, D_V_CHUNK)
        offs_tk = tl.arange(0, TILE_K)

        q_base = query_ptr + pid_token * stride_q_sq
        idx_base = topk_indices_ptr + pid_token * stride_idx_sq

        M_val = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        L_val = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D_V_CHUNK], dtype=tl.float32)

        for t_start in range(0, TOPK, TILE_K):
            t_mask = (t_start + offs_tk) < TOPK
            kv_pos = tl.load(idx_base + t_start + offs_tk, mask=t_mask, other=0)
            valid = t_mask & (kv_pos >= 0) & (kv_pos < s_kv)
            kv_pos = tl.where(valid, kv_pos, 0)

            S = tl.zeros([BLOCK_M, TILE_K], dtype=tl.float32)

            for d_start in range(0, D_QK_C, D_SCORE_CHUNK):
                d_offs = d_start + offs_dk
                q_ptrs = (q_base
                          + (head_start + offs_m[:, None]) * stride_q_hq
                          + d_offs[None, :])
                Q_d = tl.load(q_ptrs)

                k_i8 = tl.load(
                    kv_data_ptr
                    + kv_pos[None, :] * stride_kv
                    + d_offs[:, None],
                    mask=valid[None, :], other=0,
                )  # [D_SCORE_CHUNK, TILE_K] int8
                # d_start is a Python int here (constexpr unroll) so the
                # group index is compile-time.
                if SPLIT > 0 and d_start >= SPLIT:
                    g_idx = 1
                else:
                    g_idx = 0
                k_scale = tl.load(
                    kv_scale_ptr + kv_pos * stride_scale + g_idx
                )  # [TILE_K] f32
                K_d = k_i8.to(tl.float32) * k_scale[None, :]

                S = tl.dot(Q_d, K_d.to(Q_d.dtype), acc=S)

            S *= scale
            S = tl.where(valid[None, :], S, float("-inf"))

            m_j = tl.max(S, axis=1)
            m_new = tl.maximum(M_val, m_j)
            m_new = tl.where(m_new > float("-inf"), m_new, 0.0)

            alpha = tl.exp(M_val - m_new)
            P = tl.exp(S - m_new[:, None])
            l_j = tl.sum(P, axis=1)

            acc = acc * alpha[:, None]
            L_val = L_val * alpha + l_j
            M_val = m_new

            v_i8 = tl.load(
                kv_data_ptr
                + kv_pos[:, None] * stride_kv
                + (dv_start + offs_dv)[None, :],
                mask=valid[:, None], other=0,
            )  # [TILE_K, D_V_CHUNK] int8
            # d_v <= D_LATENT always (asserted by the launcher), so the value
            # side lives entirely in group 0 for both G=1 and the (512,64)
            # split.
            v_scale = tl.load(kv_scale_ptr + kv_pos * stride_scale)
            V = v_i8.to(tl.float32) * v_scale[:, None]

            acc = tl.dot(P.to(V.dtype), V, acc=acc)

        acc = acc / L_val[:, None]
        out_ptrs = (output_ptr
                    + pid_token * stride_out_sq
                    + (head_start + offs_m[:, None]) * stride_out_hq
                    + (dv_start + offs_dv)[None, :])
        tl.store(out_ptrs, acc.to(output_ptr.type.element_ty))

    @triton.jit
    def _int8_paged_mqa_logits_kernel(
        batch_size, next_n,
        Q_buffer,
        stride_q_batch: tl.int64, stride_q_next_n: tl.int64,
        stride_q_heads: tl.int64, stride_q_dim: tl.int64,
        kv_data_ptr,                 # int8 rows
        kv_scale_ptr,                # f32 scales
        stride_kv_blk: tl.int64, stride_kv_tok: tl.int64,
        stride_scale_blk: tl.int64, stride_scale_tok: tl.int64,
        context_len_ptr,
        block_table,
        weights,
        stride_w_row: tl.int64,
        Out_buffer,
        stride_out_row: tl.int64,
        max_num_blocks,
        NUM_HEADS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CHUNK_K: tl.constexpr,
        SPLIT: tl.constexpr,
        G: tl.constexpr,
    ):
        """_deepgemm_fp16_paged_mqa_logits_stage1 with int8 dequant-load
        (tl.load int8 -> .to(tl.float32) * scale).  Same grid contract:
        (max_num_blocks, batch_size*next_n)."""
        pid_kv_block = tl.program_id(0)
        pid_bn = tl.program_id(1)

        pid_batch = pid_bn // next_n
        pid_next_n = pid_bn % next_n

        context_length = tl.load(context_len_ptr + pid_batch)
        num_kv_blocks = tl.cdiv(context_length, CHUNK_K)
        if pid_kv_block >= num_kv_blocks:
            return

        context_idx = pid_kv_block * CHUNK_K
        physical_block_id = tl.load(
            block_table + pid_batch * max_num_blocks + pid_kv_block
        )

        kv_offsets = tl.arange(0, CHUNK_K)
        mask_kv = (context_idx + kv_offsets) < context_length
        causal_mask = (
            (context_idx + kv_offsets) <= (context_length - next_n + pid_next_n)
        )
        combined_mask = mask_kv & causal_mask

        acc = tl.zeros([CHUNK_K], dtype=tl.float32)
        d_range = tl.arange(0, BLOCK_D)

        q_base = (Q_buffer
                  + pid_batch * stride_q_batch
                  + pid_next_n * stride_q_next_n)
        w_base = weights + (pid_batch * next_n + pid_next_n) * stride_w_row

        for hg_start in range(0, NUM_HEADS, 8):
            s0 = tl.zeros([CHUNK_K], dtype=tl.float32)
            s1 = tl.zeros([CHUNK_K], dtype=tl.float32)
            s2 = tl.zeros([CHUNK_K], dtype=tl.float32)
            s3 = tl.zeros([CHUNK_K], dtype=tl.float32)
            s4 = tl.zeros([CHUNK_K], dtype=tl.float32)
            s5 = tl.zeros([CHUNK_K], dtype=tl.float32)
            s6 = tl.zeros([CHUNK_K], dtype=tl.float32)
            s7 = tl.zeros([CHUNK_K], dtype=tl.float32)

            for d_start in range(0, HEAD_DIM, BLOCK_D):
                d_offs = d_start + d_range

                kv_i8 = tl.load(
                    kv_data_ptr
                    + physical_block_id * stride_kv_blk
                    + kv_offsets[:, None] * stride_kv_tok
                    + d_offs[None, :],
                    mask=mask_kv[:, None], other=0,
                )  # [CHUNK_K, BLOCK_D] int8
                if SPLIT > 0 and d_start >= SPLIT:
                    g_idx = 1
                else:
                    g_idx = 0
                kv_scale = tl.load(
                    kv_scale_ptr
                    + physical_block_id * stride_scale_blk
                    + kv_offsets * stride_scale_tok
                    + g_idx,
                    mask=mask_kv, other=1.0,
                )  # [CHUNK_K] f32
                kv_tile = kv_i8.to(tl.float32) * kv_scale[:, None]

                q0 = tl.load(q_base + (hg_start + 0) * stride_q_heads + d_offs)
                q1 = tl.load(q_base + (hg_start + 1) * stride_q_heads + d_offs)
                q2 = tl.load(q_base + (hg_start + 2) * stride_q_heads + d_offs)
                q3 = tl.load(q_base + (hg_start + 3) * stride_q_heads + d_offs)
                q4 = tl.load(q_base + (hg_start + 4) * stride_q_heads + d_offs)
                q5 = tl.load(q_base + (hg_start + 5) * stride_q_heads + d_offs)
                q6 = tl.load(q_base + (hg_start + 6) * stride_q_heads + d_offs)
                q7 = tl.load(q_base + (hg_start + 7) * stride_q_heads + d_offs)

                s0 += tl.sum(kv_tile * q0[None, :], axis=1)
                s1 += tl.sum(kv_tile * q1[None, :], axis=1)
                s2 += tl.sum(kv_tile * q2[None, :], axis=1)
                s3 += tl.sum(kv_tile * q3[None, :], axis=1)
                s4 += tl.sum(kv_tile * q4[None, :], axis=1)
                s5 += tl.sum(kv_tile * q5[None, :], axis=1)
                s6 += tl.sum(kv_tile * q6[None, :], axis=1)
                s7 += tl.sum(kv_tile * q7[None, :], axis=1)

            w0 = tl.load(w_base + (hg_start + 0))
            w1 = tl.load(w_base + (hg_start + 1))
            w2 = tl.load(w_base + (hg_start + 2))
            w3 = tl.load(w_base + (hg_start + 3))
            w4 = tl.load(w_base + (hg_start + 4))
            w5 = tl.load(w_base + (hg_start + 5))
            w6 = tl.load(w_base + (hg_start + 6))
            w7 = tl.load(w_base + (hg_start + 7))

            acc += tl.maximum(s0, 0.0) * w0
            acc += tl.maximum(s1, 0.0) * w1
            acc += tl.maximum(s2, 0.0) * w2
            acc += tl.maximum(s3, 0.0) * w3
            acc += tl.maximum(s4, 0.0) * w4
            acc += tl.maximum(s5, 0.0) * w5
            acc += tl.maximum(s6, 0.0) * w6
            acc += tl.maximum(s7, 0.0) * w7

        acc = tl.where(combined_mask, acc, float("-inf"))
        out_ptrs = (Out_buffer
                    + (pid_batch * next_n + pid_next_n) * stride_out_row
                    + context_idx + kv_offsets)
        tl.store(out_ptrs, acc, mask=mask_kv)

    def int8_sparse_vec_decode(
        q: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        d_v: int = D_V,
        group_sizes: tuple[int, ...] = GS_SINGLE,
        BLOCK_M: int = 16,
        TILE_K: int = 32,
        D_V_CHUNK: int = 256,
        num_warps: int = 4,
    ) -> torch.Tensor:
        """GPU triton decode variant (int8 loads + scale multiply)."""
        _as_i8(kv_data, "kv_data")
        _as_f32(kv_scale, "kv_scale")
        s_q, h_q, d_qk = q.shape
        assert d_qk == D_QK, d_qk
        assert d_v <= D_LATENT, d_v
        assert kv_data.stride(-1) == 1 and kv_scale.stride(-1) == 1
        s_kv = kv_data.shape[0]
        if indices.dim() == 3:
            indices = indices[:, 0, :]
        idx = _as_i64(indices, "indices").contiguous()
        topk = idx.shape[1]
        out = torch.empty((s_q, h_q, d_v), device=q.device, dtype=q.dtype)
        G = num_groups(group_sizes)
        if G == 1:
            assert group_sizes == GS_SINGLE, group_sizes
            SPLIT = 0
        else:
            assert G == 2 and group_sizes == GS_SPLIT, (
                "triton G=2 supports the (512, 64) split only"
            )
            SPLIT = D_LATENT
        assert h_q % BLOCK_M == 0 and d_v % D_V_CHUNK == 0
        grid = (s_q, h_q // BLOCK_M, d_v // D_V_CHUNK)
        _int8_sparse_vec_kernel[grid](
            output_ptr=out, query_ptr=q, kv_data_ptr=kv_data, kv_scale_ptr=kv_scale,
            topk_indices_ptr=idx,
            stride_out_sq=out.stride(0), stride_out_hq=out.stride(1),
            stride_q_sq=q.stride(0), stride_q_hq=q.stride(1),
            stride_kv=kv_data.stride(0), stride_scale=kv_scale.stride(0),
            stride_idx_sq=idx.stride(0),
            scale=sm_scale, s_kv=s_kv,
            D_QK_C=d_qk, D_V_C=d_v, TOPK=topk,
            D_SCORE_CHUNK=64, D_V_CHUNK=D_V_CHUNK, BLOCK_M=BLOCK_M, TILE_K=TILE_K,
            SPLIT=SPLIT, G=G,
            num_warps=num_warps, num_stages=1,
        )
        return out

    def int8_paged_mqa_logits_triton(
        q: torch.Tensor,             # [B, next_n, H, D]
        kv_raw: torch.Tensor,        # [num_blocks, block_size, ROW] int8
        weights: torch.Tensor,       # [B*next_n, H] f32
        context_lens: torch.Tensor,  # [B] int32
        block_tables: torch.Tensor,  # [B, max_blocks] int32
        max_model_len: int,
        group_sizes: tuple[int, ...] = GS_SINGLE,
        BLOCK_D: int = 32,
        num_warps: int = 4,
    ) -> torch.Tensor:
        _as_i8(kv_raw, "kv_raw")
        B, next_n, H, dim = q.shape
        num_blocks, block_size, _ = kv_raw.shape
        assert dim == D_QK, dim
        assert H % 8 == 0, H
        G = num_groups(group_sizes)
        if G == 1:
            assert group_sizes == GS_SINGLE
            SPLIT = 0
        else:
            assert G == 2 and group_sizes == GS_SPLIT
            SPLIT = D_LATENT
        data, scale = views_from_packed(kv_raw, group_sizes)
        out_qk = torch.full(
            (B * next_n, max_model_len), float("-inf"),
            device=q.device, dtype=torch.float32,
        )
        max_num_blocks = block_tables.shape[1]
        grid = (max_num_blocks, B * next_n)
        _int8_paged_mqa_logits_kernel[grid](
            B, next_n,
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            data, scale,
            data.stride(0), data.stride(1),
            scale.stride(0), scale.stride(1),
            context_lens, block_tables,
            weights, weights.stride(0),
            out_qk, out_qk.stride(0),
            max_num_blocks,
            NUM_HEADS=H, BLOCK_D=BLOCK_D, HEAD_DIM=dim, CHUNK_K=block_size,
            SPLIT=SPLIT, G=G,
            num_warps=num_warps, num_stages=1,
        )
        return out_qk

else:

    def int8_sparse_vec_decode(*_a, **_k):  # type: ignore
        raise RuntimeError("triton not available; use *_torch reference")

    def int8_paged_mqa_logits_triton(*_a, **_k):  # type: ignore
        raise RuntimeError("triton not available; use *_torch reference")


# ---------------------------------------------------------------------------
# Self-test harness
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    # Logits tensors carry -inf at masked positions by design; -inf - -inf is
    # NaN.  Compare only mutually-finite entries and require the non-finite
    # pattern to agree exactly (an extra/missing -inf is a real error).
    fin = torch.isfinite(a) & torch.isfinite(b)
    if not bool((torch.isfinite(a) == torch.isfinite(b)).all()):
        return float("inf")
    # Non-finite entries must agree exactly; NaN==NaN is False, so this also
    # fails loudly on a NaN-vs-NaN coincidence.
    nf = ~fin
    if nf.any() and not bool((a[nf] == b[nf]).all()):
        return float("inf")
    af = a[fin]
    bf = b[fin]
    denom = float(torch.linalg.norm(bf))
    if denom == 0.0:
        denom = 1.0
    return float(torch.linalg.norm(af - bf)) / denom


def _mk_qkv(rows: int, s_q: int, h_q: int, topk: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(rows, D_QK, generator=g, dtype=torch.float32)
    kv[:, D_LATENT:] *= 0.5  # rope dims typically smaller than latent
    q = torch.randn(s_q, h_q, D_QK, generator=g, dtype=torch.float32).to(torch.float16)
    idx = torch.randint(-2, rows, (s_q, topk), generator=g, dtype=torch.int64)
    idx[:, : max(1, topk // 8)] = -1  # force invalid slots (masking path)
    return kv, q, idx


def test_layout(ctx: Ctx) -> None:
    for gs in (GS_SINGLE, GS_SPLIT, (64,) * 9):
        ROW = row_bytes(gs)
        G = num_groups(gs)
        assert ROW >= D_QK + 4 * G and ROW % 8 == 0
        x = torch.randn(37, D_QK)
        q, s = quantize_rows_int8(x, gs)
        raw = pack_rows_int8(q, s, gs)
        q2, s2 = unpack_rows_int8(raw, gs)
        ok = torch.equal(q, q2) and torch.equal(s, s2)
        ctx.check(f"pack/unpack bit-exact gs={gs}", ok, f"ROW={ROW} G={G}")
        dv, sv = views_from_packed(raw.contiguous(), gs)
        okv = torch.equal(dv, q2) and torch.equal(sv, s2)
        ctx.check(f"views_from_packed gs={gs}", okv)


def test_quant_dequant(ctx: Ctx) -> None:
    for gs in (GS_SINGLE, GS_SPLIT):
        x = torch.randn(256, D_QK) * 3.0
        q, s = quantize_rows_int8(x, gs)
        x_hat = dequantize_rows_int8(q, s, gs)
        r = rel_err(x_hat, x)
        ctx.check(f"quant/dequant rel gs={gs}", r < 1e-2, f"rel={r:.3e} gate 1e-2")
        assert q.dtype == torch.int8 and s.dtype == torch.float32
        assert int(q.abs().max()) <= 127
    x0 = torch.zeros(4, D_QK)
    q0, s0 = quantize_rows_int8(x0)
    ok = (
        bool(torch.isfinite(s0).all())
        and bool((s0 == 1.0).all())
        and int(q0.abs().max()) == 0
    )
    ctx.check("zero-row scale=1.0 q=0", ok)


def test_write_path(ctx: Ctx) -> None:
    """Write via int8_concat_and_cache_mla, read back — must equal a direct
    per-row quantize.  Catches scale-row misalignment (the production bug
    class this inline layout exists to prevent)."""
    for gs in (GS_SINGLE, GS_SPLIT):
        ROW = row_bytes(gs)
        T, rows = 32, 128
        g = torch.Generator().manual_seed(1)
        kv_c = torch.randn(T, D_LATENT, generator=g).to(torch.float16)
        k_pe = torch.randn(T, D_ROPE, generator=g).to(torch.float16)
        # int32 slots on purpose: the write must force int64 internally.
        slots = torch.arange(T, dtype=torch.int32)  # row t -> slot t
        slots[0:3] = -1  # padded tokens must not write
        cache = torch.zeros(rows, ROW, dtype=torch.int8)
        int8_concat_and_cache_mla_torch(kv_c, k_pe, cache, slots, gs)

        full = torch.cat([kv_c, k_pe], dim=-1).float()
        q_ref, s_ref = quantize_rows_int8(full, gs)
        dv, sv = views_from_packed(cache, gs)
        ok = True
        for t in range(T):
            s = int(slots[t].item())
            if s < 0:
                continue
            if not (torch.equal(dv[s], q_ref[t]) and torch.equal(sv[s], s_ref[t])):
                ok = False
                break
        ctx.check(f"write/read exact gs={gs}", ok, f"T={T} rows={rows}")
        untouched = bool((cache[:3] == 0).all())  # slots -1 -> rows 0..2 blank
        ctx.check(f"slot=-1 no-write gs={gs}", untouched)


def test_prefill_rel(ctx: Ctx) -> None:
    rows, s_q, h_q, topk = 512, 24, 8, 96
    sm_scale = D_QK**-0.5
    kv_fp, q, idx = _mk_qkv(rows, s_q, h_q, topk, seed=2)
    out_fp16 = fp16_mla_sparse_prefill(q, kv_fp.to(torch.float16), idx, sm_scale)
    for gs in (GS_SINGLE, GS_SPLIT, (64,) * 9):
        q8, s8 = quantize_rows_int8(kv_fp, gs)
        out_i8 = int8_mla_sparse_prefill(q, q8, s8, idx, sm_scale, group_sizes=gs)
        r = rel_err(out_i8, out_fp16)
        ctx.check(
            f"prefill int8-vs-fp16 rel gs={gs}", r < 2e-2,
            f"rel={r:.3e} gate 2e-2",
        )
        assert out_i8.dtype == q.dtype


def test_decode_rel(ctx: Ctx) -> None:
    rows, s_q, h_q, topk = 512, 3, 8, 96  # decode-ish: few query rows
    sm_scale = D_QK**-0.5
    kv_fp, q, idx = _mk_qkv(rows, s_q, h_q, topk, seed=3)
    out_fp16 = fp16_mla_sparse_prefill(q, kv_fp.to(torch.float16), idx, sm_scale)
    for gs in (GS_SINGLE, GS_SPLIT):
        q8, s8 = quantize_rows_int8(kv_fp, gs)
        out_i8 = int8_sparse_vec_decode_torch(q, q8, s8, idx, sm_scale, group_sizes=gs)
        r = rel_err(out_i8, out_fp16)
        ctx.check(
            f"decode int8-vs-fp16 rel gs={gs}", r < 2e-2,
            f"rel={r:.3e} gate 2e-2",
        )


def test_paged_logits_rel(ctx: Ctx) -> None:
    ROW = row_bytes(GS_SINGLE)
    num_blocks, block_size = 8, 32
    B, next_n, H = 2, 2, 8
    max_model_len = num_blocks * block_size
    g = torch.Generator().manual_seed(4)
    kv_fp = torch.randn(num_blocks * block_size, D_QK, generator=g)
    q = torch.randn(B, next_n, H, D_QK, generator=g).to(torch.float16)
    weights = torch.randn(B * next_n, H, generator=g, dtype=torch.float32)
    context_lens = torch.tensor([100, 90], dtype=torch.int32)
    block_tables = torch.zeros(B, num_blocks, dtype=torch.int32)
    block_tables[0] = torch.arange(num_blocks)
    block_tables[1] = torch.arange(num_blocks)

    q8, s8 = quantize_rows_int8(kv_fp, GS_SINGLE)
    raw = pack_rows_int8(q8, s8, GS_SINGLE).reshape(num_blocks, block_size, ROW)
    out_i8 = int8_paged_mqa_logits_torch(
        q, raw, weights, context_lens, block_tables, max_model_len, GS_SINGLE
    )
    out_ref = _paged_logits_float_ref(
        q, kv_fp, weights, context_lens, block_tables,
        max_model_len, num_blocks, block_size,
    )
    r = rel_err(out_i8, out_ref)
    ctx.check("paged-logits int8-vs-fp32 rel", r < 2e-2, f"rel={r:.3e} gate 2e-2")
    # causal sanity: strictly past context_len must be -inf
    ok_mask = bool(torch.isneginf(out_ref[0, 100:110]).all())
    ctx.check("paged-logits causal -inf mask", ok_mask)


def test_layout_sensitivity(ctx: Ctx) -> None:
    """The harness must DETECT a wrong scale layout: rolled scales must blow
    the rel gate (guard against a micro-test that passes anyway)."""
    rows, s_q, h_q, topk = 256, 8, 4, 64
    sm_scale = D_QK**-0.5
    kv_fp, q, idx = _mk_qkv(rows, s_q, h_q, topk, seed=5)
    out_fp16 = fp16_mla_sparse_prefill(q, kv_fp.to(torch.float16), idx, sm_scale)
    q8, s8 = quantize_rows_int8(kv_fp)
    s_bad = torch.roll(s8, shifts=1, dims=0)  # wrong row alignment
    out_bad = int8_mla_sparse_prefill(q, q8, s_bad, idx, sm_scale)
    r = rel_err(out_bad, out_fp16)
    ctx.check(
        "layout-sensitivity (rolled scales must fail)", r > 5e-2,
        f"rel={r:.3e} want > 5e-2",
    )


def test_gpu(ctx: Ctx) -> None:
    if not HAS_TRITON or not torch.cuda.is_available():
        ctx.check("gpu triton", False, "triton/CUDA unavailable")
        return
    dev = torch.device("cuda")
    rows, s_q, h_q, topk = 512, 16, 16, 96
    sm_scale = D_QK**-0.5
    kv_fp, q, idx = _mk_qkv(rows, s_q, h_q, topk, seed=7)
    q = q.to(dev)
    idx = idx.to(dev)
    out_fp16 = fp16_mla_sparse_prefill(
        q, kv_fp.to(torch.float16).to(dev), idx, sm_scale
    )
    for gs in (GS_SINGLE, GS_SPLIT):
        q8, s8 = quantize_rows_int8(kv_fp, gs)
        q8, s8 = q8.to(dev), s8.to(dev)
        out_torch = int8_sparse_vec_decode_torch(
            q, q8, s8, idx, sm_scale, group_sizes=gs
        )
        out_tri = int8_sparse_vec_decode(q, q8, s8, idx, sm_scale, group_sizes=gs)
        r_t = rel_err(out_torch, out_fp16)
        r_k = rel_err(out_tri, out_torch)
        ctx.check(
            f"gpu sparse-vec triton gs={gs}",
            r_k < 2e-2 and r_t < 2e-2,
            f"triton-vs-torch rel={r_k:.3e} int8-vs-fp16 rel={r_t:.3e}",
        )
    # paged-logits triton vs torch
    ROW = row_bytes(GS_SINGLE)
    num_blocks, block_size = 8, 32
    B, next_n, H = 2, 2, 8
    max_model_len = num_blocks * block_size
    g = torch.Generator().manual_seed(8)
    kv_fp2 = torch.randn(num_blocks * block_size, D_QK, generator=g)
    q2 = torch.randn(B, next_n, H, D_QK, generator=g).to(torch.float16).to(dev)
    w2 = torch.randn(B * next_n, H, generator=g, dtype=torch.float32).to(dev)
    cl = torch.tensor([100, 90], dtype=torch.int32, device=dev)
    bt = torch.zeros(B, num_blocks, dtype=torch.int32, device=dev)
    bt[0] = torch.arange(num_blocks, device=dev)
    bt[1] = torch.arange(num_blocks, device=dev)
    q8b, s8b = quantize_rows_int8(kv_fp2, GS_SINGLE)
    raw = pack_rows_int8(q8b, s8b, GS_SINGLE).reshape(
        num_blocks, block_size, ROW
    ).to(dev)
    out_t = int8_paged_mqa_logits_torch(
        q2, raw, w2, cl, bt, max_model_len, GS_SINGLE
    )
    out_k = int8_paged_mqa_logits_triton(
        q2, raw, w2, cl, bt, max_model_len, GS_SINGLE
    )
    r = rel_err(out_k, out_t)
    ctx.check("gpu paged-logits triton-vs-torch", r < 2e-2, f"rel={r:.3e} gate 2e-2")


def main() -> int:
    ap = argparse.ArgumentParser(description="int8 MLA latent KV self-test")
    ap.add_argument("--gpu", action="store_true", help="also run triton GPU tests")
    args = ap.parse_args()

    torch.manual_seed(0)
    ctx = Ctx()
    print("== int8_kv_kernel self-test ==")
    print(
        f"torch={torch.__version__} triton={'yes' if HAS_TRITON else 'no'} "
        f"cuda={torch.cuda.is_available()}"
    )
    test_layout(ctx)
    test_quant_dequant(ctx)
    test_write_path(ctx)
    test_prefill_rel(ctx)
    test_decode_rel(ctx)
    test_paged_logits_rel(ctx)
    test_layout_sensitivity(ctx)
    if args.gpu:
        test_gpu(ctx)
    print(f"== {ctx.passed} passed, {ctx.failed} failed ==")
    return 1 if ctx.failed else 0


if __name__ == "__main__":
    sys.exit(main())
