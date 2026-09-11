# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# GLM53-PORT: fp16-only ROCm/gfx906 implementation of the GLM-5.3-Flash kpool
# sparse-attention indexer op. Upstream main implements this flow in
# model_executor/layers/sparse_attn_indexer_kpool.py on top of fp8/DeepGEMM
# primitives; this file instead builds on the fork's DSV4-validated fp16
# family (fp16 MQA logits triton/torch kernels + CP gather fp16 cache op).
#
# Semantics vs upstream (fp8) path: identical pool construction
# (softmax(gate+ape)-weighted sum) and pool-granular topk + tail expansion,
# but K entries are stored/attended in fp16 without the fp8 Hadamard rotation
# and per-vector scale.
"""Custom Sparse Attention Indexer with K-pooling (fp16 radix) for gfx906."""

from typing import Optional

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType, _resolve_layer_name
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def _build_decode_scatter_indices(
    decode_lens: torch.Tensor,
    num_requests: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (request id, intra-request index) for a non-uniform decode
    batch, with ``n == decode_lens.sum()`` as a host int (no device sync).

    GLM53-PORT: vendored from upstream sparse_attn_indexer_kpool.py.
    """
    device = decode_lens.device
    dl = decode_lens.to(torch.int64)
    req_id = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int64),
        dl,
        output_size=n,
    )
    req_starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.int64), dl[:-1]]),
        dim=0,
    )
    starts = torch.repeat_interleave(req_starts, dl, output_size=n)
    intra = torch.arange(n, device=device, dtype=torch.int64) - starts
    return req_id, intra


def _scatter_decode_tokens_by_request(
    tokens: torch.Tensor,
    pad_value,
    num_requests: int,
    lmax: int,
    scatter_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """GLM53-PORT: vendored from upstream sparse_attn_indexer_kpool.py."""
    req_id, intra = scatter_indices
    out = torch.full(
        (num_requests, lmax, *tokens.shape[1:]),
        pad_value,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    out[req_id, intra] = tokens
    return out


def _decode_topk_seq_lens(
    positions: torch.Tensor,
    decode_lens: torch.Tensor,
    num_decode_tokens: int,
    batch_size: int,
    next_n: int,
    requires_padding: bool,
) -> torch.Tensor:
    """GLM53-PORT: vendored from upstream sparse_attn_indexer_kpool.py.

    Token-granular seq_len (pos + 1) per pool-topk row, layout-aware.
    """
    n = batch_size * next_n
    if not requires_padding:
        return positions[:n].to(torch.int32) + 1
    scatter_idx = _build_decode_scatter_indices(
        decode_lens, batch_size, num_decode_tokens
    )
    padded = _scatter_decode_tokens_by_request(
        positions[:num_decode_tokens].to(torch.int32),
        -1,
        batch_size,
        next_n,
        scatter_idx,
    )
    return padded.reshape(n) + 1  # pad rows: -1 + 1 = 0 -> empty tail


def _fill_causal_indices(rows: torch.Tensor, positions: torch.Tensor) -> None:
    causal_range = torch.arange(rows.shape[1], device=rows.device, dtype=torch.int32)
    positions = positions.to(torch.int32)
    rows[:] = causal_range[None, :]
    rows[causal_range[None, :] > positions[:, None]] = -1


def rocm_aiter_sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    gate_score: torch.Tensor,
    compress_ape: torch.Tensor,
    index_kpool: int,
    positions: torch.Tensor,
    tail_kv_cache: torch.Tensor | None,
    tail_prefix: LayerNameType | None,
) -> torch.Tensor:
    # profile run - workspace reservation is done by the caller
    return topk_indices_buffer


def rocm_aiter_sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    gate_score: torch.Tensor,
    compress_ape: torch.Tensor,
    index_kpool: int,
    positions: torch.Tensor,
    tail_kv_cache: torch.Tensor | None,
    tail_prefix: LayerNameType | None,
) -> torch.Tensor:
    """fp16 kpool indexer: pool-compress K into the paged cache, score pools
    with the DSV4 fp16 MQA-logits family, pick top-512 pools, expand to 2048
    token indices (+ tail), and stash the incomplete-pool tail.
    """
    from vllm.model_executor.models.glm5next.amd.ops import kpool_compress as kpool_ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp16_mqa_logits_triton,
        rocm_paged_mqa_logits,
    )

    attn_metadata = get_forward_context().attn_metadata
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace during profiling run so lock doesn't fail later
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), torch.float16),
        )
        return rocm_aiter_sparse_attn_indexer_kpool_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q,
            k,
            weights,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            tail_kv_cache,
            tail_prefix,
        )
    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    slot_mapping = layer_attn_metadata.slot_mapping
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens

    # Truncation guard for spec-decode cg padding (keep parity with DSV4 op).
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]

    assert index_kpool > 1, "kpool indexer op requires index_kpool > 1"
    kpool = index_kpool

    # ---------------- kpool prefill write + tail seed ----------------
    n_prefill = num_tokens - num_decode_tokens
    if n_prefill > 0:
        # Decode tokens are batched first; prefill tokens follow. Only the
        # last token of each complete pool carries a valid (>=0) slot.
        prefill_slice = slice(num_decode_tokens, num_tokens)
        slot_prefill = slot_mapping[prefill_slice]
        n = slot_prefill.shape[0]
        if n >= kpool:
            pos = torch.arange(n, device=k.device)
            valid = slot_prefill >= 0
            # Drop pools whose start falls before the batch (leading padding).
            write_mask = valid & (pos >= kpool - 1)
            offs = torch.arange(kpool, device=k.device)
            idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
            kpool_ops.kpool_compress_and_write_cache_fp16(
                kv_cache,
                k[idx],  # [n, kpool, head_dim]
                gate_score[idx],
                compress_ape,
                slot_prefill.to(torch.int64),
                pool_size=kpool,
                head_dim=head_dim,
                write_mask=write_mask,
            )
        if tail_kv_cache is not None and tail_prefix is not None:
            tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                kpool_ops.kpool_seed_tail_cache_generic(
                    tail_kv_cache,
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    tail_meta.slot_mapping[prefill_slice],
                    kpool,
                    head_dim,
                )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    select_k = topk_tokens // kpool

    # ---------------- prefill: gather + logits + pool topk ----------------
    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None

        # Short sequences select every pool; fill exact causal indices.
        short_prefill = False
        if n_prefill > 0 and positions.numel() > 0:
            # NOTE: int() here is a device sync per (layer, prefill) — the
            # upstream host-side max_prefill_seq_len optimization is not in
            # this fork's chunk metadata (GLM53-PORT follow-up).
            max_pos = int(positions[num_decode_tokens:num_tokens].max().item()) + 1
            short_prefill = max_pos <= topk_tokens
        if short_prefill:
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _fill_causal_indices(_buf, _pos)
        else:
            workspace_manager = current_workspace_manager()
            (k_fp_full,) = workspace_manager.get_simultaneous(
                ((total_seq_lens, head_dim), torch.float16),
            )
            for chunk in prefill_metadata.chunks:
                k_fp = k_fp_full[: chunk.total_seq_lens]
                if not chunk.skip_kv_gather:
                    ops.cp_gather_indexer_k_cache_fp16(
                        kv_cache,
                        k_fp,
                        chunk.block_table,
                        chunk.cu_seq_lens,
                    )
                logits = fp16_mqa_logits_triton(
                    q[chunk.token_start : chunk.token_end],
                    k_fp,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
                num_rows = logits.shape[0]
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    pool_topk,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
                pool_ids = pool_topk.to(torch.int64)
                # seq_len per query row (token-granular) from positions.
                q_seq = positions[chunk.token_start : chunk.token_end].to(
                    torch.int32
                ) + 1
                expanded = kpool_ops.expand_pools_and_append_tail(
                    pool_ids, q_seq, kpool
                )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded

    # ---------------- decode: tail update + paged logits + pool topk ----------------
    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None

        num_requests = layer_attn_metadata.num_decodes
        decode_lens = decode_metadata.decode_lens
        requires_padding = decode_metadata.requires_padding
        use_uniform = not requires_padding
        if use_uniform:
            next_n_u = num_decode_tokens // num_requests
            lmax = next_n_u
        else:
            lmax = int(decode_lens.max().item())

        # Tail update + completed-pool writes for decode tokens.
        if tail_kv_cache is not None and tail_prefix is not None:
            tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
        else:
            tail_meta = None
        if tail_meta is not None:
            assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if not use_uniform:
                scatter_idx = _build_decode_scatter_indices(
                    decode_lens, num_requests, num_decode_tokens
                )
                dec_k = _scatter_decode_tokens_by_request(
                    k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:num_decode_tokens], -1, num_requests, lmax, scatter_idx
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_tail_slot = _scatter_decode_tokens_by_request(
                    tail_meta.slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                shape2 = (num_requests, lmax)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
                dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
            kpool_ops.kpool_decode_update_and_maybe_write_cache_batched_fp16(
                kv_cache,
                tail_kv_cache,
                dec_tail_slot,
                dec_k,
                dec_gate,
                compress_ape,
                dec_slot,
                dec_pos,
                kpool,
                head_dim,
            )

        # Paged pool-granular logits (DSV4 fp16 family).
        if requires_padding:
            padded_q_decode_tokens = pack_seq_triton(
                q[:num_decode_tokens], decode_lens
            )
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_decode_tokens = q[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q.shape[1:]
            )
            padded_weights = weights[:num_decode_tokens]
        batch_size = padded_q_decode_tokens.shape[0]
        next_n = padded_q_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n

        # [num_blocks, storage_block, head_dim] -> [num_blocks, storage_block, 1, head_dim]
        kv_cache_4d = kv_cache.unsqueeze(-2)
        logits = rocm_paged_mqa_logits(
            padded_q_decode_tokens,
            kv_cache_4d,
            padded_weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )
        num_rows = logits.shape[0]
        assert select_k in (512, 1024, 2048), (
            "top_k_per_row_decode supports sizes 512/1024/2048"
        )
        pool_topk = torch.empty(
            (num_rows, select_k), dtype=torch.int32, device=logits.device
        )
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            pool_topk,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            select_k,
        )

        pool_ids = pool_topk.to(torch.int64)
        dec_seq = _decode_topk_seq_lens(
            positions,
            decode_lens,
            num_decode_tokens,
            batch_size,
            next_n,
            requires_padding,
        )
        out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, kpool)

        if requires_padding:
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer
