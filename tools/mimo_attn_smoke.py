#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline smoke test for the MiMo-V2 Triton attention port (v1).

Exercises the fork's TritonAttentionImpl (KV-cache write + attention forward)
at MiMo-V2.6-Flash shapes (d_qk=192, d_v=128 zero-padded to 192 like
``MiMoV2Attention.forward`` does, heads=64, kv_heads in {4, 8}, window=128,
optional attention sinks) and compares against a torch reference attention
computed in fp32 (same semantics as the HF MiMoV2 eager path: causal, sliding
window ``q - k < window``, virtual sink key).

Each case runs a prefill step (2 sequences, lens 512 / 256) and an incremental
decode step (1 new token per sequence) through both:
  * ``TritonAttentionImpl.forward`` (the boot path), and
  * ``unified_attention`` directly (forces the Triton kernel).

Usage:
    HIP_VISIBLE_DEVICES=<gpu> python3 tools/mimo_attn_smoke.py [--device cuda]

Pass criterion: relative L2 error < 2e-2 in fp16.
"""

import argparse
import sys

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    NUM_PAR_SOFTMAX_SEGMENTS,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.utils.math_utils import next_power_of_2

NUM_HEADS = 64
D_QK = 192
D_V = 128
SCALE = D_QK**-0.5
REL_ERR_LIMIT = 2e-2


def reference_attention(q, k, v, qpos, window, sinks):
    """fp32 reference for one sequence.

    q:   [Nq, H, D_QK] query rows (positions ``qpos``)
    k/v: [L, KV, ...] full sequence cache, L > max(qpos)
    qpos: LongTensor [Nq], absolute positions of the query rows
    """
    nq, H, _ = q.shape
    kv = k.shape[1]
    length = k.shape[0]
    rep = H // kv
    qq = q.float()
    kk = k.float().repeat_interleave(rep, dim=1)  # [L, H, D_QK]
    vv = v.float().repeat_interleave(rep, dim=1)  # [L, H, D_V]
    scores = torch.einsum("ihd,jhd->ihj", qq, kk) * SCALE  # [Nq, H, L]
    j = torch.arange(length, device=q.device)
    mask = j[None, :] <= qpos[:, None].to(j.device)
    if window is not None:
        mask = mask & ((qpos[:, None].to(j.device) - j[None, :]) < window)
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    if sinks is not None:
        sink_col = sinks.float()[None, :, None].expand(nq, H, 1)
        scores = torch.cat([scores, sink_col], dim=-1)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    if sinks is not None:
        probs = probs[..., :-1]
    return torch.einsum("ihj,jhd->ihd", probs, vv)  # [Nq, H, D_V]


def rel_err(got, want):
    got = got.float()
    want = want.float()
    denom = want.norm().clamp_min(1e-12)
    return (got - want).norm().item() / denom.item()


def install_route_probe():
    """Record which internal route TritonAttentionImpl.forward takes."""
    from vllm.v1.attention.backends import triton_attn as triton_attn_mod

    routes: list[str] = []

    orig_unified = triton_attn_mod.unified_attention

    def unified(*args, **kwargs):
        routes.append("unified_attention")
        return orig_unified(*args, **kwargs)

    triton_attn_mod.unified_attention = unified
    for name, tag in (
        ("torch_sdpa_prefill_attention", "torch_sdpa_prefill"),
        ("torch_sdpa_decode_attention", "torch_sdpa_decode"),
        ("torch_sdpa_mtp_decode_attention", "torch_sdpa_mtp_decode"),
    ):
        orig = getattr(triton_attn_mod, name)

        def make(orig=orig, tag=tag):

            def wrapper(*args, **kwargs):
                routes.append(tag)
                return orig(*args, **kwargs)

            return wrapper

        setattr(triton_attn_mod, name, make())
    return routes


class _FakeScaleLayer:
    """Stand-in for the Attention layer's quant-scale attributes."""

    def __init__(self, device):
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0


def build_case_metadata(
    query_lens,
    seq_lens,
    block_table,
    slot_mapping,
    segm_buffers,
    seq_threshold_3d,
    device,
):
    edges = [0]
    for qlen in query_lens:
        edges.append(edges[-1] + qlen)
    query_start_loc = torch.tensor(edges, dtype=torch.int32, device=device)
    seq_lens_gpu = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    is_prefilling = torch.tensor(
        [qlen > 1 for qlen in query_lens], dtype=torch.bool, device=device
    )
    segm_output, segm_max, segm_expsum = segm_buffers
    return TritonAttentionMetadata(
        num_actual_tokens=edges[-1],
        max_query_len=max(query_lens),
        max_decode_query_len=1,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        max_seq_len=max(seq_lens),
        seq_lens=seq_lens_gpu,
        seq_lens_cpu=seq_lens_gpu.cpu(),
        seq_lens_cpu_upper_bound=None,
        is_prefilling=is_prefilling,
        block_table=block_table,
        slot_mapping=slot_mapping,
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        causal=True,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def run_case(tag, num_kv_heads, window, use_sink, block_size, device, dtype, routes):
    torch.manual_seed(1234)
    kv_heads = num_kv_heads
    seq_lens = [512, 256]
    seq_ranges = []
    start = 0
    for length in seq_lens:
        seq_ranges.append((start, start + length))
        start += length
    num_tokens = start
    num_seqs = len(seq_lens)

    q = torch.randn(num_tokens, NUM_HEADS, D_QK, device=device, dtype=dtype) * 0.5
    k = torch.randn(num_tokens, kv_heads, D_QK, device=device, dtype=dtype) * 0.5
    v_real = torch.randn(num_tokens, kv_heads, D_V, device=device, dtype=dtype) * 0.5
    # v1 port: zero-pad V to D_QK exactly like MiMoV2Attention.forward.
    v_pad = torch.nn.functional.pad(v_real, (0, D_QK - D_V))

    sinks = None
    if use_sink:
        sinks = torch.randn(NUM_HEADS, device=device, dtype=torch.float32) * 0.5

    # Paged KV cache with the padded-V layout:
    # [num_blocks, 2, block_size, kv_heads, D_QK].
    blocks_per_seq = (max(seq_lens) + 4 + block_size - 1) // block_size
    num_blocks = num_seqs * blocks_per_seq + 8
    kv_cache = torch.zeros(
        num_blocks, 2, block_size, kv_heads, D_QK, device=device, dtype=dtype
    )

    block_table = torch.zeros(
        num_seqs, blocks_per_seq, dtype=torch.int32, device=device
    )
    slot_mapping = torch.empty(num_tokens, dtype=torch.int64, device=device)
    pos = torch.arange(max(seq_lens) + 4, device=device)
    for i, length in enumerate(seq_lens):
        base = i * blocks_per_seq
        block_table[i] = base + torch.arange(blocks_per_seq, device=device)
        slots = (base + pos // block_size) * block_size + pos % block_size
        slot_mapping[seq_ranges[i][0] : seq_ranges[i][1]] = slots[:length]

    impl = TritonAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=D_QK,
        scale=SCALE,
        num_kv_heads=kv_heads,
        alibi_slopes=None,
        sliding_window=window if window is not None else None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        sinks=sinks,
    )
    layer = _FakeScaleLayer(device)

    headdim_padded = next_power_of_2(D_QK)
    seq_threshold_3d = max(1, 128 // kv_heads)
    segm_buffers = (
        torch.zeros(
            seq_threshold_3d,
            NUM_HEADS,
            NUM_PAR_SOFTMAX_SEGMENTS,
            headdim_padded,
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros(
            seq_threshold_3d,
            NUM_HEADS,
            NUM_PAR_SOFTMAX_SEGMENTS,
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros(
            seq_threshold_3d,
            NUM_HEADS,
            NUM_PAR_SOFTMAX_SEGMENTS,
            dtype=torch.float32,
            device=device,
        ),
    )
    window_size = (window - 1, 0) if window is not None else (-1, -1)
    results = {}

    # ---- step 1: prefill of both sequences --------------------------------
    impl.do_kv_cache_update(layer, k, v_pad, kv_cache, slot_mapping)
    md_prefill = build_case_metadata(
        seq_lens,
        seq_lens,
        block_table,
        slot_mapping,
        segm_buffers,
        seq_threshold_3d,
        device,
    )
    out_impl = torch.empty(num_tokens, NUM_HEADS, D_QK, device=device, dtype=dtype)
    routes.clear()
    impl.forward(layer, q, k, v_pad, kv_cache, md_prefill, out_impl)
    prefill_route = ",".join(routes) or "unknown"

    out_kern = torch.empty(num_tokens, NUM_HEADS, D_QK, device=device, dtype=dtype)
    unified_attention(
        q=q,
        k=kv_cache[:, 0],
        v=kv_cache[:, 1],
        out=out_kern,
        cu_seqlens_q=md_prefill.query_start_loc,
        max_seqlen_q=md_prefill.max_query_len,
        seqused_k=md_prefill.seq_lens,
        max_seqlen_k=md_prefill.max_seq_len,
        softmax_scale=SCALE,
        causal=True,
        window_size=window_size,
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
    )

    ref_rows = []
    for lo, hi in seq_ranges:
        qpos = torch.arange(hi - lo, device=device)
        ref_rows.append(
            reference_attention(
                q[lo:hi], k[lo:hi], v_real[lo:hi], qpos, window, sinks
            )
        )
    ref = torch.cat(ref_rows, dim=0)
    results["prefill_impl"] = rel_err(out_impl[..., :D_V], ref)
    results["prefill_kernel"] = rel_err(out_kern[..., :D_V], ref)
    results["prefill_pad_leak"] = out_kern[..., D_V:].float().abs().max().item()

    # ---- step 2: incremental decode, one new token per sequence -----------
    num_new = num_seqs
    q_new = torch.randn(num_new, NUM_HEADS, D_QK, device=device, dtype=dtype) * 0.5
    k_new = torch.randn(num_new, kv_heads, D_QK, device=device, dtype=dtype) * 0.5
    v_new = torch.randn(num_new, kv_heads, D_V, device=device, dtype=dtype) * 0.5
    v_new_pad = torch.nn.functional.pad(v_new, (0, D_QK - D_V))

    new_slots = []
    for i, length in enumerate(seq_lens):
        base = i * blocks_per_seq
        t = length  # append at position `length`
        new_slots.append((base + t // block_size) * block_size + t % block_size)
    slot_mapping_new = torch.tensor(new_slots, dtype=torch.int64, device=device)
    impl.do_kv_cache_update(layer, k_new, v_new_pad, kv_cache, slot_mapping_new)

    seq_lens_dec = [length + 1 for length in seq_lens]
    md_decode = build_case_metadata(
        [1] * num_seqs,
        seq_lens_dec,
        block_table,
        slot_mapping_new,
        segm_buffers,
        seq_threshold_3d,
        device,
    )
    out_dec_impl = torch.empty(num_new, NUM_HEADS, D_QK, device=device, dtype=dtype)
    routes.clear()
    impl.forward(layer, q_new, k_new, v_new_pad, kv_cache, md_decode, out_dec_impl)
    decode_route = ",".join(routes) or "unknown"

    out_dec_kern = torch.empty(num_new, NUM_HEADS, D_QK, device=device, dtype=dtype)
    unified_attention(
        q=q_new,
        k=kv_cache[:, 0],
        v=kv_cache[:, 1],
        out=out_dec_kern,
        cu_seqlens_q=md_decode.query_start_loc,
        max_seqlen_q=1,
        seqused_k=md_decode.seq_lens,
        max_seqlen_k=md_decode.max_seq_len,
        softmax_scale=SCALE,
        causal=True,
        window_size=window_size,
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
    )

    ref_dec_rows = []
    for i, length in enumerate(seq_lens):
        k_full = torch.cat([k[seq_ranges[i][0] : seq_ranges[i][1]], k_new[i : i + 1]])
        v_full = torch.cat(
            [v_real[seq_ranges[i][0] : seq_ranges[i][1]], v_new[i : i + 1]]
        )
        qpos = torch.tensor([length], device=device)
        ref_dec_rows.append(
            reference_attention(q_new[i : i + 1], k_full, v_full, qpos, window, sinks)
        )
    ref_dec = torch.cat(ref_dec_rows, dim=0)
    results["decode_impl"] = rel_err(out_dec_impl[..., :D_V], ref_dec)
    results["decode_kernel"] = rel_err(out_dec_kern[..., :D_V], ref_dec)
    results["decode_pad_leak"] = out_dec_kern[..., D_V:].float().abs().max().item()

    ok = all(
        results[key] < REL_ERR_LIMIT
        for key in (
            "prefill_impl",
            "prefill_kernel",
            "decode_impl",
            "decode_kernel",
        )
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {tag}: kv_heads={kv_heads} window={window} "
        f"sink={use_sink} block={block_size}"
    )
    print(
        "  prefill  route={route} impl={prefill_impl:.3e} "
        "kernel={prefill_kernel:.3e} pad_leak={prefill_pad_leak:.2e}".format(
            route=prefill_route, **results
        )
    )
    print(
        "  decode   route={route} impl={decode_impl:.3e} "
        "kernel={decode_kernel:.3e} pad_leak={decode_pad_leak:.2e}".format(
            route=decode_route, **results
        )
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="torch device to use")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(
            f"device={device} name={torch.cuda.get_device_name(device)} "
            f"free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB"
        )

    cases = [
        # (tag, kv_heads, window, use_sink, block_size)
        ("full-attn", 4, None, False, 16),
        ("full-attn", 4, None, False, 256),
        ("full-attn+sink", 4, None, True, 16),
        ("swa", 8, 128, False, 16),
        ("swa+sink", 8, 128, True, 16),
        ("swa+sink", 8, 128, True, 256),
    ]
    routes = install_route_probe()
    all_ok = True
    for tag, kv_heads, window, use_sink, block_size in cases:
        ok = run_case(
            tag, kv_heads, window, use_sink, block_size, device, dtype, routes
        )
        all_ok = all_ok and ok

    print("ALL PASS" if all_ok else "SOME CASES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
