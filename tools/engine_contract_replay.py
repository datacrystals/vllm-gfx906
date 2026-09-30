#!/usr/bin/env python3
"""engine_contract_replay.py — capture/replay kernel-dev harness for the
sparse-MLA prefill call contract of ROCMAiterMLASparseImpl._forward_kv.

Develop attention kernels against the REAL engine call contract (shapes,
dtypes, strides, index semantics) without booting vLLM per iteration:
capture once from a live engine, then replay candidates offline in seconds.
CPU is fine for correctness; use --device cuda for representative timing.

CAPTURE (in-engine, once per boot)
    Instrumentation lives in vllm/v1/attention/backends/mla/
    rocm_aiter_mla_sparse.py (`_glm53_kc_dump`, marker GLM53-KC-DUMP).
    Launch the server with a clean reference configuration
    (VLLM_GLM53_PREFILL_UNION unset) plus:

        export VLLM_GLM53_KC_DUMP=/data/tmp/kc_dump.npz

    then send at least one long prompt. The first _forward_kv call with
    num_tokens > 256 is written to that path; the first call with
    num_tokens <= 4 (decode) to /data/tmp/kc_dump.decode.npz. Dumps are
    once-per-file (O_EXCL) and TP-rank-safe. The dump is skipped during
    CUDA-graph capture.

    npz fields:
        q        [s_q, h_q, d_qk]   REAL tensor (logical layout preserved)
        kv       [num_blocks, block_size, d_qk]   full kv_c_and_k_pe_cache
        indices  [s_q, topk]        int32 GLOBAL cache rows == rows of
                                    kv.view(-1, 1, d) after flattening
        out_ref  [s_q, h_q, d_v]    engine reference_mla_sparse_prefill output
        out_ref_unpadded            the same after AiterMLAHelper
                                    .get_mla_unpadded_o (what forward_mqa
                                    actually returns)
        scale, num_tokens, num_heads_impl, kv_lora_rank,
        q_shape, q_stride, kv_shape, kv_stride, indices_shape, indices_dtype,
        q_dtype, kv_dtype, block_size, topk_tokens,
        req_id_per_token, block_table

REPLAY (offline)
        python3 engine_contract_replay.py /data/tmp/kc_dump.npz
        python3 engine_contract_replay.py /data/tmp/kc_dump.npz --bench 5 --device cuda
        python3 engine_contract_replay.py /data/tmp/kc_dump.decode.npz --gate 1e-3

    For each captured call it runs:
      reference    - verbatim port of reference_mla_sparse_prefill
                     (checked against out_ref when present)
      union        - in-tree union_gather_prefill candidate
      union_flat   - variant: one unique set over all rows (no 64-chunking)
    and prints wall time + max-abs/rel error vs reference.
    PASS gate: rel < --gate (default 1e-3).

WRAPPER CONTRACT (why a kernel can pass standalone and fail in-engine)
    _forward_kv receives a possibly PADDED head dim: forward_mqa calls
    AiterMLAHelper.get_mla_padded_q, which repeat_interleaves heads when
    num_heads < 16 (GLM-5.3 on TP8: 64 heads -> 8/rank -> 16 padded, factor
    2). The reference path ends with AiterMLAHelper.get_mla_unpadded_o
    (o[:, ::factor, :]). A candidate that early-returns from _forward_kv
    without that unpad hands [s_q, 16, d_v] to _v_up_proj, which does
    x.view(-1, num_heads=8, d_v) = [2*s_q, 8, d_v] and silently scrambles
    token/head layout (or auto-resizes the output buffer). The harness
    prints a WRAPPER-WARNING whenever q.shape[1] != num_heads_impl.

NOTES
    * Keep the captured dtype of `indices` (int32 in production): deploy-
      blocking candidate bugs (e.g. scatter_add_ with int32 indices on ROCm)
      must surface here, not in production.
    * Engine reference numerics are NOT pure fp32: q@K^T and the output GEMM
      run in the kv dtype; only softmax bookkeeping is fp32. The reference
      port below mirrors that exactly so out_ref comparisons are meaningful.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

CHUNK = 64  # engine default envs.VLLM_ROCM_MLA_SPARSE_CHUNK_SIZE


# ---------------------------------------------------------------------------
# ground truth: verbatim port of reference_mla_sparse_prefill
# ---------------------------------------------------------------------------
def reference(q, kv, indices, sm_scale, d_v, chunk=CHUNK):
    """kv [rows, d_qk]; indices [s_q, topk] (any int dtype)."""
    topk = indices.shape[-1]
    s_kv = kv.shape[0]
    s_q, h_q, d_qk = q.shape
    out = torch.empty(s_q, h_q, d_v, device=q.device, dtype=kv.dtype)
    for start in range(0, s_q, chunk):
        end = min(start + chunk, s_q)
        idx_chunk = indices[start:end].clone()  # engine zeroes invalid in place
        invalid_mask = (idx_chunk < 0) | (idx_chunk >= s_kv)
        idx_chunk[invalid_mask] = 0
        gathered_kv = kv.index_select(0, idx_chunk.reshape(-1)).reshape(
            end - start, topk, d_qk)
        if kv.dtype == torch.float32:
            P = q[start:end] @ gathered_kv.transpose(1, 2)
        else:
            P = (q[start:end] @ gathered_kv.transpose(1, 2)).float()
        P.masked_fill_(invalid_mask.unsqueeze(1), float("-inf"))
        P = P * sm_scale
        orig_lse = torch.logsumexp(P, dim=-1)
        s_for_o = torch.exp(P - orig_lse.unsqueeze(-1))
        if kv.dtype == torch.float32:
            out[start:end] = s_for_o @ gathered_kv[..., :d_v]
        else:
            out[start:end] = s_for_o.to(kv.dtype) @ gathered_kv[..., :d_v]
    return out


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
def union(q, kv, indices, sm_scale, d_v):
    """In-tree union_gather_prefill (patch_prefill_union.py) as deployed."""
    s_q, h_q, d_qk = q.shape
    if s_q > 64:
        outs = []
        for s0 in range(0, s_q, 64):
            s1 = min(s0 + 64, s_q)
            outs.append(union(q[s0:s1], kv, indices[s0:s1], sm_scale, d_v))
        return torch.cat(outs, dim=0)
    topk = indices.shape[1]
    s_kv = kv.shape[0]
    invalid = (indices < 0) | (indices >= s_kv)
    safe = indices.masked_fill(invalid, 0)
    u_idx, inv = torch.unique(safe.reshape(-1), return_inverse=True)
    inv = inv.view(s_q, topk).to(torch.int64)  # the int32-corruption fix
    ku = kv.index_select(0, u_idx).contiguous()
    su = torch.einsum("thd,ud->thu", q.float(), ku.float()) * sm_scale
    s = su.gather(2, inv[:, None, :].expand(-1, h_q, -1))
    s = s.masked_fill(invalid[:, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = p.masked_fill(invalid[:, None, :], 0.0)
    p_sc = torch.zeros(s_q, h_q, u_idx.shape[0], device=q.device,
                       dtype=torch.float32)
    p_sc.scatter_add_(2, inv[:, None, :].expand(-1, h_q, -1), p)
    out = torch.einsum("thu,uv->thv", p_sc, ku[:, :d_v].float())
    return out.to(q.dtype)


def union_flat(q, kv, indices, sm_scale, d_v):
    """Variant: one unique set over ALL rows (no per-64 chunking)."""
    s_q, h_q, d_qk = q.shape
    topk = indices.shape[1]
    s_kv = kv.shape[0]
    invalid = (indices < 0) | (indices >= s_kv)
    safe = indices.masked_fill(invalid, 0)
    u_idx, inv = torch.unique(safe.reshape(-1), return_inverse=True)
    inv = inv.view(s_q, topk).to(torch.int64)
    ku = kv.index_select(0, u_idx).contiguous()
    su = torch.einsum("thd,ud->thu", q.float(), ku.float()) * sm_scale
    s = su.gather(2, inv[:, None, :].expand(-1, h_q, -1))
    s = s.masked_fill(invalid[:, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = p.masked_fill(invalid[:, None, :], 0.0)
    p_sc = torch.zeros(s_q, h_q, u_idx.shape[0], device=q.device,
                       dtype=torch.float32)
    p_sc.scatter_add_(2, inv[:, None, :].expand(-1, h_q, -1), p)
    out = torch.einsum("thu,uv->thv", p_sc, ku[:, :d_v].float())
    return out.to(q.dtype)


CANDIDATES = {
    "union": union,
    "union_flat": union_flat,
}


def unpad(o, num_heads):
    """AiterMLAHelper.get_mla_unpadded_o semantics."""
    if o.shape[1] == num_heads:
        return o
    return o[:, :: o.shape[1] // num_heads, :]


def load_contract(path):
    z = np.load(path, allow_pickle=False)
    q = torch.as_tensor(z["q"])
    kv = torch.as_tensor(z["kv"])
    if kv.dim() == 3:  # [blocks, block_size, d]
        kv = kv.reshape(-1, kv.shape[-1])
    indices = torch.as_tensor(z["indices"])
    if indices.dim() == 3:
        indices = indices[:, 0, :]
    scale = float(z["scale"])
    meta = {}
    for k in ("num_tokens", "num_heads_impl", "kv_lora_rank", "q_shape",
              "q_stride", "indices_dtype", "q_dtype", "kv_dtype",
              "block_size", "topk_tokens", "out_ref", "out_ref_unpadded"):
        if k in z.files:
            meta[k] = z[k]
    return q, kv, indices, scale, meta


def rel_err(out, ref):
    err = (out.float() - ref.float()).abs().max().item()
    denom = ref.float().abs().max().item() + 1e-6
    return err, err / denom


def run_one(path, dev, bench, gate, cand_names):
    print(f"=== {path}")
    q, kv, indices, scale, meta = load_contract(path)
    q = q.to(dev)
    kv = kv.to(dev)
    indices = indices.to(dev)
    d_v = int(meta.get("kv_lora_rank", 512))

    idx_dtype = str(indices.dtype).replace("torch.", "")
    if "indices_dtype" in meta:
        idx_dtype = str(meta["indices_dtype"])
    print(f"contract: q{tuple(q.shape)} {q.dtype} stride={tuple(q.stride())}  "
          f"kv_rows={kv.shape[0]} {kv.dtype}  idx{tuple(indices.shape)} "
          f"{idx_dtype}  scale={scale:.5f}  num_tokens="
          f"{int(meta.get('num_tokens', q.shape[0]))}")

    n_heads = meta.get("num_heads_impl")
    if n_heads is not None:
        n_heads = int(n_heads)
        if q.shape[1] != n_heads:
            factor = q.shape[1] // n_heads
            print(f"WRAPPER-WARNING: q heads PADDED {n_heads}->{q.shape[1]} "
                  f"(repeat_interleave x{factor}); _forward_kv must return "
                  f"get_mla_unpadded_o = out[:, ::{factor}, :], NOT the raw "
                  f"kernel output. A candidate that early-returns here "
                  f"silently scrambles _v_up_proj's "
                  f"x.view(-1, {n_heads}, d) fold.")
        else:
            print(f"wrapper: q heads == num_heads_impl == {n_heads} (no padding)")

    t0 = time.time()
    ref = reference(q, kv, indices, scale, d_v)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt_ref = time.time() - t0

    if "out_ref" in meta:
        o = meta["out_ref"]
        o = torch.as_tensor(o).to(dev)
        err, rel = rel_err(ref, o)
        print(f"reference vs engine out_ref: max_abs={err:.5f} rel={rel:.2e}"
              f"  ({'MATCH' if rel < 1e-2 else 'MISMATCH - harness/contract ' 'problem!'})")
    if "out_ref_unpadded" in meta and n_heads is not None:
        o = torch.as_tensor(meta["out_ref_unpadded"]).to(dev)
        err, rel = rel_err(unpad(ref, n_heads), o)
        print(f"unpad(reference) vs out_ref_unpadded: rel={rel:.2e}")

    print(f"reference: {dt_ref*1000:.1f} ms")
    ok_all = True
    for name in cand_names:
        fn = CANDIDATES[name]
        t0 = time.time()
        out = fn(q, kv, indices, scale, d_v)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if bench > 1:
            t0 = time.time()
            for _ in range(bench - 1):
                out = fn(q, kv, indices, scale, d_v)
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / (bench - 1)
        err, rel = rel_err(out, ref)
        ok = rel < gate
        ok_all &= ok
        print(f"  {name:12s} {dt*1000:8.1f} ms  speedup {dt_ref/dt:5.2f}x  "
              f"max_abs={err:.5f} rel={rel:.2e}  "
              f"{'PASS' if ok else 'FAIL'} (gate {gate:.0e})")
    return ok_all


def main():
    ap = argparse.ArgumentParser(
        description="replay captured _forward_kv contracts against candidates")
    ap.add_argument("capture", nargs="+", help="npz from VLLM_GLM53_KC_DUMP")
    ap.add_argument("--bench", type=int, default=1,
                    help="timed iterations per candidate (default 1)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gate", type=float, default=1e-3,
                    help="rel-error PASS gate (default 1e-3)")
    ap.add_argument("--candidates", default="union,union_flat",
                    help=f"comma list from {sorted(CANDIDATES)}")
    args = ap.parse_args()

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    cand_names = [c for c in args.candidates.split(",") if c]
    for c in cand_names:
        if c not in CANDIDATES:
            sys.exit(f"unknown candidate {c!r}; have {sorted(CANDIDATES)}")

    ok = True
    for path in args.capture:
        ok &= run_one(path, dev, args.bench, args.gate, cand_names)
    print("REPLAY-PASS" if ok else "REPLAY-FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
