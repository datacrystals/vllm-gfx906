"""patch_prefill_union.py — wire union-GEMM prefill into rocm_aiter_mla_sparse.

GLM53-PREFILL: replaces the reference_mla_sparse_prefill gather+bmm core with
union-of-indices GEMM + scatter (measured 2.06x on the attention core at
topk=2048, rel 0.0004). Env-gated: VLLM_GLM53_PREFILL_UNION=1 (default off =
production behavior unchanged).

Also honors VLLM_GLM53_INDEX_TOPK (already installed) — the two stack.

GLM53-UNION-UNPAD-FIX (2026-09-30): the union path must NOT early-return from
_forward_kv. The caller pads the head dim (AiterMLAHelper.get_mla_padded_q
repeat_interleaves heads when num_heads < 16 — GLM-5.3 on TP8 is 8 heads/rank
padded to 16) and every return path must fall through get_mla_unpadded_o.
Early-returning the raw kernel output made _v_up_proj fold [s_q,16,d_v] as
[2*s_q,8,d_v] and scramble the layer output (needle retrieval died while
generation stayed fluent). See PREFILL_KERNEL_PLAN.md for the full story.

Usage: python3 patch_prefill_union.py <rocm_aiter_mla_sparse.py>
"""
import sys

path = sys.argv[1]
src = open(path).read()

if "GLM53-PREFILL-UNION" in src:
    print("already patched:", path)
    sys.exit(0)

UNION_FN = '''

def union_gather_prefill(
    q: torch.Tensor,       # [s_q, h_q, d_qk]
    kv: torch.Tensor,      # [total_kv, 1, d_qk] (or [total_kv, d_qk])
    indices: torch.Tensor, # [s_q, 1, topk] (or [s_q, topk])
    sm_scale: float,
    d_v: int,
) -> torch.Tensor:
    """GLM53-PREFILL-UNION: attention over unique index rows with scatter.

    S_u[t,h,u] = scale * q[t,h] . KU[u];   S[t,h,j] = S_u[t,h,inv[t,j]]
    P = softmax(S over j);  out[t,h] = P_scatter[t,h,:] @ KU[:, :d_v]

    Semantics match reference_mla_sparse_prefill: invalid indices contribute
    nothing (dedicated zero row), duplicate indices within a row count
    once each (scatter_add accumulates them onto the row — same total).
    Returns [s_q, h_q, d_v] in q.dtype; the caller (_forward_kv) must still
    apply AiterMLAHelper.get_mla_unpadded_o to un-pad the head dim.
    """
    if indices.dim() == 3:
        indices = indices[:, 0, :]
    if kv.dim() == 3:
        kv = kv[:, 0, :]
    s_q, h_q, d_qk = q.shape
    if s_q > 64:
        # GLM53: chunk so the [T,H,U] score tensor stays bounded
        outs = []
        for s0 in range(0, s_q, 64):
            s1 = min(s0 + 64, s_q)
            outs.append(union_gather_prefill(
                q[s0:s1], kv[:, None, :], indices[s0:s1], sm_scale, d_v))
        return torch.cat(outs, dim=0)
    topk = indices.shape[1]
    s_kv = kv.shape[0]

    invalid = (indices < 0) | (indices >= s_kv)
    safe = indices.masked_fill(invalid, 0)

    u_idx, inv = torch.unique(safe.reshape(-1), return_inverse=True)
    # GLM53-BUGFIX: engine passes int32 indices; scatter_add_ with int32
    # index corrupts silently on ROCm (micro-tests passed only because
    # randint gives int64). Force int64 everywhere downstream.
    inv = inv.view(s_q, topk).to(torch.int64)
    ku = kv.index_select(0, u_idx).contiguous()          # [U, d_qk]

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

'''

# 1. insert the function before reference_mla_sparse_prefill
anchor = "def reference_mla_sparse_prefill("
assert anchor in src
src = src.replace(anchor, UNION_FN + anchor, 1)

# 2. route _forward_kv's prefill through it when env is on.
# GLM53-UNION-UNPAD-FIX: assign `output` and fall through the common
# get_mla_unpadded_o return — never early-return the raw kernel output.
old = """        if envs.VLLM_ROCM_MLA_SPARSE_FP16:
            # Force ref Torch (instead of using mla_sparse) as triton is still slower than chunked torch (1.5 vs 8 TFLOPS) and not steady enough (HSA_STATUS_ERROR_OUT_OF_RESOURCES when running with max-num-batched-tokens 8192)
            output = reference_mla_sparse_prefill("""
new = """        if envs.VLLM_ROCM_MLA_SPARSE_FP16:
            # GLM53-PREFILL-UNION: env-gated union-GEMM path (measured 2.06x
            # on the attention core at topk=2048). Default off = reference.
            # GLM53-UNION-UNPAD-FIX: do NOT early-return the union output.
            # The caller pads the head dim (AiterMLAHelper.get_mla_padded_q
            # repeat_interleaves heads when num_heads < 16: GLM-5.3 TP8 is
            # 64 heads -> 8/rank -> 16 padded), and every path must fall
            # through get_mla_unpadded_o below. Returning the raw 16-head
            # tensor made _v_up_proj fold [s_q, 16, d_v] as [2*s_q, 8, d_v]
            # and scramble the layer output (needle quality died while
            # generation stayed fluent).
            if (os.environ.get("VLLM_GLM53_PREFILL_UNION", "0") == "1"
                    and not torch.cuda.is_current_stream_capturing()):
                output = union_gather_prefill(
                    q,
                    kv_c_and_k_pe_cache.view(-1, 1, kv_c_and_k_pe_cache.shape[-1]),
                    topk_indices.view(num_tokens, 1, -1),
                    self.softmax_scale,
                    self.kv_lora_rank,
                )
            else:
                output = reference_mla_sparse_prefill("""
assert old in src
src = src.replace(old, new, 1)

# 3. ensure os import
if "\nimport os\n" not in src:
    src = src.replace("import torch", "import os\n\nimport torch", 1)

open(path, "w").write(src)
print("patched:", path)
