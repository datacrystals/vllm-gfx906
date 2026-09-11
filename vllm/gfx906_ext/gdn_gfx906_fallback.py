# SPDX-License-Identifier: Apache-2.0
# Track C — gfx906 GDN (gated delta-net) fallback implementation.
#
# Vendored in-tree as vllm/gfx906_ext/gdn_gfx906_fallback.py (supersedes
# /data/vllm-gfx906-dsv4/patches/gdn/gdn_gfx906_fallback.py, provenance only).
# Pure-PyTorch (fp32) reference path for the two GDN compute stages that
# misbehave on gfx906 under the bundled triton-gfx906 3.5.0:
#   1. the FLA chunked prefill pipeline (chunk_local_cumsum /
#      chunk_scaled_dot_kkt / solve_tril / wy_fast / chunk_delta_h / chunk_o),
#   2. the decode recurrence (fused_recurrent_gated_delta_rule_packed_decode /
#      fused_sigmoid_gating_delta_rule_update).
#
# Wiring (this module is vendored in the fork as vllm/gfx906_ext/; it
# supersedes /data/vllm-gfx906-dsv4/patches/gdn/, kept for provenance only):
#   (a) env-gated anchor at the tail of
#       vllm/model_executor/models/hy_v3.py: VLLM_GDN_GFX906_AUTOPATCH=1 ->
#       `from vllm.gfx906_ext import gdn_gfx906_fallback` +
#       `install_gfx906_gdn_fallback()` before the engine builds the model, or
#   (b) sitecustomize-style: import this module and call
#       `install_gfx906_gdn_fallback()` from a sitecustomize.py on PYTHONPATH
#       of the serve env.
#
# Correctness beats speed here: all math in fp32 states/activations, no
# tl.dot, no block ptrs. q/k/v casts are done on the fly; the SSM state
# cache is read/cast fp32 and written back in its cache dtype.
#
# Env knobs:
#   VLLM_GDN_GFX906_FALLBACK=auto|0|1   (auto = on iff triton arch is gfx906)
#   VLLM_GDN_GFX906_TRIL_SOLVE=triangular_solve|inv|loop  (default loop; the
#       two fast variants need on-GPU verification on gfx906)

import functools
import os

import torch

EPS_L2NORM = 1e-6
SOFTPLUS_THRESHOLD = 20.0


# ---------------------------------------------------------------------------
# enablement detection
# ---------------------------------------------------------------------------

@functools.cache
def _triton_arch() -> str | None:
    try:
        import triton

        tgt = triton.runtime.driver.active.get_current_target()
        return getattr(tgt, "arch", None) or (
            tgt[1] if isinstance(tgt, tuple) else str(tgt)
        )
    except Exception:
        return None


def fallback_enabled() -> bool:
    mode = os.environ.get("VLLM_GDN_GFX906_FALLBACK", "auto").strip().lower()
    if mode in ("1", "true", "on", "yes"):
        return True
    if mode in ("0", "false", "off", "no"):
        return False
    arch = _triton_arch()
    return arch == "gfx906"


# ---------------------------------------------------------------------------
# shared small helpers (mirror vllm/model_executor/layers/fla/ops/*.py math)
# ---------------------------------------------------------------------------

def _gating(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor,
            dt_bias: torch.Tensor):
    """fp32 g/beta from raw a,b projections.

    g = -exp(A_log) * softplus(a + dt_bias)   (log space)
    beta = sigmoid(b)
    shapes: a,b: [T, HV] -> g,beta: [T, HV] fp32
    """
    a = a.float()
    b = b.float()
    x = a + dt_bias.view(1, -1).float()
    # numerically safe softplus with threshold, mirror fused kernels:
    # softplus = log1p(exp(x)) where x <= 20 else x
    sp = torch.where(x <= SOFTPLUS_THRESHOLD,
                     torch.log1p(torch.exp(x.clamp(max=SOFTPLUS_THRESHOLD))), x)
    g = -torch.exp(A_log.view(1, -1).float()) * sp
    beta = torch.sigmoid(b)
    return g, beta


def _l2norm(x: torch.Tensor, eps: float = EPS_L2NORM) -> torch.Tensor:
    x32 = x.float()
    inv = torch.rsqrt((x32 * x32).sum(-1, keepdim=True) + eps)
    return x32 * inv


# ---------------------------------------------------------------------------
# decode recurrence (replaces fused_recurrent_gated_delta_rule_packed_decode)
# ---------------------------------------------------------------------------

_CALL_COUNTS = {"packed_decode": 0, "sigmoid_gating": 0, "chunk_prefill": 0}


def _tally(key):
    _CALL_COUNTS[key] += 1
    if _CALL_COUNTS[key] in (1, 2, 3, 10, 100, 1000):
        print(f"[GDN-FB] {key} call#{_CALL_COUNTS[key]}", flush=True)


def packed_decode_ref(*, mixed_qkv: torch.Tensor, a: torch.Tensor,
                      b: torch.Tensor, A_log: torch.Tensor,
                      dt_bias: torch.Tensor, scale: float,
                      initial_state: torch.Tensor, out: torch.Tensor,
                      ssm_state_indices: torch.Tensor,
                      use_qk_l2norm_in_kernel: bool = True):
    """One-token-per-seq GDN update in pure torch.

    Drop-in for fused_recurrent_gated_delta_rule_packed_decode()
    (vllm/model_executor/layers/fla/ops/fused_recurrent.py:339).

    mixed_qkv: [B, 2*H*K + HV*V] post-conv packed qkv (contiguous last dim)
    a, b:      [B, HV] raw gating projections
    initial_state: full SSM state cache [num_slots, HV, V, K] (fp16 or fp32)
    out:       [B, 1, HV, V]
    ssm_state_indices: [B] slot ids; slot id 0 (<=0) == NULL sentinel ->
                       emit zero output and skip touch (upstream semantics).
    """
    _tally("packed_decode")
    B, qkv_dim = mixed_qkv.shape
    HV, V, K = initial_state.shape[-3:]
    H = (qkv_dim - HV * V) // (2 * K)
    dev = mixed_qkv.device

    x = mixed_qkv.float()
    q = x[:, :H * K].view(B, H, K)
    k = x[:, H * K:2 * H * K].view(B, H, K)
    v = x[:, 2 * H * K:].view(B, HV, V)
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)
    q = q * scale
    g, beta = _gating(a, b, A_log, dt_bias)  # [B, HV] fp32

    # K-heads are shared across groups of (HV // H) V-heads.
    rep = HV // H
    q = q.repeat_interleave(rep, dim=1)   # [B, HV, K]
    k = k.repeat_interleave(rep, dim=1)   # [B, HV, K]

    idx = ssm_state_indices.long()        # [B]
    valid = idx > 0
    safe_idx = idx.clamp(min=0)
    S = initial_state[safe_idx].float()   # [B, HV, V, K]

    S = S * torch.exp(g).view(B, HV, 1, 1)
    # delta rule: dv = v - S @ k ; dv *= beta ; S += dv (x) k
    dv = v - torch.einsum("bhvk,bhk->bhv", S, k)
    dv = dv * beta.view(B, HV, 1)
    S = S + dv.unsqueeze(-1) * k.unsqueeze(-2)
    o = torch.einsum("bhvk,bhk->bhv", S, q)  # [B, HV, V]

    # write state back only for valid slots
    upd = valid.view(B, *([1] * (initial_state.dim() - 1)))
    # TODO(track-C): avoid the full-cache float() if cache is fp32 — index_put
    # per valid slot is fine since decode batch B is small.
    initial_state[safe_idx] = torch.where(
        upd, S.to(initial_state.dtype), initial_state[safe_idx])

    o = torch.where(valid.view(B, 1, 1), o, torch.zeros_like(o))
    out.copy_(o.view(B, 1, HV, V).to(out.dtype))
    return out, initial_state


def sigmoid_gating_update_ref(*, A_log: torch.Tensor, a: torch.Tensor,
                              b: torch.Tensor, dt_bias: torch.Tensor,
                              q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              beta: float = 1.0, threshold: float = 20.0,
                              scale: float = None, initial_state: torch.Tensor = None,
                              inplace_final_state: bool = True,
                              cu_seqlens: torch.Tensor | None = None,
                              ssm_state_indices: torch.Tensor | None = None,
                              num_accepted_tokens: torch.Tensor | None = None,
                              use_qk_l2norm_in_kernel: bool = False,
                              is_kda: bool = False,
                              **_ignored):
    """Decode-step replacement for fused_sigmoid_gating_delta_rule_update.

    Supports only the usage inside GatedDeltaNetAttention._forward_core:
    B=1 varlen decode rows (cu_seqlens step of 1 per seq), no spec decode.
    q,k: [1, T, H, K]  v: [1, T, HV, V]
    """
    # TODO(track-C): spec-decode path (num_accepted_tokens is not None) is NOT
    # implemented; upstream spec path is unused by current serving recipes
    # (max_num_seqs=1, no MTP in GDN models here). Raise loudly if hit.
    if num_accepted_tokens is not None:
        raise NotImplementedError("GDN gfx906 fallback: spec decode not supported")
    if is_kda:
        raise NotImplementedError("GDN gfx906 fallback: IS_KDA branch unused")
    _tally("sigmoid_gating")

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    assert B == 1
    if scale is None:
        scale = K ** -0.5

    qq = q[0].float()  # [T, H, K]
    kk = k[0].float()
    vv = v[0].float()  # [T, HV, V]
    if use_qk_l2norm_in_kernel:
        qq = _l2norm(qq)
        kk = _l2norm(kk)
    qq = qq * scale

    aa = a[:T].float()  # [T, HV]
    bb = b[:T].float()
    gg, bbeta = _gating(aa, bb, A_log, dt_bias)  # [T, HV]

    starts = cu_seqlens[:-1].tolist()
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    assert all(l == 1 for l in lens), (
        "sigmoid_gating_update_ref only supports 1-token/seq steps")
    idx = ssm_state_indices
    if idx.ndim > 1:
        idx = idx[:, 0]
    idx = idx.long()
    valid = idx > 0
    safe_idx = idx.clamp(min=0)

    qg = qq.repeat_interleave(HV // H, dim=1)
    kg = kk.repeat_interleave(HV // H, dim=1)

    S = initial_state[safe_idx].float()          # [N, HV, V, K]
    S = S * torch.exp(gg)[:, :, None, None]
    dv = vv - torch.einsum("nhvk,nhk->nhv", S, kg)
    dv = dv * bbeta[:, :, None]
    S = S + dv.unsqueeze(-1) * kg.unsqueeze(-2)
    o = torch.einsum("nhvk,nhk->nhv", S, qg)     # [N, HV, V]

    upd = valid.view(-1, *([1] * (initial_state.dim() - 1)))
    initial_state[safe_idx] = torch.where(
        upd, S.to(initial_state.dtype), initial_state[safe_idx])
    o = torch.where(valid.view(-1, 1, 1), o, torch.zeros_like(o))
    o = o.view(1, T, HV, V).to(initial_state.dtype)
    return o, initial_state if inplace_final_state else initial_state.clone()


# ---------------------------------------------------------------------------
# chunked prefill (replaces ChunkGatedDeltaRule.forward_native / FLA pipeline)
# ---------------------------------------------------------------------------

def _invert_unit_lower(M: torch.Tensor) -> torch.Tensor:
    """Compute (I + A)^{-1} for batched [..., n, n] lower-triangular M with
    unit diagonal. n <= 64 here.

    VLLM_GDN_GFX906_TRIL_SOLVE selects the strategy:
      loop             - forward substitution, 63 steps of tiny batched ops
                         (default; zero torch-cublas/hipsolver risk).
      triangular_solve - torch.linalg.solve_triangular (VERIFY on gfx906).
      inv              - torch.linalg.inv (VERIFY on gfx906).
    """
    mode = os.environ.get("VLLM_GDN_GFX906_TRIL_SOLVE", "loop")
    n = M.shape[-1]
    if mode == "triangular_solve":
        I = torch.eye(n, device=M.device, dtype=M.dtype).expand(M.shape[:-2] +
                                                                (n, n))
        return torch.linalg.solve_triangular(M, I, upper=False,
                                             unitriangular=True)
    if mode == "inv":
        return torch.linalg.inv(M)
    # loop: classical forward substitution on rows of X, solving (I+L)X = I.
    # Row i of (I+L)^{-1}: X[i] = I[i] - sum_{j<i} L[i,j] X[j].
    # Eager torch slice writes; fine for prefill fallback. TODO(track-C):
    # if this turns out to be the prefill bottleneck (63 sequential iters x
    # tiny ops x NC x layer count), replace with one of the verified modes.
    X = torch.zeros_like(M)
    off = M.tril(diagonal=-1)  # strictly lower part
    X[..., 0, 0] = 1.0
    for i in range(1, n):
        acc = (off[..., i:i + 1, :i] @ X[..., :i, :])  # [..., 1, n]
        X[..., i:i + 1, :] -= acc
        X[..., i, i] += 1.0
    return X


def _chunked_gdn_fwd_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         g: torch.Tensor, beta: torch.Tensor, scale: float,
                         initial_state: torch.Tensor | None,
                         output_final_state: bool):
    """Chunked gated delta rule, fp32, mirroring the FLA kernel math.

    q,k: [T, H, K]  v: [T, HV, V]  g,beta: [T, HV] fp32
    initial_state: [H? -> HV, V, K] fp32 or None (single sequence)
    returns o: [T, HV, V] fp32, final_state [HV, V, K] fp32 or None
    """
    T, H, K = k.shape
    HV, V = v.shape[1], v.shape[2]
    CT = 64
    rep = HV // H
    dev = q.device

    q = q.float().repeat_interleave(rep, dim=1)  # [T, HV, K]
    k = k.float().repeat_interleave(rep, dim=1)  # [T, HV, K]
    v = v.float()
    g = g.float()
    beta = beta.float()

    pad = (-T) % CT
    if pad:
        pad_t = lambda t, val=0.0: torch.cat(
            [t, t.new_zeros(pad, *t.shape[1:])], 0)
        q, k, v, g, beta = (pad_t(x) for x in (q, k, v, g, beta))
    Tp = T + pad
    NC = Tp // CT

    q = q.view(NC, CT, HV, K)
    k = k.view(NC, CT, HV, K)
    v = v.view(NC, CT, HV, V)
    g = g.view(NC, CT, HV)
    beta = beta.view(NC, CT, HV)
    gc = g.cumsum(dim=1)                        # [NC, CT, HV]

    # A[t,s] = sum_k k_t k_s * beta_t * e^{gc_t - gc_s}, t > s
    dgc = gc.unsqueeze(2) - gc.unsqueeze(1)     # [NC, CT(t), CT(s), HV]
    kk = torch.einsum("nthk,nshk->ntsh", k, k)  # [NC, CT, CT, HV]
    A = kk * beta.unsqueeze(2) * torch.exp(dgc)
    tri = torch.arange(CT, device=dev)
    strict_lower = (tri[:, None] > tri[None, :]).view(1, CT, CT, 1)
    A = torch.where(strict_lower, A, torch.zeros_like(A))
    # invert per (chunk, head): [NC, HV, CT, CT]
    Mp = (torch.eye(CT, device=dev, dtype=A.dtype).view(1, 1, CT, CT) +
          A.permute(0, 3, 1, 2))
    Ai = _invert_unit_lower(Mp).permute(0, 2, 3, 1)  # [NC, CT(t), CT(s), HV]

    # w = Ai @ (k * beta * e^{gc}); u = Ai @ (v * beta)   (sum over s)
    kb = k * beta.unsqueeze(-1) * torch.exp(gc).unsqueeze(-1)  # [NC,CT,HV,K]
    w = torch.einsum("ntsh,nshk->nthk", Ai, kb)
    u = torch.einsum("ntsh,nshv->nthv", Ai, v * beta.unsqueeze(-1))

    S = (initial_state.float() if initial_state is not None
         else torch.zeros(HV, V, K, device=dev))

    o = torch.empty(NC, CT, HV, V, device=dev)
    lower_incl = (tri[:, None] >= tri[None, :]).view(CT, CT, 1)
    G = gc[:, -1, :]                            # [NC, HV]
    for c in range(NC):
        Sc = S                                  # [HV, V, K]
        # v_new = u - w @ S  (kernel order: store BEFORE within-chunk decay)
        v_new = u[c] - torch.einsum("thk,hvk->thv", w[c], Sc)
        # intra-chunk attention: A_intra[t,s] = q_t.k_s * e^{gc_t-gc_s}, t>=s
        qk = torch.einsum("thk,shk->tsh", q[c], k[c])       # [CT, CT, HV]
        Ain = qk * torch.exp(dgc[c])                        # [CT, CT, HV]
        Ain = torch.where(lower_incl, Ain, torch.zeros_like(Ain))
        # o = (q @ S^T) * e^{gc} + Ain @ v_new
        o_c = (torch.einsum("thk,hvk->thv", q[c], Sc) *
               torch.exp(gc[c]).unsqueeze(-1))
        o_c = o_c + torch.einsum("tsh,shv->thv", Ain, v_new)
        o[c] = o_c
        # state update: S = e^G S + sum_t e^{G-gc_t} k_t (x) v_new_t
        decay = torch.exp(G[c] - gc[c]).unsqueeze(-1)       # [CT, HV, 1]
        S = Sc * torch.exp(G[c]).view(HV, 1, 1) + torch.einsum(
            "thk,thv->hvk", k[c] * decay, v_new)
    if pad:
        o = o.view(Tp, HV, V)[:T]
    else:
        o = o.view(T, HV, V)
    return (o * scale), (S if output_final_state else None)


def chunk_gated_delta_rule_ref(*, q: torch.Tensor, k: torch.Tensor,
                               v: torch.Tensor, g: torch.Tensor,
                               beta: torch.Tensor, scale: float = None,
                               initial_state: torch.Tensor = None,
                               output_final_state: bool = False,
                               cu_seqlens: torch.Tensor | None = None,
                               chunk_indices: torch.Tensor | None = None,
                               chunk_offsets: torch.Tensor | None = None,
                               use_qk_l2norm_in_kernel: bool = False):
    """Drop-in for fla chunk_gated_delta_rule (chunk.py:129).

    q,k,v: [1, T, H, K-or-V]; g,beta: [1, T, HV] fp32 log/decay
    initial_state: [N, HV, V, K]
    returns o: [1, T, HV, V] (model dtype), final_state [N, HV, V, K] fp32.
    """
    _tally("chunk_prefill")
    assert q.shape[0] == 1, "fallback expects flattened varlen batch"
    H, K = q.shape[2], q.shape[3]
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K ** -0.5
    model_dtype = q.dtype

    qq = q[0].float()
    kk = k[0].float()
    vv = v[0].float()
    if use_qk_l2norm_in_kernel:
        qq = _l2norm(qq)
        kk = _l2norm(kk)
    gg = g[0].float()
    bb = beta[0].float()

    if cu_seqlens is None:
        lens = [q.shape[1]]
    else:
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    o = torch.empty(q.shape[1], HV, V, dtype=torch.float32, device=q.device)
    final = [] if output_final_state else None
    off = 0
    for n, L in enumerate(lens):
        st = initial_state[n] if initial_state is not None else None
        o_n, S_n = _chunked_gdn_fwd_ref(
            qq[off:off + L], kk[off:off + L], vv[off:off + L],
            gg[off:off + L], bb[off:off + L], scale, st, output_final_state)
        o[off:off + L] = o_n
        if output_final_state:
            final.append(S_n)
        off += L
    o = o.unsqueeze(0).to(model_dtype)
    if output_final_state:
        return o, torch.stack(final, 0).float()
    return o, None


# ---------------------------------------------------------------------------
# installer: monkeypatch guards
# ---------------------------------------------------------------------------

def install_gfx906_gdn_fallback() -> bool:
    """Patch dsv4 vllm GDN layer to route around gfx906-broken triton kernels.
    Returns True if patches were applied. Idempotent."""
    if not fallback_enabled():
        return False
    if getattr(install_gfx906_gdn_fallback, "_installed", False):
        return True

    from vllm.logger import init_logger
    logger = init_logger("gdn_gfx906_fallback")

    from vllm.model_executor.layers.mamba import gdn_linear_attn as gla

    # --- decode recurrence --------------------------------------------------
    gla.fused_recurrent_gated_delta_rule_packed_decode = packed_decode_ref
    gla.fused_sigmoid_gating_delta_rule_update = sigmoid_gating_update_ref

    # Optional: replace the ~30-op eager recurrence with a single fused
    # triton kernel per GDN layer per decode step (Track P1, env-gated).
    if os.environ.get("VLLM_GDN_GFX906_FUSED_DECODE", "0") == "1":
        try:
            from vllm.gfx906_ext import gdn_decode_fused as _fz
            _fz.install_fused_decode()
            logger.info_once("GDN gfx906 fused decode kernel enabled")
        except Exception:
            logger.exception("fused decode unavailable; keeping eager "
                             "torch fallback")

    # --- chunked prefill ----------------------------------------------------
    # ChunkGatedDeltaRule dispatches through CustomOp machinery; swapping
    # the bound method on the class is enough (forward_native is the
    # non-flashinfer branch, always taken on ROCm).
    def _forward_native_patched(self, **kwargs):
        return chunk_gated_delta_rule_ref(**kwargs)

    gla.ChunkGatedDeltaRule.forward_native = _forward_native_patched

    if os.environ.get("VLLM_GDN_GFX906_NAN_PROBE", "0") == "1":
        _install_nan_probe(logger)

    # --- shared-expert MLP safety: clamp the silu·mul output so the
    # GPTQ down-proj internal fp16 cast can't overflow/NaN-poison the stream
    # (env VLLM_GFX906_MLP_CLAMP=1, value float overrides bound; 0 disables)
    clamp_env = os.environ.get("VLLM_GFX906_MLP_CLAMP", "0")
    try:
        clamp_val = float(clamp_env)
    except ValueError:
        clamp_val = 0.0
    if clamp_val != 0.0:
        _install_mlp_clamp(clamp_val)

    # --- shared-expert down_proj fp32 bypass (VLLM_GFX906_MLP_FP32_DOWN=1) --
    # exact fp32 dequant GEMM replacing the fp16-saturating gptq_gemm; the
    # principled fix where the clamp above would distort large activations.
    if os.environ.get("VLLM_GFX906_MLP_FP32_DOWN", "0") == "1":
        _install_mlp_fp32_down()

    logger.info_once(
        "GDN gfx906 fallback ACTIVE: pure-torch fp32 chunk prefill + decode "
        "recurrence (VLLM_GDN_GFX906_FALLBACK=%s)",
        os.environ.get("VLLM_GDN_GFX906_FALLBACK", "auto"))
    install_gfx906_gdn_fallback._installed = True
    return True


# ---------------------------------------------------------------------------
# NaN probe: global module forward hook (VLLM_GDN_GFX906_NAN_PROBE=1)
# ---------------------------------------------------------------------------

_NAN_PROBE_STATE = None


def _iter_float_tensors(obj):
    import torch as _t
    if isinstance(obj, _t.Tensor):
        if obj.is_floating_point():
            yield obj
    elif isinstance(obj, (tuple, list)):
        for it in obj:
            yield from _iter_float_tensors(it)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_float_tensors(v)


def _iter_mod_params(mod):
    """Yield (name, tensor) for direct float/int params incl. GPTQ packed."""
    for pname, p in mod.named_parameters(recurse=False):
        yield pname, p.data


# ---------------------------------------------------------------------------
# Shared-expert MLP clamp fix (VLLM_GFX906_MLP_CLAMP=<bound>)
# ---------------------------------------------------------------------------

def _install_mlp_clamp(bound: float):
    """Clamp silu·mul activations (and x) inside Qwen2MoeMLP/Qwen3NextMLP so
    the GPTQLinear down-proj internal fp16 cast cannot overflow to inf -> NaN
    in gptq_gemm. Shared expert on gfx906 with --dtype float32 is the known
    trigger (see EXPERIMENTS.md 2026-09-09 late evening)."""
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    def forward_patched(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out = out.clamp(-bound, bound)
        out, _ = self.down_proj(out)
        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out
        return out

    Qwen2MoeMLP.forward = forward_patched
    print(f"[GDN gfx906 fix] MLP clamp bound={bound} installed", flush=True)


# ---------------------------------------------------------------------------
# Shared-expert down_proj fp32 fix (VLLM_GFX906_MLP_FP32_DOWN=1)
#
# Root cause, pinned 2026-09-10 (probe round 3 + offline A/B, see
# EXPERIMENTS.md): gptq_gemm (csrc/quantization/gptq/q_gemm.cu) rounds each
# fp32 k-block partial to fp16 on the way into the fp16 output buffer
# (atomicAdd of half2). With --dtype float32 the shared-expert silu*mul
# activations legitimately reach ~5.6e3, and the L22 down_proj partial at
# [token0, ch3055] exceeds 65504 -> +/-inf -> spreads as NaN from the next
# RMSNorm on. The linear-class GPTQ weights (shared expert, o/qkv projs) are
# the only users of gptq_gemm here; FusedMoE uses moe_wna16 (true w4a32).
# Fix: for Qwen2MoeMLP instances ONLY, replace the down_proj gptq_gemm call
# with an fp32 matmul against a cached fp32 dequant of the (exllama-shuffled)
# packed weight. Numerically faithful (fp16 GEMM rounding removed, not
# emulated), weight values identical to what the kernel computes.
# ---------------------------------------------------------------------------

def _dequant_gptq_row_fp32(layer):
    """Build+cache fp32 [K_shard, N] weight from a GPTQ RowParallelLinear's
    packed runtime tensors (post vllm exllama shuffle). None if unsupported
    layout (caller falls back to the stock path)."""
    W = getattr(layer, "_gfx906_w_fp32", None)
    if W is not None or getattr(layer, "_gfx906_w_fp32_bad", False):
        return W
    import torch
    try:
        qw = layer.qweight.data          # [K/8, N] int32, shuffled nibbles
        qz = layer.qzeros.data           # [G, N/8] int32 (not shuffled)
        sc = layer.scales.data           # [G, N]
        g_idx = layer.g_idx.data
        if qw.dtype != torch.int32 or qz.dtype != torch.int32:
            return None
        K = qw.shape[0] * 8
        N = qw.shape[1]
        G = sc.shape[0]
        if K % G != 0 or qz.shape[0] != G or qz.shape[1] * 8 != N:
            return None
        if g_idx.numel() > 0 and (g_idx.numel() != K or int(g_idx.max()) >= G):
            return None
        gs = K // G
        dev = qw.device
        ar8 = torch.arange(8, device=dev, dtype=torch.int32)
        nib = (qw.unsqueeze(-1) >> (ar8 * 4)) & 0xF          # [K/8, N, 8] post
        # invert shuffle_4bit_8: orig feature f <- post nibble p(f)
        p = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)
        q = nib[..., p].permute(0, 2, 1).reshape(K, N).float()  # [K, N]
        zn = (qz.unsqueeze(-1) >> (ar8 * 4)) & 0xF            # [G, N/8, 8]
        z = zn.reshape(G, N).float()
        use_v2 = bool(getattr(layer.quant_method, "use_v2_format", False))
        z_eff = z if use_v2 else (z + 1.0)
        if g_idx.numel() > 0:
            z_eff = z_eff[g_idx.long()]                     # [K, N]
            scf = sc.float()[g_idx.long()]
        else:
            z_eff = z_eff.repeat_interleave(gs, 0)
            scf = sc.float().repeat_interleave(gs, 0)
        W = (q - z_eff) * scf.contiguous()                  # [K, N] fp32
    except Exception as e:
        print(f"[GDN gfx906 fix] dequant failed ({e}); stock path", flush=True)
        layer._gfx906_w_fp32_bad = True
        return None
    layer._gfx906_w_fp32 = W
    return W


def _install_mlp_fp32_down():
    """Route Qwen2MoeMLP down_proj (shared expert of Qwen3Next/Qwen3.5 MoE)
    through an fp32 dequant GEMM instead of the fp16-output gptq_gemm."""
    import torch
    import torch.nn.functional as F
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    def forward_patched(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        down = self.down_proj
        W = None
        if type(down).__name__ == "RowParallelLinear" and \
                hasattr(down, "qweight") and \
                type(down.quant_method).__name__ == "GPTQLinearMethod":
            W = _dequant_gptq_row_fp32(down)
        if W is not None:
            y = out.float() @ W
            if down.reduce_results and down.tp_size > 1:
                y = tensor_model_parallel_all_reduce(y)
            out = y.to(x.dtype)
        else:
            out, _ = down(out)
        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out
        return out

    Qwen2MoeMLP.forward = forward_patched
    print("[GDN gfx906 fix] MLP fp32 down_proj bypass installed", flush=True)


def _install_nan_probe(logger):
    """Global nn.Module forward hook: find the FIRST module output that goes
    non-finite, print module type + decoder-layer index + NaN locations.
    Uses print() because vllm logger output didn't surface from the anchor
    context. Decoder-layer index tracked via pre-hook ordering."""
    global _NAN_PROBE_STATE
    _NAN_PROBE_STATE = {"cur_layer": None, "layer_counter": 0, "seen": set()}
    state = _NAN_PROBE_STATE
    import torch

    WATCH = ("DecoderLayer", "Embedding", "LMHead", "LogitsProcessor",
             "GatedDeltaNetAttention", "Attention", "MoeBlock", "SparseMoe",
             "RMSNorm", "FusedMoE", "MLP", "ReplicatedLinear",
             "ColumnParallelLinear", "RotaryEmbedding", "RowParallelLinear",
             "SiluAndMul")

    def _pre(mod, inputs):
        tname = type(mod).__name__
        if tname in ("Qwen3_5Model", "Qwen3_5_MoeModel", "Qwen3NextModel"):
            state["layer_counter"] = 0
        elif tname.endswith("DecoderLayer"):
            state["cur_layer"] = state["layer_counter"]
            state["layer_counter"] += 1

    def _fmt_idx(t, n=6):
        bad = ~torch.isfinite(t)
        idx = torch.nonzero(bad)
        return idx[:n].cpu().tolist()

    def _hook(mod, inputs, output):
        tname = type(mod).__name__
        if not any(w in tname for w in WATCH):
            return
        try:
            # isfinite().item() is a D2H sync — illegal during cudagraph
            # capture; probe only observes eager (non-captured) execution.
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return
        for t in _iter_float_tensors(output):
            finite = torch.isfinite(t).all()
            if not finite.item():
                amax = t.nan_to_num(0.0, 0.0, 0.0).abs().max().item()
                nbad = (~torch.isfinite(t)).sum().item()
                print(f"[NANPROBE] NONFINITE module={tname} "
                      f"layer={state['cur_layer']} shape={tuple(t.shape)} "
                      f"nbad={nbad} amax={amax:.6g} "
                      f"idx={_fmt_idx(t)} "
                      f"first={tname not in state['seen']}", flush=True)
                if tname not in state["seen"]:
                    state["seen"].add(tname)
                    print(f"[NANPROBE]  inputs_finite="
                          f"{[bool(torch.isfinite(i).all().item()) for i in _iter_float_tensors(inputs)]} "
                          f"inputs_absmax="
                          f"{[float(i.nan_to_num(0.0, 0.0, 0.0).abs().max().item()) for i in _iter_float_tensors(inputs)]}",
                          flush=True)
                    # weight stats for the first offender (GPTQ attrs etc.)
                    for pname, p in _iter_mod_params(mod):
                        if not pname.startswith("_"):
                            try:
                                fin = bool(torch.isfinite(p.float()).all().item())
                                am = float(p.float().nan_to_num(0.0, 0.0, 0.0).abs().max().item()) \
                                    if p.is_floating_point() else float(p.abs().max().item())
                                print(f"[NANPROBE]  weight {tname}.{pname} "
                                      f"finite={fin} amax={am:.5g}", flush=True)
                            except Exception:
                                pass
                return

    torch.nn.modules.module.register_module_forward_pre_hook(_pre)
    torch.nn.modules.module.register_module_forward_hook(_hook)
    print("[NANPROBE] global forward+pre hooks installed", flush=True)
    return _hook


if os.environ.get("VLLM_GDN_GFX906_AUTOPATCH", "0") == "1":
    install_gfx906_gdn_fallback()

if os.environ.get("VLLM_GFX906_PROF_DIR"):
    try:
        from vllm.gfx906_ext import prof_patch  # noqa: F401  (self-installs; see prof_patch.py)
    except Exception as _e:
        print(f"[PROF] install skipped: {_e}", flush=True)
