# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash track: hand-built skinny int4 (CT pack-quantized GPTQ v2,
# group_size>=32) GEMV for gfx906 (MI50, no MFMA), plus an optional M==1
# dense bf16/fp16 GEMV redirect for the aiter `LLGemm1` calls.
#
# Why (trace attribution, /data/llmbench/glm53-prof2 rank0, ONE decode step;
# analysis scripts in /data/llmbench/glm53-prof3):
#   * `vllm::gptq::gemm_half_q_half_gptq_4bit_kernel<true,1>` runs x142/step
#     (4.5 ms/step, avg 31.6 us): the CT int4 dense linears
#       - KDA q/k/v_proj      [N=1024, K=4096] x90 (30 KDA-MoE layers x3)
#       - KDA o_proj          [N=4096, K=1024] x30
#       - MLA q_b_proj        [N=2048, K=1536] x11
#       - MLA o_proj          [N=4096, K=2048] x11
#     Launched via ExllamaLinearKernel.apply_weights -> ops.gptq_gemm
#     (kernels/linear/mixed_precision/exllama.py:153-177, graphics grid =
#     (N/1024, 1, K/256): 2-16 CTAs per launch on a 60-CU gfx906 -> ~12x off
#     the HBM roofline (packed int4 weights ~350 MB/step -> ~0.4-0.9 ms).
#   * `LLGemm1_kernel<Half,4>` x362/step (3.5 ms/step, avg 9.7 us): all fp16
#     dense GEMVs at M==1 via rocm_unquantized_gemm_impl -> ops.LLMM1
#     (layers/utils.py:374-381). The bf16 dense set per step: KDA
#     b/f_a/f_b/g_a/g_b (+q/k/v/o on unquantized layers), MLA q_a/kv_a/
#     indexer wq_b/wk|weights_proj, MoE router gate [288,4096] bf16,
#     shared-expert gate_up/down, dense-MLP gate_up/down (layers 0-2),
#     lm_head [19360, 4096]. The existing VLLM_GFX906_GEMV override only
#     intercepts `triton_matmul` (M=2..16); the n==1 branch never hits it.
#
# This module:
#   * `int4_gemv_m(x, w_q, qzeros, scales) -> y[M,N]` — Triton kernel,
#     pure FMA fp32-accumulation (matching the CUDA kernel's fp32 acc),
#     no tl.dot, num_warps<=4, tiles <= 4096 elems (gfx906-safe).
#     Grid = cdiv(N, BLOCK_N): every packed weight byte is read exactly once.
#   * `install_glm53_int4_gemv()` — wraps ExllamaLinearKernel.apply_weights;
#     only decode shapes (M<=8, fp16 x, 4-bit, v2 semantics, empty g_idx,
#     no bias) take the Triton path; everything else falls through to the
#     original gptq_gemm call. MoE wna16 experts are NOT affected (different
#     method class).
#   * `install_glm53_dense_gemv()` — optional; wraps vllm._custom_ops.LLMM1
#     so the n==1 fp16 dense branch uses gfx906_gemv.gemv_m (fp32 accum,
#     fp16 out; same numerics class as LLGemm1's fp32 warp-shuffle reduce).
#
# Env gates (default OFF), mirroring the gfx906_gemv convention
# (install anchors at the tail of models/glm5next/__init__.py):
#   VLLM_GLM53_INT4_GEMV=1        int4 dense linears -> triton GEMV
#   VLLM_GLM53_INT4_GEMV_MAX_TOKENS   decode-shape threshold (default 8)
#   VLLM_GLM53_DENSE_GEMV=1       ops.LLMM1 M==1 fp16 calls -> gemv_m
#
# Numerics: w = (q - zp) * s with 4-bit q/zp unpacked exactly as the CUDA
# kernel's MatrixView semantics (qweight int32 [K/8, N] k-ascending nibbles,
# qzeros int32 [groups, N/8] column-packed nibbles, scales fp16 [groups, N],
# group(k) = k // group_size, v2 => raw zero points, no +1). Accumulation in
# fp32 elementwise per output column; store fp16. Offline CPU check:
# /data/llmbench/glm53-prof3/check_glm53_int4_gemv.py (bit-exact unpack math
# vs vLLM's unpack_quantized_values_into_int32 + direct matmul reference).
# GPU correctness+time A/B:
# /data/llmbench/glm53-prof3/bench_glm53_int4_gemv.py (vs ops.gptq_gemm).

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

_ENV_INT4 = "VLLM_GLM53_INT4_GEMV"
_ENV_INT4_MAXT = "VLLM_GLM53_INT4_GEMV_MAX_TOKENS"
_ENV_DENSE = "VLLM_GLM53_DENSE_GEMV"
_MAX_TOKENS_DEFAULT = 8


def int4_gemv_enabled() -> bool:
    return os.environ.get(_ENV_INT4, "0").strip().lower() in ("1", "true", "on")


def dense_gemv_enabled() -> bool:
    return os.environ.get(_ENV_DENSE, "0").strip().lower() in ("1", "true", "on")


def int4_gemv_max_tokens() -> int:
    try:
        return int(os.environ.get(_ENV_INT4_MAXT, str(_MAX_TOKENS_DEFAULT)))
    except ValueError:
        return _MAX_TOKENS_DEFAULT


if HAS_TRITON:

    def _autotune_configs():
        cfgs = []

        def add(bk, bn, warps):
            # tile elems kept <= 4096; num_warps <= 4 (gfx906 GEMV class)
            if bk * bn <= 4096 and warps <= 4:
                cfgs.append(
                    triton.Config(
                        {"BLOCK_K": bk, "BLOCK_N": bn},
                        num_warps=warps,
                        num_stages=1,
                    ))

        add(32, 32, 1)
        add(32, 64, 2)
        add(64, 64, 2)
        add(128, 32, 2)
        add(32, 128, 4)
        add(64, 64, 4)
        return cfgs

    @triton.autotune(configs=_autotune_configs(), key=["N", "K"])
    @triton.jit
    def _glm53_int4_gemv_kernel(
        x_ptr,  # [M, K] fp16 (reduced to BLOCK_M=next_pow2(M) rows)
        qw_ptr,  # [K/8, N] int32/uint32 (packed 4-bit, k ascending nibbles)
        qz_ptr,  # [G, N/8] int32/uint32 (packed 4-bit zp, n ascending nibbles)
        sc_ptr,  # [G, N] fp16
        y_ptr,  # [M, N] fp16
        M,
        GSIZE,
        N,
        K,
        stride_xm,
        stride_y_m,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_rows = k0 + offs_k  # [BLOCK_K] absolute k
            # packed weight rows: each uint32 covers 8 k for one column n
            pk = tl.load(
                qw_ptr + (k_rows // 8)[:, None].to(tl.int64) * N
                + offs_n[None, :],
                mask=n_mask[None, :] & (k_rows < K)[:, None],
                other=0,
            )
            q = (pk >> ((k_rows % 8) * 4)[:, None]) & 0xF  # [BLOCK_K, N] int

            g = k_rows // GSIZE  # [BLOCK_K]
            kmask = k_rows < K
            s = tl.load(
                sc_ptr + g[:, None].to(tl.int64) * N + offs_n[None, :],
                mask=n_mask[None, :] & kmask[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_K, N]
            zw = tl.load(
                qz_ptr
                + g[:, None].to(tl.int64) * (N // 8)
                + (offs_n // 8)[None, :],
                mask=n_mask[None, :] & kmask[:, None],
                other=0,
            )
            zp = (zw >> ((offs_n % 8) * 4)[None, :]) & 0xF  # [BLOCK_K, N]

            w = (q.to(tl.float32) - zp.to(tl.float32)) * s  # [BLOCK_K, N] f32

            for mi in tl.static_range(BLOCK_M):
                xm = tl.load(
                    x_ptr + mi * stride_xm + k_rows,
                    mask=(mi < M) & (k_rows < K),
                    other=0.0,
                ).to(tl.float32)
                part = tl.sum(w * xm[:, None], axis=0)  # [BLOCK_N] fp32
                offs_bm = tl.arange(0, BLOCK_M)
                acc = tl.where((offs_bm == mi)[:, None],
                               acc + part[None, :], acc)

        offs_bm = tl.arange(0, BLOCK_M)
        y_ptrs = y_ptr + offs_bm[:, None].to(tl.int64) * stride_y_m \
            + offs_n[None, :]
        tl.store(y_ptrs, acc.to(tl.float16),
                 mask=(offs_bm < M)[:, None] & n_mask[None, :])


def _next_pow2(x: int) -> int:
    return max(1, 1 << (x - 1).bit_length())


def int4_gemv_m(x: torch.Tensor, w_q: torch.Tensor, qzeros: torch.Tensor,
                scales: torch.Tensor) -> torch.Tensor:
    """Skinny int4-GPTQ GEMV: y[M,N] = x @ dequant(W).T, M <= 8, fp16 I/O.

    Layouts are exactly what csrc/quantization/gptq/q_gemm.cu
    (gemm_half_q_half_gptq_4bit_kernel, use_exllama=True, use_v2_format) reads:
      w_q:    int32 [K/8, N]   nibble i (LSB first) = k-octet position i
      qzeros: int32 [G, N/8]   nibble (n % 8) of word [g, n//8], raw (v2)
      scales: fp16  [G, N]     group of k = k // (K / G) == k // group_size
    Numerics: fp32 accumulate; dequant (q - zp) * s in fp32; fp16 store.
    """
    if not HAS_TRITON:
        raise RuntimeError("glm53_int4_gemv requires triton")
    assert x.is_cuda and w_q.is_cuda
    assert x.dtype == torch.float16 and scales.dtype == torch.float16
    M, K = x.shape
    K8, N = w_q.shape
    G = scales.shape[0]
    assert K8 * 8 == K, f"w_q {w_q.shape} incompatible with K={K}"
    assert K % G == 0, "group count must divide K"
    assert M <= 8, f"int4_gemv_m is the skinny path only (M={M} > 8)"
    if not x.is_contiguous():
        x = x.contiguous()
    assert w_q.is_contiguous() and qzeros.is_contiguous()
    assert scales.is_contiguous()

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
    _glm53_int4_gemv_kernel[grid](
        x,
        w_q,
        qzeros,
        scales,
        y,
        M,
        K // G,
        N,
        K,
        x.stride(0),
        y.stride(0),
        BLOCK_M=_next_pow2(M),
        waves_per_eu=int(os.environ.get("GFX906_INT4_GEMV_WPE", "2")),
    )
    return y


# ---------------------------------------------------------------------------
# Hatch 1: ExllamaLinearKernel (CT pack-quantized int4 dense linears)
# ---------------------------------------------------------------------------
_GPTQ_SUPPORTED_PARAMS: dict[int, int] = {}


def _glm53_apply_weights(self, layer, x, bias=None):
    """Mirror of ExllamaLinearKernel.apply_weights with a decode GEMV hatch."""
    from vllm import _custom_ops as ops

    c = self.config
    x_2d = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

    w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

    use_gemv = (
        _GPTQ_SUPPORTED_PARAMS.get(id(self), None) is True
        and x_2d.shape[0] <= int4_gemv_max_tokens()
        and x_2d.dtype == torch.float16
        and bias is None
    )
    if use_gemv:
        y = int4_gemv_m(x_2d, w_q, w_zp, w_s)
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y.reshape(out_shape)

    # --- original path (verbatim) ---
    use_v2_format = True

    assert w_zp is not None, "Zero points are required by Exllama"
    assert w_g_idx is not None, "Group index is required by Exllama"

    x_2d_fp16 = x_2d.to(torch.float16) if x_2d.dtype == torch.float32 else x_2d

    output = ops.gptq_gemm(
        x_2d_fp16, w_q, w_zp, w_s, w_g_idx, True, use_v2_format,
        c.weight_type.size_bits)

    if output.dtype != x.dtype:
        output = output.to(x.dtype)

    if bias is not None:
        output.add_(bias)
    return output.reshape(out_shape)


def _probe_kernel_compat(self, layer) -> bool:
    """Statically decide if this layer can use the Triton GEMV (checked once)."""
    try:
        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)
        c = self.config
        if c.weight_type.size_bits != 4:
            return False
        if w_g_idx is None or w_g_idx.numel() != 0:
            return False  # act-order permutations not supported
        if w_zp is None:
            return False
        if not (c.partition_weight_shape[0] % 32 == 0):
            return False
        # layouts
        n_out = c.partition_weight_shape[1]
        k_in = c.partition_weight_shape[0]
        if w_q.shape != (k_in // 8, n_out):
            return False
        groups = w_s.shape[0]
        if w_s.shape != (groups, n_out) or k_in % groups != 0:
            return False
        if k_in // groups < 8:
            return False
        if w_zp.shape != (groups, n_out // 8):
            return False
        if not (w_q.is_contiguous() and w_s.is_contiguous()
                and w_zp.is_contiguous()):
            return False
        return True
    except Exception:
        return False


_orig_apply_forward = None


def _apply_weights_dispatch(self, layer, x, bias=None):
    key = id(self)
    ok = _GPTQ_SUPPORTED_PARAMS.get(key)
    if ok is None:
        ok = HAS_TRITON and torch.cuda.is_available() and \
            _probe_kernel_compat(self, layer)
        _GPTQ_SUPPORTED_PARAMS[key] = ok
        # note: on the failure path we remember False forever (shapes fixed
        # post-load); nothing numeric is decided here.
    if ok:
        return _glm53_apply_weights(self, layer, x, bias)
    return _orig_apply_forward(self, layer, x, bias)


def install_glm53_int4_gemv() -> bool:
    """Env-gated monkeypatch of ExllamaLinearKernel.apply_weights.

    Called from the glm5next __init__ tail anchor; only active when
    VLLM_GLM53_INT4_GEMV=1. Any decode call failing the (cached, static)
    compatibility probe falls back to the stock gptq_gemm path, so module
    coverage may be partial without breaking forward.
    """
    global _orig_apply_forward
    if not int4_gemv_enabled():
        return False
    if not HAS_TRITON:
        return False
    if _orig_apply_forward is not None:
        return True  # already installed

    from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
        ExllamaLinearKernel,
    )

    _orig_apply_forward = ExllamaLinearKernel.apply_weights
    ExllamaLinearKernel.apply_weights = _apply_weights_dispatch
    print("[GLM53 int4 GEMV] decode M<=%d int4 linear hatch ACTIVE "
          "(VLLM_GLM53_INT4_GEMV=1)" % int4_gemv_max_tokens(), flush=True)
    return True


def uninstall_glm53_int4_gemv() -> bool:
    global _orig_apply_forward
    if _orig_apply_forward is None:
        return False
    from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
        ExllamaLinearKernel,
    )
    ExllamaLinearKernel.apply_weights = _orig_apply_forward
    _orig_apply_forward = None
    _GPTQ_SUPPORTED_PARAMS.clear()
    return True


# ---------------------------------------------------------------------------
# Hatch 2 (optional): ops.LLMM1 (dense fp16 n==1) -> gfx906_gemv.gemv_m
# ---------------------------------------------------------------------------
_orig_llmm1 = None


def install_glm53_dense_gemv() -> bool:
    """Route the fp16 M==1 dense GEMVs (LLGemm1 call sites) through gemv_m."""
    global _orig_llmm1
    if not dense_gemv_enabled():
        return False
    if _orig_llmm1 is not None:
        return True
    from vllm import _custom_ops as ops_mod
    if not hasattr(ops_mod, "LLMM1"):
        return False
    import vllm.gfx906_ext.gfx906_gemv as _gemv

    _orig_llmm1 = ops_mod.LLMM1

    def _llmm1_dispatch(a: torch.Tensor, b: torch.Tensor,
                        rows_per_block: int) -> torch.Tensor:
        # a = weight [N, K], b = x [M(==1), K]; original contract.
        try:
            if (b.dtype == torch.float16 and a.dtype == torch.float16
                    and b.shape[0] <= 8 and a.shape[1] % 8 == 0
                    and a.is_contiguous()):
                return _gemv.gemv_m(b, a)
        except Exception:
            pass
        return _orig_llmm1(a, b, rows_per_block)

    ops_mod.LLMM1 = _llmm1_dispatch
    print("[GLM53 dense GEMV] LLMM1 M<=8 fp16 hatch ACTIVE "
          "(VLLM_GLM53_DENSE_GEMV=1)", flush=True)
    return True


def uninstall_glm53_dense_gemv() -> bool:
    global _orig_llmm1
    if _orig_llmm1 is None:
        return False
    from vllm import _custom_ops as ops_mod
    ops_mod.LLMM1 = _orig_llmm1
    _orig_llmm1 = None
    return True
