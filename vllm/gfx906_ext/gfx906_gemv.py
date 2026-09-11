# SPDX-License-Identifier: Apache-2.0
# Track H / C2 — hand-built skinny-GEMV for fp16 M<=8 on gfx906 (MI50, no MFMA).
#
# Replaces the generic tiled `triton_matmul_kernel` used by
# `rocm_unquantized_gemm` for the 4 skinny fp16 GEMMs per Hy3 layer
# (qkv / o_proj / shared gate_up / shared down) + lm_head at decode
# batch sizes 1..8. Measured baseline: ~29 ms/step (321 calls @ bs2, TP8),
# ~10-15x off HBM roofline (see /data/llmbench/hy3-prof/SUMMARY.md, C2).
#
# Design (bandwidth-bound GEMV):
#   * one program per BLOCK_N output rows of W (W is [N, K] row-major,
#     K contiguous => every row is 16B-aligned when K % 8 == 0)
#   * k-major loop: each iteration loads a [BLOCK_N, BLOCK_K] tile of W
#     (vectorized 16B/lane along K for the right BLOCK_K/num_warps combos)
#     and the matching [BLOCK_M, BLOCK_K] tile of x (tiny, L1-resident)
#   * pure FMA (no tl.dot): fp32 accumulation; the W tile is reused
#     across all M rows via a static per-row loop (acc[m, n] +=
#     sum_k x[m, k] * w[n, k]) => register footprint stays
#     W-tile-sized for every M in 1..8
#   * grid = cdiv(N, BLOCK_N); each weight byte is read exactly once
#   * BLOCK_M = next_pow2(M) (mask pads rows), so the same kernel covers
#     M = 1..8 with zero wasted weight traffic
#   * low register pressure: num_warps <= 4 (<= 256 threads/block),
#     W tile <= BLOCK_N x BLOCK_K <= 4096 elems
#
# Env gating (reversible):
#   VLLM_GFX906_GEMV=1 enables the override inside
#   `install_gfx906_gemv()` (wired at the env-gated anchors at the tail of
#   vllm/model_executor/models/hy_v3.py and
#   vllm/model_executor/models/glm5next/__init__.py, mirroring the
#   VLLM_GFX906_PROF_DIR / VLLM_GDN_GFX906_AUTOPATCH anchors). This module is
#   vendored at vllm/gfx906_ext/ and supersedes
#   /data/vllm-gfx906-dsv4/patches/gdn/gfx906_gemv.py (provenance only).
#   With the flag unset the module only defines `gemv_m()`; the vLLM
#   dispatch is untouched and the original triton_matmul path runs.

import os

import torch
import triton
import triton.language as tl

_GFX906_GEMV_ENABLED = os.environ.get("VLLM_GFX906_GEMV", "0") == "1"


def _autotune_configs():
    cfgs = []

    def add(bn, bk, warps, stages=1):
        cfgs.append(
            triton.Config(
                {"BLOCK_N": bn, "BLOCK_K": bk},
                num_warps=warps,
                num_stages=stages,
            )
        )

    # Small BLOCK_N -> more blocks (occupancy) for narrow-N shapes;
    # BLOCK_K >= 8 elems/lane along K keeps 16B global loads.
    add(1, 512, 1)
    add(1, 1024, 2)
    add(2, 512, 2)
    add(2, 1024, 4)
    add(4, 256, 2)
    add(4, 512, 4)
    add(8, 256, 4)
    add(8, 512, 4)
    add(16, 128, 4)
    add(16, 256, 4)
    # shallow software pipeline variants (registers are cheap here)
    add(8, 256, 4, stages=2)
    add(4, 512, 4, stages=2)
    return cfgs


def _next_pow2(x: int) -> int:
    return max(1, 1 << (x - 1).bit_length())


@triton.autotune(
    configs=_autotune_configs(),
    key=["M", "N", "K"],
    # BLOCK_M is derived from M at launch time; not an autotune dimension
    # (the padding is inside one pow2 bucket so M=1..4 don't retune each).
)
@triton.jit
def _gfx906_gemv_kernel(
    x_ptr,  # [M, K] fp16, row-major
    w_ptr,  # [N, K] fp16, row-major (K contiguous)
    y_ptr,  # [M, N] fp16, row-major
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = w_ptr + offs_n * stride_wn

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        # [BLOCK_N, BLOCK_K] fp16 tile of W: k-major, 16B/lane vectorized
        # for (BLOCK_K / threads) >= 8.
        w = tl.load(
            w_base[:, None] + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # Static per-row M loop: W tile register footprint is shared by all
        # rows, so cost of M>1 is only extra FMAs + a [BLOCK_K] x row, not a
        # [BLOCK_M, BLOCK_N, BLOCK_K] live product tile (that form spills at
        # BLOCK_M >= 4 and halves effective HBM bandwidth on gfx906).
        for mi in tl.static_range(BLOCK_M):
            xm = tl.load(
                x_ptr + mi * stride_xm + offs_k,
                mask=k_mask & (mi < M),
                other=0.0,
            ).to(tl.float32)
            part = tl.sum(w * xm[None, :], axis=1)  # [BLOCK_N] fp32
            acc = tl.where((offs_m == mi)[:, None], acc + part[None, :], acc)

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


_gemv_call_count = 0


def gemv_m(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Skinny GEMV: y[M, N] = x[M, K] @ w[N, K].T for fp16, M <= 8.

    Requires x row-major (stride(1) == 1) and w row-major [N, K] with
    K % 8 == 0 (so every K-contiguous row is 16B aligned). fp32
    accumulation, fp16 output — numerics match `triton_matmul` semantics
    (fp32 partner + fp32 acc, fp16 store) up to fp reduction order.
    """
    global _gemv_call_count
    _gemv_call_count += 1

    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw, f"K mismatch: x {x.shape} vs w {w.shape}"
    assert x.dtype == torch.float16 and w.dtype == torch.float16
    assert M <= 8, f"gemv_m is the skinny path only (M={M} > 8)"
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    block_m = _next_pow2(M)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
    _gfx906_gemv_kernel[grid](
        x,
        w,
        y,
        M,
        N,
        K,
        x.stride(0),
        w.stride(0),
        y.stride(0),
        BLOCK_M=block_m,
        waves_per_eu=int(os.environ.get("GFX906_GEMV_WPE", "1")),
    )
    return y


def install_gfx906_gemv() -> bool:
    """Route the fork's skinny fp16 triton_matmul through gemv_m.

    Called from the hy_v3.py tail anchor; only active when
    VLLM_GFX906_GEMV=1. Falls back to the original kernel for
    non-fp16 dtypes or M > 8 (prefill chunk path), and for
    non-contiguous/misaligned weights.
    """
    if not _GFX906_GEMV_ENABLED:
        return False

    import vllm.model_executor.layers.utils as _u

    if getattr(_u, "_gfx906_gemv_orig", None) is not None:
        return True  # already installed

    _orig_triton_matmul = _u.triton_matmul
    _u._gfx906_gemv_orig = _orig_triton_matmul

    def _dispatch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # b is [N, K] fp16/bf16 dense weight, a is [tokens, K]
        if (
            a.dtype == torch.float16
            and a.dim() == 2
            and a.shape[0] <= 8
            and b.shape[1] % 8 == 0
            and b.is_contiguous()
        ):
            return gemv_m(a, b)
        return _orig_triton_matmul(a, b)

    _u.triton_matmul = _dispatch
    print("[gfx906 GEMV] skinny fp16 GEMV override ACTIVE "
          "(M<=8 dense unquantized linears)", flush=True)
    return True


def uninstall_gfx906_gemv() -> bool:
    """Restore the original triton_matmul dispatch (reversibility)."""
    try:
        import vllm.model_executor.layers.utils as _u
    except Exception:
        return False
    orig = getattr(_u, "_gfx906_gemv_orig", None)
    if orig is None:
        return False
    _u.triton_matmul = orig
    _u._gfx906_gemv_orig = None
    return True
