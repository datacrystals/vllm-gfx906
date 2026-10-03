# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash track MoE: hand-built int4-group-quantized expert GEMV for
# gfx906 (MI50: 60 CUs, no MFMA, no TMA, wave64), replacing the
# `moe_wna16_gemm_kernel` CUDA path (and the stock tl.dot triton fallback)
# for decode shapes M <= 4 (topk=8 -> <=32 routed slots per call).
#
# Why (trace attribution, /data/llmbench/glm53-prof{,2} rank0, one decode
# step, TP8 so 36 local experts/rank; analysis in MOE_WNA16_PLAN.md at the
# fork root / inspect scripts in /data/llmbench/glm53-prof3):
#   * vllm::moe_wna16_gemm_kernel<__half,4,{16,8}> runs x82/step
#     (41 MoE layers x gate_up + down), ~26 ms/step of the ~85 ms budget.
#   * At bs1-4 the avg call moves only ~1 active expert's weights
#     (9.7 MB gate_up / 4.9 MB down) in ~317 us => ~25-55 GB/s effective,
#     i.e. 3-6% of the measured 950 GB/s HBM. It is NOT roofline-bound;
#     it is latency/scratch bound:
#       - `float res[64]` indexed by the runtime `num_valid_tokens` loop
#         variable can not be register-promoted (moe_wna16.cu:131,258-279)
#         -> local-memory (DRAM-backed scratch) traffic per k-iteration.
#       - runtime BLOCK_SIZE_K kernel arg prevents compile-time unroll of
#         the k loop -> one exposed HBM round trip per float4 weight load.
#       - tp8 decode activates only 36*(1-(287/288)^(8M)) ~ 1-4 of the 36
#         local experts, but NR split is BN=256 -> only 16 n-blocks so the
#         K-split z axis (8-16 CTAs) does all the parallelism, at the cost
#         of `output.zero_()` fills + half CAS atomics (utils.h:54-67).
#
# This module:
#   * `glm53_wna16_gemv(...)` — drop-in replacement for
#     `invoke_fused_moe_wna16_triton_kernel` (same signature/contract),
#     specialized GEMV-style: BLOCK_M=4 rows of A, lanes run along the
#     PACKED K axis (int32 view [E, N, K/8]), so each lane reads 16B
#     contiguous (nibble j of word u = k 8u+j, the CT pack_to_int32 /
#     stock-triton-kernel convention). Pure FMA, fp32 accumulator in
#     registers, no tl.dot, no PDL/TMA, num_warps <= 4, tiles <= 4096.
#   * optional K-split via a 3rd grid axis with native fp32
#     global_atomic_add (gfx906 has native f32 atomics; fp16 atomics used
#     by the CUDA kernel are CAS loops on gfx9) + a tiny epilogue kernel.
#     Default SPLIT_K=1 (N/BN=128 CTAs per active expert is already >=
#     2 full waves of 60 CUs).
#   * `install_glm53_wna16_gemv()` — monkeypatches
#     futures: forces the wna16 dispatch off the CUDA kernel for small
#     batches and routes it to launch_glm53_wna16_gemv; everything else
#     (int8, big-M prefill, odd shapes) falls through to the originals.
#     Anchor edit, mirroring gfx906_gemv at the tail of
#     vllm/model_executor/models/glm5next/__init__.py:
#         # --- gfx906 MoE wna16 GEMV anchor (VLLM_GLM53_WNA16_GEMV=1) ---
#         if __import__("os").environ.get("VLLM_GLM53_WNA16_GEMV") == "1":
#             from vllm.gfx906_ext import glm53_wna16_gemv as _w16v
#             _w16v.install_glm53_wna16_gemv()
#
# Env gates (default OFF):
#   VLLM_GLM53_WNA16_GEMV=1          enable the decode GEMV override
#   VLLM_GLM53_WNA16_MAX_TOKENS   decode-shape threshold on M*topk
#                                    (default 64; above this -> stock CUDA)
#   VLLM_GLM53_WNA16_BN   BLOCK_N (default 32)
#   VLLM_GLM53_WNA16_BK   BLOCK_K (default 256; must be >= group_size, K
#                                  must be divisible by BK*SPLIT_K)
#   VLLM_GLM53_WNA16_NW   num_warps (default 2; keep <=4, wave64)
#   VLLM_GLM53_WNA16_NS   num_stages (default 2)
#   VLLM_GLM53_WNA16_SPLITK  K-split over program ids with fp32 atomic add
#                            (default 1 => deterministic, no atomics)
#
# Numerics: identical dequant semantics to stock kernels,
#   w = (q - zp) * s  per int4 group of 32, q/zp nibbles unpacked with the
#   CT pack_to_int32 byte order (nibble j of u32 u == k = 8u + j; zp byte
#   n//2, low nibble = even n). Stock triton/CUDA paths multiply in fp16;
#   this kernel keeps dequantized W in fp32 (A is fp16 from the layer) and
#   accumulates fp32 in registers via tl.sum - strictly >= stock accuracy,
#   deterministic at SPLIT_K=1. Offline CPU check:
#   /data/llmbench/glm53-prof3/check_glm53_wna16_gemv.py (bit-exact gate on
#   integer-representable data, tolerance gate on realistic data).
#   GPU correctness+time A/B:
#   /data/llmbench/glm53-prof3/bench_glm53_wna16_gemv.py

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

_ENV_GATE = "VLLM_GLM53_WNA16_GEMV"
_ENV_MAX_TOK = "VLLM_GLM53_WNA16_MAX_TOKENS"
_ENV_BN = "VLLM_GLM53_WNA16_BN"
_ENV_BK = "VLLM_GLM53_WNA16_BK"
_ENV_NW = "VLLM_GLM53_WNA16_NW"
_ENV_NS = "VLLM_GLM53_WNA16_NS"
_ENV_SK = "VLLM_GLM53_WNA16_SPLITK"

_MAX_TOKENS_DEFAULT = 64
_BLOCK_M = 4  # kernel m-tile; requires ALIGN_BM % BLOCK_M == 0


def wna16_gemv_enabled() -> bool:
    return os.environ.get(_ENV_GATE, "0").strip().lower() in ("1", "true", "on")


def max_tokens() -> int:
    try:
        return int(os.environ.get(_ENV_MAX_TOK, str(_MAX_TOKENS_DEFAULT)))
    except ValueError:
        return _MAX_TOKENS_DEFAULT


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


if HAS_TRITON:

    @triton.jit
    def _glm53_wna16_gemv_kernel(
        a_ptr,  # [rows, K] fp16; row = slot // top_k
        b_ptr,  # int32 view of qweight: [E, N, K//8], nibble j of word u -> k=8u+j
        c_ptr,  # [M*topk, N] out dtype (fp16 in production)
        c_acc_ptr,  # fp32 [num_slots, N] scratch when SPLIT_K>1 (else unused)
        s_ptr,  # [E, N, K//GROUP_SIZE] fp16 (kg fastest, stride 1)
        z_ptr,  # uint8 [E, N//2, K//GROUP_SIZE]; byte n//2, low nibble = even n
        topk_weights_ptr,  # f32 [M*topk]
        sorted_token_ids_ptr,  # int32 [EM]
        expert_ids_ptr,  # int32 [EM // ALIGN_BM]
        ntpp_ptr,  # int32 [1]
        num_valid_tokens,
        N,
        K,
        top_k,
        stride_am,
        stride_be,
        stride_bn,
        stride_se,
        stride_sn,
        stride_ze,
        stride_zn,
        stride_cm,
        stride_c32m,
        GROUP_SIZE: tl.constexpr,
        ALIGN_BM: tl.constexpr,
        HAS_ZP: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        BK8: tl.constexpr = BLOCK_K // 8
        KG: tl.constexpr = BLOCK_K // GROUP_SIZE
        GS8: tl.constexpr = GROUP_SIZE // 8

        pid_m = tl.program_id(0)
        ntpp = tl.load(ntpp_ptr)
        if pid_m * BLOCK_M >= ntpp:
            return
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)

        offs_sm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        slot = tl.load(sorted_token_ids_ptr + offs_sm)
        tok_ok = slot < num_valid_tokens
        slot = tl.where(tok_ok, slot, 0).to(tl.int64)
        expert = tl.load(expert_ids_ptr + (pid_m * BLOCK_M) // ALIGN_BM)

        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        n_mask = offs_n < N
        out_mask = tok_ok[:, None] & n_mask[None, :]

        if expert == -1:
            if SPLIT_K == 1:
                z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                c_ptrs = c_ptr + slot[:, None] * stride_cm + offs_n[None, :]
                tl.store(
                    c_ptrs, z.to(c_ptr.dtype.element_ty), mask=out_mask
                )
            return

        offs_k8 = tl.arange(0, BK8)
        offs_kg = tl.arange(0, KG)

        w_base = b_ptr + expert * stride_be + offs_n[:, None] * stride_bn
        s_base = s_ptr + expert * stride_se + offs_n[:, None] * stride_sn
        if HAS_ZP:
            z_base = z_ptr + expert * stride_ze + (offs_n // 2)[:, None] * stride_zn
            z_shift = ((offs_n % 2) * 4)[:, None]
        row = slot // top_k  # [BLOCK_M] int64
        a_base = a_ptr + row[:, None] * stride_am

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        n_iters = K // (BLOCK_K * SPLIT_K)
        for it in range(0, n_iters):
            kb = pid_k * n_iters + it
            k8 = kb * BK8 + offs_k8  # u32 word indices along K
            kg = kb * KG + offs_kg  # quantization-group indices
            # [BLOCK_N, BK8] int32 tile; lanes walk the contiguous packed-K
            # axis -> 4 consecutive u32 per lane = 16B vector loads.
            w_u32 = tl.load(w_base + k8[None, :])
            sca = tl.load(s_base + kg[None, :]).to(tl.float32)  # [BN, KG]
            sc_e = tl.reshape(
                tl.broadcast_to(sca[:, :, None], (BLOCK_N, KG, GS8)),
                (BLOCK_N, BK8),
            )
            if HAS_ZP:
                zb = tl.load(z_base + kg[None, :])  # [BN, KG] uint8
                zpf = ((zb.to(tl.int32) >> z_shift) & 0xF).to(tl.float32)
                zp_e = tl.reshape(
                    tl.broadcast_to(zpf[:, :, None], (BLOCK_N, KG, GS8)),
                    (BLOCK_N, BK8),
                )
            else:
                zp_e = tl.zeros((BLOCK_N, BK8), dtype=tl.float32) + 8.0
            for j in tl.static_range(8):
                # nibble j of u32 word u <-> k = 8u + j (CT pack order)
                wj = ((w_u32 >> (j * 4)) & 0xF).to(tl.float32)
                wd = (wj - zp_e) * sc_e
                aj = tl.load(
                    a_base + ((kb * BLOCK_K) + offs_k8 * 8 + j)[None, :],
                    mask=tok_ok[:, None],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(aj[:, None, :] * wd[None, :, :], axis=2)

        if SPLIT_K == 1:
            if MUL_ROUTED_WEIGHT:
                tw = tl.load(topk_weights_ptr + slot, mask=tok_ok, other=0.0)
                acc = acc * tw[:, None]
            c_ptrs = c_ptr + slot[:, None] * stride_cm + offs_n[None, :]
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=out_mask)
        else:
            # gfx906 has native f32 global atomics (unlike fp16 CAS loops).
            # routed weight is applied once, in the split-k epilogue.
            c32_ptrs = c_acc_ptr + slot[:, None] * stride_c32m + offs_n[None, :]
            tl.atomic_add(c32_ptrs, acc, mask=out_mask)

    @triton.jit
    def _glm53_wna16_splitk_epilogue(
        c_acc_ptr,  # fp32 [num_valid, N]
        c_ptr,  # out dtype [M*topk, N]
        tw_ptr,
        num_valid_tokens,
        N,
        stride_c32m,
        stride_cm,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        v = tl.load(c_acc_ptr + row * stride_c32m + offs_n, mask=n_mask, other=0.0)
        if MUL_ROUTED_WEIGHT:
            v = v * tl.load(tw_ptr + row)
        tl.store(
            c_ptr + row * stride_cm + offs_n,
            v.to(c_ptr.dtype.element_ty),
            mask=n_mask,
        )


def _launch_cfg():
    return (
        _env_int(_ENV_BN, 32),  # BLOCK_N
        _env_int(_ENV_BK, 256),  # BLOCK_K
        _env_int(_ENV_NW, 2),  # num_warps (<=4, wave64 hardware)
        _env_int(_ENV_NS, 2),  # num_stages
        _env_int(_ENV_SK, 1),  # SPLIT_K (1 = deterministic, no atomics)
    )


def glm53_wna16_supported(
    A, B, C, B_scale, B_zp, topk_weights, top_k, config, use_int8_w8a16,
    use_int4_w4a16, block_shape, sorted_token_ids=None,
) -> tuple[bool, str]:
    """Decide whether the decode GEMV path can serve this call."""
    if not HAS_TRITON:
        return False, "no triton"
    if not (use_int4_w4a16 and not use_int8_w8a16):
        return False, "not int4_w4a16"
    if block_shape is None or block_shape[1] <= 0:
        return False, "no group quant"
    if A.size(0) * top_k > max_tokens():
        return False, f"tokens {A.size(0) * top_k} > max {max_tokens()}"
    BLOCK_N, BLOCK_K, _, _, SPLIT_K = _launch_cfg()
    K, N = A.size(1), B.size(1)
    gs = block_shape[1]
    if gs % 8 != 0 or K % gs != 0:
        return False, f"group_size {gs} not multiple-of-8 divisor of K"
    if BLOCK_K % gs != 0 or BLOCK_K % 8 != 0:
        return False, f"BLOCK_K {BLOCK_K} incompatible with gs {gs}"
    if K % (BLOCK_K * SPLIT_K) != 0:
        return False, f"K {K} not divisible by BLOCK_K*SPLIT_K {BLOCK_K*SPLIT_K}"
    if N % BLOCK_N != 0:
        return False, f"N {N} not divisible by BLOCK_N {BLOCK_N}"
    align_bm = int(config.get("BLOCK_SIZE_M", 16))
    if align_bm % _BLOCK_M != 0:
        return False, f"align block {align_bm} not a multiple of {_BLOCK_M}"
    if B.dtype != torch.uint8 or B.dim() != 3 or B.stride(2) != 1:
        return False, "unexpected qweight layout"
    if B_scale is None or B_scale.dim() != 3 or B_scale.stride(2) != 1:
        return False, "unexpected scale layout"
    if B_zp is not None and (B_zp.dim() != 3 or B_zp.stride(2) != 1):
        return False, "unexpected zp layout"
    if A.stride(1) != 1 or C.stride(-1) != 1:
        return False, "non-contiguous A/C rows"
    if sorted_token_ids is not None and sorted_token_ids.stride(0) != 1:
        return False, "sorted_token_ids not contiguous"
    return True, "ok"


def glm53_wna16_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type=None,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = True,
    block_shape: list | None = None,
) -> None:
    """Drop-in replacement for invoke_fused_moe_wna16_triton_kernel.

    Same tensor contract and output semantics (writes every valid slot;
    zero-fills slots whose expert is off-rank at SPLIT_K==1; relies on the
    caller zeroing / downstream 0-weights for padding slots).
    """
    BLOCK_N, BLOCK_K, num_warps, num_stages, SPLIT_K = _launch_cfg()
    M = A.size(0)
    num_tokens = M * top_k
    N = B.size(1)
    K = A.size(1)
    gs = block_shape[1]
    align_bm = int(config.get("BLOCK_SIZE_M", 16))
    # C is [M, top_k, N] in production (stride_cm = C.stride(1) == N);
    # accept a flat [M*top_k, N] buffer too (bench harness).
    stride_cm = C.stride(1) if C.dim() == 3 else C.stride(0)

    b32 = B.view(torch.int32)  # [E, N, K//8]  (B is uint8 [E, N, K//2])
    E_local = B.size(0)
    del E_local

    EM = sorted_token_ids.size(0)
    if M < align_bm:
        EM = min(EM, M * top_k * align_bm)

    has_zp = B_zp is not None
    if has_zp:
        z_arg = B_zp
        z_strides = (B_zp.stride(0), B_zp.stride(1))
    else:
        z_arg = B  # dummy, never read
        z_strides = (0, 0)
    tw = topk_weights if topk_weights is not None else A  # dummy

    if SPLIT_K > 1:
        c_acc = torch.zeros(num_tokens, N, device=A.device, dtype=torch.float32)
        stride_c32m = c_acc.stride(0)
    else:
        c_acc = C  # dummy, never touched
        stride_c32m = 0

    grid = (triton.cdiv(EM, _BLOCK_M), N // BLOCK_N, SPLIT_K)
    _glm53_wna16_gemv_kernel[grid](
        A,
        b32,
        C,
        c_acc,
        B_scale,
        z_arg,
        tw,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_tokens,
        N,
        K,
        top_k,
        A.stride(0),
        b32.stride(0),
        b32.stride(1),
        B_scale.stride(0),
        B_scale.stride(1),
        z_strides[0],
        z_strides[1],
        stride_cm,
        stride_c32m,
        GROUP_SIZE=gs,
        ALIGN_BM=align_bm,
        HAS_ZP=has_zp,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if SPLIT_K > 1:
        epi_bn = 1024 if N % 1024 == 0 else (256 if N % 256 == 0 else 128)
        _glm53_wna16_splitk_epilogue[(num_tokens, triton.cdiv(N, epi_bn))](
            c_acc,
            C,
            tw,
            num_tokens,
            N,
            c_acc.stride(0),
            stride_cm,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            BLOCK_N=epi_bn,
            num_warps=4,
        )


# ---------------------------------------------------------------------------
# CPU mirror of the kernel dataflow (used by the offline numerics check and
# by the GPU bench as the ground truth for tier-tolerance comparisons).
# Pure torch, no GPU. Same layout/index/unpack semantics as the kernel.
# ---------------------------------------------------------------------------


def moe_wna16_gemv_mirror(
    A: torch.Tensor,  # [rows, K] fp16 (CPU)
    B: torch.Tensor,  # uint8 [E, N, K//2] (packed, contiguous)
    B_scale: torch.Tensor,  # [E, N, K//g] fp16
    B_zp: torch.Tensor | None,  # uint8 [E, N//2, K//g]
    topk_weights: torch.Tensor | None,  # f32 [M*top_k]
    sorted_token_ids: torch.Tensor,  # int32 [EM]
    expert_ids: torch.Tensor,  # int32 [EM // align_bm]
    num_tokens_post_padded: torch.Tensor,  # int32 [1]
    mul_routed_weight: bool,
    top_k: int,
    group_size: int,
    C: torch.Tensor | None = None,  # [num_valid, N] target dtype buffer
    align_bm: int = 16,  # EM-block granularity used for expert_ids
) -> torch.Tensor:
    M = A.size(0)
    E, N = B.size(0), B.size(1)
    K = A.size(1)
    num_valid = M * top_k
    ntpp = int(num_tokens_post_padded.item())
    if C is None:
        C = torch.zeros(num_valid, N, dtype=A.dtype)
    b32 = B.view(torch.int32)  # [E, N, K//8], nibble j of word u = k 8u+j
    u = torch.arange(K // 8)
    g_of = u // (group_size // 8)  # quant group of u32 word u
    # slots whose expert block was remapped to -1 (off-rank experts under
    # TP/EP): kernel writes zeros for them
    neg_mask = torch.zeros_like(sorted_token_ids, dtype=torch.bool)
    for b in range(expert_ids.numel()):
        if int(expert_ids[b].item()) == -1:
            lo = b * align_bm
            hi = min(lo + align_bm, ntpp, sorted_token_ids.numel())
            neg_mask[lo:hi] = True
    for e in range(E):
        tok_blocks = torch.nonzero(expert_ids == e).flatten()
        if tok_blocks.numel() == 0:
            continue
        slot_idx = torch.cat(
            [torch.arange(int(b) * align_bm,
                          min(int(b) * align_bm + align_bm, ntpp))
             for b in tok_blocks]
        )
        slots = sorted_token_ids[slot_idx]
        slots = slots[slots < num_valid].to(torch.int64)
        if slots.numel() == 0:
            continue
        rows = slots // top_k
        a = A.index_select(0, rows).to(torch.float32)  # [S, K]
        q = b32[e].to(torch.int32)  # [N, K/8]
        s = B_scale[e].to(torch.float32)  # [N, Kg]
        if B_zp is not None:
            zb = B_zp[e].to(torch.int32)  # [N/2, Kg]
            z = torch.stack([zb & 0xF, (zb >> 4) & 0xF], dim=1)
            z = z.reshape(N, -1).to(torch.float32)  # [N, Kg]
        else:
            z = torch.full_like(s, 8.0)
        s_full = s[:, g_of]  # [N, K/8] expand groups -> u32 words
        z_full = z[:, g_of]
        acc = torch.zeros(slots.numel(), N, dtype=torch.float32)
        for j in range(8):
            wj = ((q >> (4 * j)) & 0xF).to(torch.float32)  # [N, K/8]
            wd = (wj - z_full) * s_full
            aj = a[:, u * 8 + j]  # [S, K/8] activation at k = 8u+j
            acc += aj @ wd.T
        if mul_routed_weight:
            acc = acc * topk_weights.to(torch.float32).index_select(0, slots)[:, None]
        C[slots] = acc.to(C.dtype)
    neg_slots = sorted_token_ids[neg_mask & (sorted_token_ids < num_valid)]
    if neg_slots.numel() > 0:
        C[neg_slots.to(torch.int64)] = 0.0
    return C


# ---------------------------------------------------------------------------
# Monkeypatch installer (mirrors gfx906_gemv.install_gfx906_gemv convention).
# ---------------------------------------------------------------------------


def install_glm53_wna16_gemv() -> bool:
    """Route small-batch int4 g32 MoE wna16 to the decode GEMV kernel.

    Only activates with VLLM_GLM53_WNA16_GEMV=1. Anything outside the
    decode envelope (int8, big batches, non-standard shapes) falls through
    to the original CUDA/Triton paths untouched.
    """
    if not wna16_gemv_enabled():
        return False
    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    if getattr(fm, "_glm53_wna16_orig_invoke", None) is not None:
        return True  # already installed

    fm._glm53_wna16_orig_invoke = fm.invoke_fused_moe_wna16_triton_kernel
    fm._glm53_wna16_orig_should = fm.should_moe_wna16_use_cuda

    def _should_cuda(num_valid_tokens, group_size, num_experts, bit):
        # decode shapes: take the triton path (patched to our GEMV below)
        if bit == 4 and num_valid_tokens <= max_tokens():
            return False
        return fm._glm53_wna16_orig_should(
            num_valid_tokens, group_size, num_experts, bit
        )

    def _invoke(A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids,
                expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,
                config, compute_type, use_int8_w8a16, use_int4_w4a16,
                block_shape):
        ok, reason = glm53_wna16_supported(
            A, B, C, B_scale, B_zp, topk_weights, top_k, config,
            use_int8_w8a16, use_int4_w4a16, block_shape,
            sorted_token_ids=sorted_token_ids,
        )
        if ok:
            return glm53_wna16_gemv(
                A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids,
                expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,
                config, compute_type, use_int8_w8a16, use_int4_w4a16,
                block_shape,
            )
        return fm._glm53_wna16_orig_invoke(
            A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids,
            expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,
            config, compute_type, use_int8_w8a16, use_int4_w4a16, block_shape,
        )

    fm.should_moe_wna16_use_cuda = _should_cuda
    fm.invoke_fused_moe_wna16_triton_kernel = _invoke
    print(
        "[GLM53 wna16 GEMV] decode MoE int4 g32 override ACTIVE "
        f"(max_tokens={max_tokens()}, BN/BK/NW/NS/SK={_launch_cfg()})",
        flush=True,
    )
    return True


def uninstall_glm53_wna16_gemv() -> bool:
    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as fm
    except Exception:
        return False
    orig = getattr(fm, "_glm53_wna16_orig_invoke", None)
    if orig is None:
        return False
    fm.invoke_fused_moe_wna16_triton_kernel = orig
    fm.should_moe_wna16_use_cuda = fm._glm53_wna16_orig_should
    fm._glm53_wna16_orig_invoke = None
    fm._glm53_wna16_orig_should = None
    return True
