# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

# Allow ROCm to proceed without tilelang by providing pure-PyTorch fallbacks.
if TYPE_CHECKING:
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang
    import tilelang.language as T
else:
    if has_tilelang():
        import tilelang
        import tilelang.language as T
    else:
        tilelang = None  # type: ignore[assignment]
        T = None  # type: ignore[assignment]


@cache
def compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // grid_size
    if k is not None:
        # avoid split_k for small k
        num_block_k = cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


# ---------------------------------------------------------------------------
# TileLang fused kernels (CUDA only)
# ---------------------------------------------------------------------------
if tilelang is not None:
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 16,
        hc_mult: int = 4,
    ):
        """Deeply fused kernels, everything other than gemm & sqrsum in mHC pre block."""
        num_tokens = T.dynamic("num_tokens")
        hc_mult3 = hc_mult * (2 + hc_mult)
        hidden_block = math.gcd(512, hidden_size)

        gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
        gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
        hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
        hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
        residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
        # outputs
        post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
        comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
        layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

        with T.Kernel(num_tokens, threads=96) as i:
            T.pdl_sync()
            ##################################################################
            # _pre_norm_fn_fwd_norm
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            mixes_shared = T.alloc_shared(hc_mult3, T.float32)
            T.copy(mixes, mixes_shared)

            if T.get_thread_binding() < 32:
                ##################################################################
                # _pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                for j in T.Parallel(hc_mult):
                    post_mix[i, j] = (
                        T.sigmoid(
                            mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                        )
                        * hc_post_mult_value
                    )
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = (
                        mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                        + hc_base[j * hc_mult + k + hc_mult * 2]
                    )

                ##################################################################
                # _sinkhorn_fwd
                row_sum = T.alloc_fragment(hc_mult, T.float32)
                col_sum = T.alloc_fragment(hc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(hc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_mix[i, j * hc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _pre_split_mixes_fwd (pre)
                pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
                for j in T.Parallel(hc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * hc_scale[0] + hc_base[j],
                        )
                        + hc_pre_eps
                    )
                ###################################################################
                # _pre_apply_mix_fwd
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                    xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                    T.copy(residual[i, 0, i0_h * hidden_block], xs)
                    T.copy(xs, xl)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_hc in T.serial(hc_mult):
                        pre = pre_mix_shared[i_hc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_hc, i1_h]

                    T.copy(ol, layer_input[i, i0_h * hidden_block])
            T.pdl_trigger()


    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def mhc_post_tilelang(
        a,
        b,
        c,
        d,
        x,
        hc: int,
        hidden: int,
        n_thr: int = 128,
        h_blk: int = 1024,
    ) -> tilelang.JITKernel:
        # rename for shorter code
        n = T.dynamic("num_tokens")
        h = hidden

        h_blk = math.gcd(hidden, h_blk)
        a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
        b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
        c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
        d: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
        x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
        with T.Kernel(n, threads=n_thr) as i_n:
            x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((hc, h_blk), T.float32)
            b_local = T.alloc_fragment((hc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((hc, hc), T.float32)
            c_local = T.alloc_fragment(hc, T.float32)
            T.pdl_sync()
            T.copy(a[i_n, 0, 0], a_local)
            T.copy(c[i_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
                T.copy(d[i_n, i0_h * h_blk], d_shared)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_hco, i1_h in T.Parallel(hc, h_blk):
                    x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                    for i_hci in T.serial(hc):
                        x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[i_n, 0, i0_h * h_blk])
            T.pdl_trigger()


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback implementations (ROCm / no-tilelang)
# ---------------------------------------------------------------------------
# Optional fused Triton path (gfx906): gated off by default; when enabled,
# mhc_pre/mhc_post on ROCm dispatch to model_executor.layers.mhc_triton
# instead of the eager fallbacks below. Opt-in only; semantics are kept
# identical to _mhc_pre_fallback/_mhc_post_fallback.
_MHC_TRITON_ENV = "VLLM_DSV4_MHC_TRITON"


def _dsv4_mhc_triton_enabled() -> bool:
    # Read at call time so tests / A/B runs can toggle without re-import.
    return os.environ.get(_MHC_TRITON_ENV, "0").lower() not in (
        "0",
        "",
        "false",
        "off",
    )


def _mhc_pre_fallback(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback for mhc_pre."""
    assert residual.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_mult * hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x_flat = residual_flat.view(num_tokens, hc_mult * hidden_size).float()
    gemm_out_mul = x_flat @ fn.T  # (num_tokens, hc_mult3)
    gemm_out_sqrsum = x_flat.square().sum(-1)  # (num_tokens,)

    rms = torch.rsqrt(gemm_out_sqrsum / (hc_mult * hidden_size) + rms_eps).unsqueeze(-1)
    mixes = gemm_out_mul * rms

    # post_mix
    post_mix = torch.sigmoid(
        mixes[:, hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    ) * hc_post_mult_value

    # comb_mix
    comb = mixes[:, 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]
    comb = comb.view(num_tokens, hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    # pre_mix -> layer_input
    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    layer_input = (pre_mix.unsqueeze(-1) * residual_flat.float()).sum(dim=1).to(residual.dtype)

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def _mhc_post_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch fallback for mhc_post."""
    # x: (..., hidden_size)
    # residual: (..., hc_mult, hidden_size)
    # post_layer_mix: (..., hc_mult, 1)
    # comb_res_mix: (..., hc_mult, hc_mult)
    # out: (..., hc_mult, hidden_size)
    out = post_layer_mix.float() * x.unsqueeze(-2).float()
    out += torch.matmul(
        comb_res_mix.transpose(-2, -1).float(),
        residual.float(),
    )
    return out.to(residual.dtype)


# ---------------------------------------------------------------------------
# Unified mhc_pre / mhc_post that dispatch to tilelang or fallback
# ---------------------------------------------------------------------------
def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tilelang is None:
        if _dsv4_mhc_triton_enabled() and residual.is_cuda:
            from vllm.model_executor.layers import mhc_triton

            return mhc_triton.mhc_pre_triton(
                residual, fn, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat,
            )
        return _mhc_pre_fallback(
            residual, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, n_splits,
        )

    assert residual.dtype in (torch.bfloat16, torch.float16)
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    block_k = 64
    block_m = 64
    n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))

    post_mix = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )

    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn_flat,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if tilelang is None:
        if _dsv4_mhc_triton_enabled() and residual.is_cuda:
            from vllm.model_executor.layers import mhc_triton

            return mhc_triton.mhc_post_triton(x, residual, post_layer_mix,
                                              comb_res_mix)
        return _mhc_post_fallback(x, residual, post_layer_mix, comb_res_mix)

    out = torch.empty_like(residual)
    mhc_post_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def _mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def _mhc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="mhc_pre",
    op_func=mhc_pre,
    mutates_args=[],
    fake_impl=_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="mhc_post",
    op_func=mhc_post,
    mutates_args=[],
    fake_impl=_mhc_post_fake,
)
