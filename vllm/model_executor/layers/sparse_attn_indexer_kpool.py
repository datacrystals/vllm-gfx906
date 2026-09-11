# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# GLM53-PORT: vendored from upstream vLLM main (GLM-5.3-Flash, PR #53906),
# ROCm path rerouted to the fork's fp16 kpool custom op
# (vllm::rocm_aiter_sparse_attn_indexer_kpool, see
# vllm/v1/attention/ops/rocm_aiter_mla_sparse_kpool.py). The upstream CUDA
# fp8/DeepGEMM path is kept out of this fork's build.
"""Custom Sparse Attention Indexer with K-pooling (GLM-5.3-Flash)."""

import torch

import vllm.envs as envs
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse Attention Indexer Custom Op Layer with kpool compression.

    On ROCm/gfx906 this dispatches to the fp16 radix variant
    (no fp8/hadamard). The CUDA fp8/DeepGEMM path of upstream is not
    available in this fork.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        if current_platform.is_rocm():
            return self.forward_hip(
                hidden_states,
                q_quant,
                k,
                weights,
                gate_score=gate_score,
                compress_ape=compress_ape,
                index_kpool=index_kpool,
                positions=positions,
            )
        raise NotImplementedError(
            "SparseAttnIndexerKpool is only implemented for ROCm in this fork "
            "(GLM53-PORT: upstream fp8/DeepGEMM path not carried)."
        )

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError(
            "GLM53-PORT: upstream CUDA fp8/DeepGEMM path not carried in fork."
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache"
        assert not self.skip_k_cache_insert, (
            "AMD kpool indexer does not support skip_k_cache_insert yet"
        )
        assert isinstance(q_quant, torch.Tensor), (
            "ROCm kpool indexer expects a single fp16 q tensor"
        )
        assert gate_score is not None and compress_ape is not None
        assert index_kpool > 1
        assert positions is not None
        # GLM53-PORT: gfx906 fp16 path (default on ROCm; upstream fp8
        # Hadamard path requires fp8 compute the hardware lacks).
        assert envs.VLLM_GLM53_INDEXER_FP16, (
            "NON-fp16 kpool indexer path is not supported on this fork's ROCm "
            "build; keep VLLM_GLM53_INDEXER_FP16=1."
        )
        return torch.ops.vllm.rocm_aiter_sparse_attn_indexer_kpool(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            (
                _encode_layer_name(self.tail_cache.prefix)
                if self.tail_cache is not None
                else None
            ),
        )
