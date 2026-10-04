#!/bin/bash
# MiMo-V2.6-Flash-INT4 OMNI boot launcher — 8x MI50 gfx906.
# This is the OMNI-capable variant (image/video/audio inputs): it mirrors the
# live serving cmdline captured 2026-10-03 (post vision-fix campaign) and is
# the one to use for prod/restarts. run_mimo_v2_6.sh is the text-only recipe
# (limit-mm-per-prompt image:0 video:0) — do not use it for omni serving.
#
# Differences vs run_mimo_v2_6.sh (all deliberate):
#   --skip-mm-profiling : profile_run's dummy encoder pass allocates ~16 GiB
#       while ~25 GiB of weights are resident -> guaranteed OOM on 32 GB.
#   --kv-cache-memory-bytes 3758096384 : manual 3.5 GiB KV pool (this
#       bypasses gpu_memory_utilization accounting; watch free VRAM in
#       rocm-smi if you change it).
#   --limit-mm-per-prompt image:2 video:1 audio:2 : omni inputs ON.
#   VLLM_VIS_DUMP is intentionally NOT set (it dumps 68 MB of vision
#       activations per MM request; debug-only hook).
#
# KV_CACHE_BYTES: KV pool override (default 3.5 GiB = 1.98x @256k).
# 3x @256k needs >=5.3 GiB (1.77 GiB/session measured). Set
# KV_CACHE_BYTES=6012954214 (5.6 GiB) for the 3x experiment.
# KV_OFFLOAD_GB: native CPU KV offload buffer (GiB, TP-total). Requires
# DISABLE_HYBRID_KM=1 with this hybrid-SWA model (fork constraint).
# Offload is for PARKED sessions only - hot decode KV on PCIe kills
# throughput. Defaults: both unset = offload disabled.
# LONG_PREFILL_THRESHOLD: chunked-prefill interleave cap (tokens).
# Default 0 = one long prefill owns the whole token budget per
# step (prefills serialize). Set to 512 to let up to 4 long prefills
# share a step -> concurrent 256k sessions.
# MAX_BATCHED_TOKENS: prefill chunk budget (default 16384 = sweep winner).
# Swept 2048/4096/8192/16384 on 2026-10-04: 154/179/197/205 tok/s @20k,
# 143/166/178/184 @60k; decode identical (24.4 vs 24.7) so 16384 is free.
# With LONG_PREFILL_THRESHOLD < budget/n, n long prefills interleave.
# DISABLE_PREFIX_CACHE: set to any non-empty value to boot with
# --no-enable-prefix-caching (quality A/B for stray-'.' / multi-turn
# incoherence). Unset (default) keeps --enable-prefix-caching.
# reasoning parser: --reasoning-parser deepseek_r1 (MiMo emits think-tag pairs).
# NOTE commit 65c4c0b487: the parser's opening-marker strip is REQUIRED —
# without it reasoning_content starts with the literal think-start token.

VENV_PATH="/data/vllm-gfx906-dsv4/vllm_dsv4_env"
MODEL_PATH="/data/ModelDownloader/MiMo-V2.6-Flash-INT4"
PORT="${1:-9700}"

export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_ROCM_ARCH="gfx906"
export OMP_NUM_THREADS=4
export VLLM_USE_TRITON_FLASH_ATTN=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export VLLM_USE_TRITON_AWQ=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_TRITON_GEMM=0
export VLLM_ROCM_USE_AITER_TRITON_ROPE=0
export VLLM_ROCM_USE_AITER_FP8BMM=0
export VLLM_ROCM_USE_AITER_FP4BMM=0
export VLLM_ROCM_USE_AITER_LINEAR=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_RMSNORM=0
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTN=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export VLLM_GFX906_GEMV="${VLLM_GFX906_GEMV:-0}"
# gfx906 decode-GEMV hatches (A/B overridable from the invoking env):
#   All three GLM53/GFX906 GEMV hatches default OFF: measured 2026-10-03
#   as net-NEGATIVE at MiMo shapes (wna16 GEMV 5.7-9x slower than the
#   moe_wna16 CUDA kernel at every config; gemv_m 3-6x slower than LLMM1
#   on small N). A/B only.
#   WNA16  : int4 expert MoE decode GEMV
#   DENSE  : LLMM1 M==1 fp16 -> gemv_m
#   INT4   : Exllama CT int4 dense decode GEMV (layer-0 qkv only on MiMo)
export VLLM_GLM53_WNA16_GEMV="${VLLM_GLM53_WNA16_GEMV:-0}"
export VLLM_GLM53_DENSE_GEMV="${VLLM_GLM53_DENSE_GEMV:-0}"
export VLLM_GLM53_INT4_GEMV="${VLLM_GLM53_INT4_GEMV:-0}"
# router gate fp16 skinny GEMM (profile: bf16 Tensile gate GEMM was 41ms/step)
export VLLM_MIMO_GATE_FP16_GEMV="${VLLM_MIMO_GATE_FP16_GEMV:-1}"
export VLLM_GFX906_MLP_FP32_DOWN=1
# gfx906 MoE PREFILL hatch: dequant int4 experts -> fp16 Tensile GEMMs in
# gfx906_ext/moe_dqmm.py (Triton fused MoE tops out ~3.5 TFLOP/s = 89% of
# prefill GPU; Tensile sustains ~12 at 16384-token chunks). Measured
# 2026-10-04: 737/521 tok/s @20k/60k vs 205/184 Triton baseline (3.6x/2.8x);
# decode unaffected (path engages only when step tokens*topk >= 8192).
# Pair-capped + cached workspaces; survives sweep->decode soak. Default ON.
export VLLM_GFX906_MOE_DQMM="1"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200

# DISABLE_PREFIX_CACHE=1 boots with prefix caching OFF (--no-enable-prefix-caching)
# for the 2026-10-03 quality A/B (stray-'.' + multi-turn incoherence). Default
# keeps --enable-prefix-caching, unchanged behavior.
PREFIX_CACHE_FLAG="--enable-prefix-caching"
if [ -n "${DISABLE_PREFIX_CACHE:-}" ]; then
  PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
  echo "prefix caching DISABLED (DISABLE_PREFIX_CACHE set)"
fi

source "$VENV_PATH/bin/activate"
echo "Launching MiMo-V2.6-Flash-INT4 OMNI (port $PORT)..."
exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name mimo-v2.6-flash \
    --tensor-parallel-size 8 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-memory-bytes "${KV_CACHE_BYTES:-3758096384}" \
    ${KV_OFFLOAD_GB:+--kv-offloading-size $KV_OFFLOAD_GB} \
    ${DISABLE_HYBRID_KM:+--disable-hybrid-kv-cache-manager} \
    --skip-mm-profiling \
    --max-num-seqs 8 \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-16384}" \
    ${LONG_PREFILL_THRESHOLD:+--long-prefill-token-threshold $LONG_PREFILL_THRESHOLD} \
    --block-size 256 \
    $PREFIX_CACHE_FLAG \
    --enable-prompt-tokens-details \
    --trust-remote-code \
    --dtype float16 \
    --kv-cache-dtype auto \
    --quantization compressed-tensors \
    --compilation-config '{"mode": 0, "cudagraph_mode": "FULL"}' \
    --limit-mm-per-prompt '{"image":2,"video":1,"audio":2}' \
    --reasoning-parser deepseek_r1
