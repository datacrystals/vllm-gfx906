## Models running on this fork (8x AMD MI50 32GB, gfx906)

Two production models; all numbers measured live on the box (no matrix
cores, PCIe gen3, no working P2P). Details are in the per-model
sections below.

| Model | Weights | Decode | Prefill | Context / concurrency |
|---|---|---|---|---|
| MiMo-V2.6-Flash (omni: audio/image/video in, text out, thinking) | in-house INT4 (compressed-tensors g32 asymmetric, `mimo_convert_int4` from [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)) | 24.6 tok/s | 737 tok/s @20k / 521 @60k | 262,144 ctx; 3x resident @256k |
| GLM-5.3-Flash | [cyankiwi/GLM-5.3-Flash-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-Flash-AWQ-INT4) (activation-aware AWQ) | 15.2-16.6 tok/s | 318 tok/s @60k | 262,144 ctx (needle @249k); 2.25x @256k |

### Optimizations shipped

MiMo-V2.6-Flash (Oct 2026):
- **dqmm prefill path**: dequant int4 experts -> fp16 Tensile GEMMs
  (Triton tl.dot has no MFMA on gfx906 and capped prefill at ~3.5
  TFLOP/s) = 3.6x/2.8x prefill; pair-capped + cached workspaces
  (`vllm/gfx906_ext/moe_dqmm.py`)
- **Router-gate fp16 skinny-GEMM reroute**: 41.4 ms -> ~1 ms of an 81 ms
  decode step (Tensile bf16 M=1 path is slow on this arch)
- **moe_wna16 fp32-atomic combine** -- numerics fix for the
  output-corruption issue (costs ~13% decode, measured)
- **KV-pool sizing + long-prefill gating** -> 3x 256k sessions resident

GLM-5.3-Flash (Sept 2026):
- **KV pool decoupling**: KDA recurrent state out of the per-block
  charge (20.76 -> 2.94 MiB/block) + fp16 KDA state w/ fp32 accumulate
  -> 256k context on 8x32GB
- **ACS register reprogramming** (PLX downstream + Intel root ports):
  8-way all-reduce 456 -> 132 us = 3.5x collectives
- **Union-GEMM sparse-MLA prefill path** (needle-gated @ 71k)
- **MTP speculative-decoding glue** (written; gated off pending a
  verify fix)

---

## MiMo-V2.6-Flash (omni) port — status (8x MI50, Oct 2026)

The same 8x MI50 box also serves **MiMo-V2.6-Flash-INT4**
(`MiMoV2OmniForCausalLM`): audio/image/video input, text output, with
reasoning content returned separately.

| Metric | Result | Notes |
|---|---|---|
| Decode (1 user) | 24.5 tok/s | Target was 20. Was 28.07 before the numerics fixes; the ~13% difference is the cost of the fp32-atomic corrections (independent of chunk-size setting: 24.37 @2048 vs 24.68 @16384, 5 reps each) |
| Needle-in-haystack recall | passing at 24k / 65k / 130k | 24k re-run after the numerics fix; 65k/130k measured before it (130k = 102.9k tokens, 803s) |
| 150k agent-behavior probe | passing (MIMO-150K-V3B) | 5/5 facts recalled, 3/3 traps refused at 177k tokens |
| Concurrency @256k | 3.18x pool; 3 resident + 3/3 completion verified | Pool holds 3.18x262144 by construction; completion run 3/3 on the dqmm config (walls 846/1696/2538s; prefills serialize at full chunks); residency gates table in MIMO_PORT_PLAN; RAM offload deferred |
| Prefill | 737 tok/s @20k / 521 tok/s @60k | dqmm path: dequantize int4 experts to fp16 and run Tensile GEMMs (VLLM_GFX906_MOE_DQMM, default on); 3.6x/2.8x over the 205/184 Triton baseline (tl.dot lowers to FMA on gfx906, no MFMA); full writeup in MIMO_PORT_PLAN |
| Modalities | audio, image, video inputs working | vision outputs not present in this checkpoint |

Decode note: profiling showed a single bf16 router-gate GEMM taking
41.4 ms of an 81 ms step (51%). Tensile's bf16 M=1 path is slow on
gfx906; rerouting it to the fp16 skinny-GEMM path reduced it to about
1 ms. The hand-written GEMV kernels were measured and left off.

Notable issues resolved (details in MIMO_PORT_PLAN.md):

- Fused-QKV layout: the checkpoint stores fused qkv_proj pre-sharded
  for TP4 with per-chunk 128x128 block scales; naive flat reads
  scramble Q/K/V. Fixed with the per-chunk-padded regroup.
- flash-attn Triton-AMD silently ignores `window_size`, so every ViT
  SWA block ran full attention. The MiMo ViT now uses an explicit SDPA
  window mask + sinks. (The LM is unaffected: it runs vLLM's own
  TritonAttentionBackend, VLLM_USE_TRITON_FLASH_ATTN=1, which honors
  window + sinks.)
- fp16 overflows in the ViT (block27 absmax ~5e5) produced NaN image
  features. The tower now runs BF16; post-fix tower matches HF at
  cosine similarity 0.9987.
- Reasoning-parser leak: the split override never stripped the opening
  think marker, leaking it into reasoning_content. Fixed with
  stream-safe head handling.
- Intermittent boot failures at post-load: silent engine-parent death
  with clean dmesg. A debug checksum block is the suspect and is now
  gated behind VLLM_MIMO_AUDIT=1.
- Machine freezes: amdgpu SVM/KFD workqueues (svm_range_restore_work)
  under pinned-VRAM churn. The expandable_segments config is
  SVM-backed and is the first A/B target. See CRASH_FORENSICS.md.

Quality investigation (measured): one root cause for both reported
symptoms -- T=0 forward-pass nondeterminism from fp16 atomic split-K
reductions in the quantized GEMM kernels (gfx906 has no native fp16
atomics; partial results land in racy order, and the GPTQ kernel's
zero-init raced the accumulation). Near-tie argmax flips occur every
~10-40 tokens; a flip onto punctuation produced the stray "."
reports, and a cascading flip produced the multi-turn incoherence
reports (behavior probe: same prompt, 3 runs -> FAIL/PASS/FAIL; the
failing run dropped 4/5 embedded facts). The earlier "prefix-cache
corruption" reading is retracted: the warm/cold splits were the noise
floor, cache-off runs are warm==cold identical (5/5), and the SWA
block accounting audits clean.
Status: both fixes shipped and verified loaded -- the moe_wna16
combine (fp32 atomicAdd shadow buffer, ~1e4 noise reduction;
b9f7226387) and the dominant site, the dense INT4 GEMM in
csrc/quantization/gptq/q_gemm.cu (all 48 layers, every token): fp32
accumulator + launcher-side zero-init + `::atomicAdd` (the compat.cuh
half-atomics hide the float builtin inside `namespace vllm::gptq`;
8c002f93cc, f1693f36f3). Residual T=0 noise is hipBLAS/Tensile
split-K; a one-boot discriminator is documented
(VLLM_ROCM_USE_SKINNY_GEMM=1 + hammer). Practical mitigation:
temperature >= 0.1 turns the tie-flips into intentional sampling.
Full evidence in QUALITY_HYPOTHESES.md.
User verification (2026-10-04): at T=1.0 top_p=1.0 the residual noise
(~1-2 nat logit perturbation) surfaced as occasional nonsense-token
bursts; T 0.6-0.8 + top_p 0.9-0.95 + min_p ~0.05 eliminated all
observed artifacts in live use. Recommended client configuration.

Remaining items: prefill NBN sweep complete (154->205 tok/s) and the
kernel-level change complete (dqmm -> 737/521 tok/s @20k/60k, with 3
integration postmortems in MIMO_PORT_PLAN.md); decode cost of the
fp32-atomic corrections measured (28.07 pre-fix -> 24.4-24.7
post-fix, NBN-independent); RAM offload for parked sessions not yet
verified (deprioritized); residual T=0 nondeterminism under greedy
decoding (mitigate with temperature >= 0.1).

---

## GLM-5.3-Flash production numbers (8x MI50, Sept 2026)

All numbers below were measured on 8x AMD MI50 32GB (gfx906, PCIe, no
matrix cores) serving GLM-5.3-Flash-AWQ-INT4 with this fork.

| Metric | Result | Notes |
|---|---|---|
| Context | 262,144 tokens | needle-validated at 249k @ 50% depth |
| Concurrency @256k | 2.25x | two full-length conversations at once |
| Context @128k | 4.72x concurrency | needle recall passing to 100k |
| Context @512k | 1.08x single-shot | known transient during deep prefill |
| Decode | 15.2-16.6 tok/s | +45% from MTP and collective tuning |
| Prefill | 318 tok/s @ 60k | union-GEMM sparse-MLA path (validated at 71k) |
| All-reduce (8-way) | 132 us (was 456) | ACS register reprogramming |
| KV pool | 590k tokens @256k config | mamba-state pool decoupling |

Engineering notes behind the numbers:

- KV pool decoupling: the KDA recurrent state was removed from the
  per-block charge (20.76 -> 2.94 MiB/block), which is what allows 256k
  context on 8x32GB.
- fp16 KDA recurrent state (fp32 accumulate in-kernel): +50% KV
  capacity, quality verified with needle probes at 100k depth.
- ACS register reprogramming: PLX downstream ports and Intel root-port
  ACS redirect bits reprogrammed after boot, giving 3.5x collectives.
- MTP speculative-decoding glue for the GLM53 draft layer (written;
  gated off pending a verify fix).
- Operational issues seen along the way: corrupt-pyc segfaults from
  hard resets, long triton compile times that look like hangs, and one
  GPU wedge producing degenerate `!!!!` outputs that was root-caused to
  hung DMA fences rather than code (postmortem in
  PREFILL_KERNEL_PLAN.md). Kernel changes are validated with needle
  probes before they ship, using exact-prompt and exact-token-count
  reproduction.

---

## Mini Install Guide for GFX906

### Using the pre-built Docker image (recommended)

If you have Docker and the AMD ROCm drivers/kernel modules installed on your host system, you can skip the manual source build by using the pre-built Docker image.

```bash
# Pull the latest image (or specify a tag instead of latest, e.g. v0.19.1rc0.x)
docker pull aiinfos/vllm-gfx906-mobydick:latest

# Run the container interactively (Make sure to pass ROCm devices into the container and have your models in host /home/ as we map /home:/home; feel free to edit the command below to a safer one, without priviledged and others)
sudo docker run -it --name vllm-gfx906-mobydick -v /home:/home --network host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add $(getent group render | cut -d: -f3) \
  --cap-add=SYS_ADMIN --volume /sys:/sys:ro --pid=host --privileged \
  --ipc=host aiinfos/vllm-gfx906-mobydick:latest
```

Once inside the container, you can start serving models (see the Quickstart example below).

---

### Manual build from source

If you prefer to build and install from source on your bare metal instead, follow the steps below:

### ROCm 6.3.4 & amdgpu drivers

```code
# Get the script that adds the AMD repo for 24.04 (noble)
wget https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb
sudo apt install ./amdgpu-install_6.3.60304-1_all.deb

# Install ROCm  6.3.4 including hip, rocblas, amdgpu-dkms etc (assuming the machine has already the advised compatible kernel 6.11)
sudo amdgpu-install --usecase=rocm --rocmrelease=6.3.4    

sudo usermod -aG render,video $USER

# Verify ROCm installation
rocm-smi --showproductname --showdriverversion
rocminfo


# Add iommu=pt if you later grow beyond two GPUs
# ROCm’s NCCL-/RCCL-based frameworks can hang on multi-GPU rigs unless the IOMMU is put in pass-through mode
# see https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.3/reference/install-faq.html#multi-gpu

sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt /' /etc/default/grub
sudo update-grub
sudo reboot
cat /proc/cmdline  # >>> to check: must return: "BOOT_IMAGE=... iommu=pt"

```

### vllm-gfx906-mobydick fork with its dependencies (python, torch, triton, flash-attn, etc)

```code

pyenv install 3.12.11
pyenv virtualenv 3.12.11 venv312
pyenv activate venv312

# PYTORCH 2.11.0

git clone --branch v2.11.0 --recursive https://github.com/pytorch/pytorch.git
cd pytorch

# Install Python Dependencies
pip install -r requirements.txt
pip install mkl-static mkl-include

# Hipify the Source (Convert CUDA to ROCm code)
python tools/amd_build/build_amd.py

# Build the wheel and install
export MAX_JOBS=96 # to be adjusted according to your setup to avoid OOM / freeze / crash
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx906
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH}"

pip wheel --no-build-isolation -v -w dist -e . 2>&1 | tee build.log
pip install ./dist/torch*.whl


# TORCHVISION 0.26.0

# Install dependencies
sudo apt-get update && sudo apt-get install -y libpng-dev libjpeg-dev ffmpeg

# Build and Install
git clone --branch v0.26.0 https://github.com/pytorch/vision.git
cd vision
export FORCE_CUDA=1
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx906

python setup.py install


# TORCHAUDIO 2.11.0

# Build and Install
git clone --branch v2.11.0 https://github.com/pytorch/audio.git
cd audio
export PYTORCH_ROCM_ARCH=gfx906
export USE_ROCM=1

python setup.py install


# TRITON-GFX906 V3.6.0

git clone --branch v3.6.0+gfx906 https://github.com/ai-infos/triton-gfx906.git
cd triton-gfx906 
pip install -r python/requirements.txt
TRITON_CODEGEN_BACKENDS="amd" pip wheel --no-build-isolation -w dist . 2>&1 | tee build.log
pip install ./dist/triton-*.whl  


# FLASH-ATTENTION-GFX906 (triton backend)

git clone https://github.com/ai-infos/flash-attention-gfx906.git
cd flash-attention-gfx906
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install

# VLLM-GFX906-MOBYDICK main

git clone https://github.com/ai-infos/vllm-gfx906-mobydick.git
cd vllm-gfx906-mobydick
pip install 'amdsmi>=6.3,<6.4'
pip install -r requirements/rocm.txt
pip wheel --no-build-isolation -v -w dist . 2>&1 | tee build.log
pip install ./dist/vllm-*.whl

# TRANSFORMERS (v5.7.0 or any other version <6 supporting your model)
pip install transformers==5.7.0
```

### Quickstart example (with Qwen3.5-0.8B)

```code
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3.5-0.8B \
  --dtype float16 \
  --kv-cache-dtype float16 \
  2>&1 | tee log.txt
```

NB: --dtype float16 is recommended to add for this gfx906 fork. If not set, vllm will take the dtype from config.json model which might be bfloat16, not natively supported on gfx906 (with potential fallback to float32, leading to slower inference)

CREDITS
-------

- https://github.com/nlzy/vllm-gfx906
- https://github.com/Said-Akbar/vllm-rocm
- https://github.com/vllm-project/vllm

---

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
