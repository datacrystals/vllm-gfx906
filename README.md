## MiMo-V2.6-Flash (omni) port — status (8x MI50, Oct 2026)

The same 8x MI50 box now also serves **MiMo-V2.6-Flash-INT4**
(`MiMoV2OmniForCausalLM`) -- audio/image/video in, text out, with thinking.

| Metric | Number | Notes |
|---|---|---|
| Needles | **24k GREEN (post-fix)**; 65k/130k GREEN pre-fix | 24k re-run after the gate fp16 change; 65k/130k predate it |
| 150k agent behavior | **PASS** (MIMO-150K-V3B) | 5/5 facts, 3/3 destructive traps refused @ 177k tok |
| Decode | **28.07 tok/s** (was 12.2) | target 20 MET. Win = router-gate fp16 skinny GEMM |
| Concurrency @256k | **3.16x capacity / 3-way live** | 5.6 GiB KV pool; 3 sessions resident simultaneously |
| Prefill | **142-156 tok/s** (untuned) | tuning pass in progress (batched-tokens sweep) |
| Modalities | **audio + image + video all work** | vision outputs not in this checkpoint (no gen head) |

Decode story: the gfx906/GLM53 GEMV hatches (int4 dense, wna16 MoE,
LLMM1) were wired and measured -- all NEGATIVE at MiMo shapes (wna16 GEMV
5.7-9x slower than the moe_wna16 CUDA kernel; gemv_m 3-6x slower than
LLMM1), so they ship defaulted OFF. The real decode hog was the MoE router
gate: a bf16 nn.Linear hitting a Tensile GEMM at 880 us/call x47 =
41 ms/step of an 81 ms step (prof_patch trace). Routing it through
rocm_unquantized_gemm in fp16 (VLLM_MIMO_GATE_FP16_GEMV=1, logits cast
back to gate dtype) took that to ~1 ms/step -> 28.07 tok/s.

Known limits / ops notes (2026-10-03 perf campaign):

- **Concurrency tuning**: `--max-num-batched-tokens 2048` gives 3.16x
  max-concurrency at 262,144 tokens (5.6 GiB KV pool); raising the budget
  to 8192 drops it to 2.85x (bigger per-session reservation). Concurrent
  long prefills need `--long-prefill-token-threshold` < budget/n (default
  0 serializes: one prefill owns the whole budget per step).
- **KV pool knob**: `KV_CACHE_BYTES` env on run_mimo_v2_6_omni.sh
  (3.5 GiB default = 1.98x, 5.6 GiB = 3.16x).
- **Router-gate fix env**: `VLLM_MIMO_GATE_FP16_GEMV=1` (default on).
- **Teardown rule** (3 hard-wedged SIGTERMs -> power resets on this box):
  signal the server only with an empty queue and >=2 GB free VRAM, else
  `sudo ipmitool chassis power reset`. Close prof_patch windows before
  restarts. After any reset: delete venv *.pyc, `systemctl --user stop
  glm53.service` (it auto-respawns and grabs :9700), then fleet_free gate.
- **`sync` before power resets** (a run-script edit was lost to dirty-page
  eviction once).
- Bare `import vllm._custom_ops` as the first vllm import segfaults in
  standalone processes on this box (import models.mimo_v2 first).

Port war stories worth knowing (full writeup in MIMO_PORT_PLAN.md):

- **Fused-QKV word-salad monster**: the checkpoint stores each layer's fused
  qkv_proj pre-sharded for TP4 with per-chunk 128x128 block scales; a naive
  flat `[q|k|v]` read scrambles Q/K/V into salad. Fix = per-chunk-padded
  scale rows + row regroup at dequant. All three implementations (vLLM, HF
  remote-code, pure-torch ref) were identically garbage until this.
- **flash-attn Triton-AMD silently ignores `window_size`** -- every
  sliding-window block silently runs full attention. Fork-wide hazard; the
  MiMo ViT's 24 SWA blocks now use an explicit SDPA window mask + sinks.
- **fp16 overflows the ViT** (block27 absmax ~5e5 > 65504) -> NaN image
  features -> "!!!!" walls. Tower now runs BF16.
- Serving preprocessor needed CLIP stats (not ImageNet); merger needs
  LayerNorm (not RMSNorm). Post-fix tower matches HF at cos 0.9987.

---

## GLM-5.3-Flash production numbers (8x MI50, Sept 2026)

Everything below is **measured on 8x AMD MI50 32GB (gfx906, PCIe, no matrix
cores)** -- the "can't run big models" GPUs -- serving
**GLM-5.3-Flash-AWQ-INT4** with this fork. Yes, really.

| Metric | Number | Notes |
|---|---|---|
| Context | **262,144 tokens** | needle-validated at 249k @ 50% depth |
| Concurrency @256k | **2.25x** | two full-length conversations at once |
| Context @128k | 4.72x concurrency | needles green to 100k |
| Context @512k | 1.08x single-shot | deep-prefill transient known |
| Decode | **15.2-16.6 tok/s** | +45% via MTP + collective tuning |
| Prefill | **318 tok/s @ 60k** | union-GEMM sparse-MLA path shipped (needle-gated @ 71k) |
| All-reduce (8-way) | **132 us** (was 456) | ACS register reprogramming |
| KV pool | 590k tokens @256k cfg | custom mamba-state pool decoupling |

Engineering highlights behind the numbers:

- **KV pool decoupling**: KDA recurrent state removed from the per-block
  charge (20.76 -> 2.94 MiB/block) -- turns 8x32GB into a 256k-context box.
- **fp16 KDA recurrent state** (fp32 accumulate in-kernel): +50% KV
  capacity, quality-verified with needle probes at 100k depth.
- **ACS register map cracked**: PLX downstream + Intel root-port ACS
  redirect bits reprogrammed live (post-boot unit) -> 3.5x collectives.
- **MTP speculative decoding glue** for the GLM53 draft layer (gated off
  pending a verify fix).
- Survived: corrupt-pyc segfaults from hard resets, triton compile
  marathons masquerading as hangs, one very stubborn switch fabric, and a
  GPU wedge that gaslit an entire evening with degenerate `!!!!` outputs --
  root-caused to hung DMA fences, not code (postmortem in
  PREFILL_KERNEL_PLAN.md). Every kernel path is needle-gated before it
  ships; exact-prompt/token-count repro is the house style.

---

## Mini Install Guide for GFX906

### 🐳 Using Pre-built Docker Image (Recommended)

If you have Docker and the AMD ROCm drivers/kernel modules installed on your host system, you can totally bypass the complex manual source-build installation by using our pre-built Docker image.

```bash
# Pull the latest image (or specify a tag instead of latest, e.g. v0.19.1rc0.x)
docker pull aiinfos/vllm-gfx906-mobydick:latest

# Run the container interactively (Make sure to pass ROCm devices into the container and have your models in host /home/ as we map /home:/home; feel free to edit the command below to a safer one, without priviledged and others)
sudo docker run -it --name vllm-gfx906-mobydick -v /home:/home --network host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add $(getent group render | cut -d: -f3) \
  --cap-add=SYS_ADMIN --volume /sys:/sys:ro --pid=host --privileged \
  --ipc=host aiinfos/vllm-gfx906-mobydick:latest
```

Once inside the container, you are all set! You can immediately start serving models (see the Quickstart example below).

---

### 🛠️ Manual Build from Source

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
