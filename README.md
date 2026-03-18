# vLLM-gfx906

This is a fork of vLLM optimized for (and only for) AMD gfx906 architecture (Radeon VII and MI50). Please feel free to make pull requests and let me know when stuff works/doesnt work. I will try to get around to this as I am able.

### Project Status
The original maintainer (nlzy) has stopped updates. I intend to provide occasional maintenance to keep the engine running on this hardware.

### Performance and Compatibility
* **Verified Model:** This fork has been tested successfully with **MiniMax-M2.5-AWQ** (a2s-ai/MiniMax-M2.5-AWQ) using 8x MI50 32GB GPUs.
* **Throughput:** Currently achieves approximately 80% of the performance seen in vLLM v0.13. It is usable but slower than older versions. 
* **Future Work:** I will attempt to improve performance through kernel tuning as time permits.

---

## System Requirements

| Component | Version |
| :--- | :--- |
| **GPU** | AMD MI50 (32GB) or Radeon VII (16GB) |
| **ROCm** | 6.4.3 |
| **Python** | 3.12 |
| **PyTorch** | 2.9.0 |

---

## Build and Installation

Follow these steps to avoid common memory and instruction set errors.


```bash
python3 -m venv venv
source venv/bin/activate

export PYTORCH_ROCM_ARCH="gfx906"

pip3 install -r requirements/rocm-build.txt -r requirements/rocm.txt

pip3 install --no-build-isolation --no-deps -v .

```

---

## Running the Server

Use the following command to run MiniMax-M2.5-AWQ. Note that `enforce-eager` and `disable-custom-all-reduce` are required for stability on this hardware.

```bash
vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 8 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.80 \
    --trust-remote-code \
    --port 9100 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --kv-cache-dtype auto \
    --max-num-batched-tokens 8192 \
    --dtype float16 \
    --disable-custom-all-reduce \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
```

---

## Technical Notes

* **Vision Model Bypass:** I have disabled several multimodal models (including InternVL) in the registry. These models import torchvision video components that cause immediate segmentation faults on gfx906.
* **Kernel Patches:** Modified moe_wna16 kernels to replace NVIDIA-specific assembly with ROCm bit-field instructions.
* **Atomic Operations:** Added software fallbacks for half-precision atomic additions which are not natively supported by this hardware's instruction set.

