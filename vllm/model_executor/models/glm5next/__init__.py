# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import current_platform

if current_platform.is_xpu():
    raise NotImplementedError("GLM-5.3-Flash does not currently support XPU.")

from .nvidia.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
from .nvidia.mtp import Glm5NextMTP

__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextMTP",
]

# --- gfx906 profiler anchor (installed when VLLM_GFX906_PROF_DIR set) ---
if __import__("os").environ.get("VLLM_GFX906_PROF_DIR"):
    from vllm.gfx906_ext import prof_patch  # noqa: F401 (self-installs on Worker.execute_model)


# --- gfx906 skinny-GEMV anchor (installed when VLLM_GFX906_GEMV=1) ---
# Replaces generic tiled triton_matmul for skinny fp16 GEMMs (M<=8) with
# hand-built bandwidth-bound GEMV (vllm/gfx906_ext/gfx906_gemv.py; Track H C2
# microbench PASS). Default OFF.
if __import__("os").environ.get("VLLM_GFX906_GEMV") == "1":
    from vllm.gfx906_ext import gfx906_gemv as _gemv
    _gemv.install_gfx906_gemv()

