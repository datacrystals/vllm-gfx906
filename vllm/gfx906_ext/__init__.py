# SPDX-License-Identifier: Apache-2.0
"""gfx906 (MI50/MI60) runtime extensions for this fork.

Vendored from the out-of-repo patch pack
``/data/vllm-gfx906-dsv4/patches/gdn/`` (kept on disk for provenance only;
this package supersedes it) so the ``glm53-gfx906`` branch is self-contained
for every production model config on this rig (Qwen3.5 hybrid-GDN, Hy3,
DSV4, GLM-5.3-Flash).

All modules are OFF unless their env gate is set; ``import vllm.gfx906_ext``
itself has no side effects. Activation happens from the env-gated anchors at
the tails of ``vllm/model_executor/models/hy_v3.py`` and
``vllm/model_executor/models/glm5next/__init__.py``, or by importing a
submodule directly (sitecustomize-style).

Modules:

- ``gfx906_gemv``         skinny fp16 GEMV (M<=8) override — ``VLLM_GFX906_GEMV=1``
- ``gdn_gfx906_fallback`` pure-torch fp32 GDN prefill/decode fallback —
                          ``VLLM_GDN_GFX906_AUTOPATCH=1`` (enablement itself via
                          ``VLLM_GDN_GFX906_FALLBACK=auto|0|1``); also handles
                          ``VLLM_GDN_GFX906_FUSED_DECODE``, ``VLLM_GDN_GFX906_NAN_PROBE``,
                          ``VLLM_GFX906_MLP_CLAMP``, ``VLLM_GFX906_MLP_FP32_DOWN``
- ``gdn_decode_fused``    fused single-kernel GDN decode recurrence —
                          ``VLLM_GDN_GFX906_FUSED_DECODE=1`` (installed from the fallback)
- ``prof_patch``          trigger-file torch.profiler harness — ``VLLM_GFX906_PROF_DIR``
- ``glm53_mtp_patch``     GLM-5.3 MTP (speculative decode) glue — PARKED/optional,
                          ``VLLM_GFX906_GLM53_MTP=1`` (+ ``VLLM_GFX906_GLM53_MTP_FULL_CG=1``)
- ``glm53_mtp_main``      PARKED/optional driver entrypoint for the above
                          (``python -m vllm.gfx906_ext.glm53_mtp_main serve ...``)
"""
