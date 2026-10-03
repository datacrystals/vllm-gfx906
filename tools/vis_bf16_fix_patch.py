#!/usr/bin/env python3
"""Fix: run the MiMo vision tower in BF16 (fp16 overflows block27)."""
import shutil

TREES = [
    "/data/vllm-gfx906-dsv4/vllm/vllm/model_executor/models/mimo_v2_omni.py",
    "/data/vllm-gfx906-dsv4/vllm_dsv4_env/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2_omni.py",
]

OLD = '''        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = MiMoVisionTransformer(
                vision_config,
                norm_eps=getattr(vllm_config, "rms_norm_eps", 1e-6),
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )
        audio_config = getattr(config, "audio_config", None)'''

NEW = '''        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = MiMoVisionTransformer(
                vision_config,
                norm_eps=getattr(vllm_config, "rms_norm_eps", 1e-6),
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )
        # The ViT's last block reaches |activation| ~ 5e5 in the reference
        # trace (exceeds the float16 max of 65504): a float16 tower overflows
        # to inf there and the merger emits NaN features, which the LM then
        # splices in for the image tokens. Checkpoint visual.* weights are
        # BF16; run the whole tower in BF16 regardless of the LM dtype.
        self.visual.to(torch.bfloat16)
        audio_config = getattr(config, "audio_config", None)'''

for p in TREES:
    src = open(p).read()
    assert "self.visual.to(torch.bfloat16)" not in src, f"already fixed: {p}"
    assert src.count(OLD) == 1, f"anchor not unique in {p}: {src.count(OLD)}"
    shutil.copy2(p, p + ".bak-bf16fix")
    open(p, "w").write(src.replace(OLD, NEW))
    print("FIXED:", p)
