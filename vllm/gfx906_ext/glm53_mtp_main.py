# GLM-5.3-Flash MTP driver entrypoint: installs glm53_mtp_patch glue before
# vllm CLI parsing, so SpeculativeConfig post-init sees the patched overrides.
# Forked engine/worker processes inherit the already-installed patch.
#
# STATUS: PARKED/optional — GLM-5.3 MTP was parked (see GLM53_GFX906_STATUS.md);
# this driver is kept for a future bring-up attempt. No-op without
# VLLM_GFX906_GLM53_MTP=1.
#
# Vendored into vllm/gfx906_ext/ (supersedes
# /data/vllm-gfx906-dsv4/patches/gdn/glm53_mtp_main.py, provenance only). Run:
#   VLLM_GFX906_GLM53_MTP=1 python -m vllm.gfx906_ext.glm53_mtp_main serve ...
from vllm.gfx906_ext import glm53_mtp_patch  # noqa: F401

from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    main()
