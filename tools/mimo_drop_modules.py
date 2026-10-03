#!/usr/bin/env python3
"""Standalone wrapper: write a text-only MiMo-V2 config.json for
/data/ModelDownloader/MiMo-V2.6-Flash-INT4 (or any destination).

Removes vision_config / audio_config / processor_config so a vLLM text-only
load does not build the vision/audio encoders; keeps `architectures` and
`model_type: mimo_v2` unchanged; installs the compressed-tensors INT4
group-32-asymmetric quantization_config (same structure as the fork's GLM
AWQ-INT4 checkpoints) with an ignore list covering every non-quantized module.

Implementation lives in mimo_convert_int4.py (kept identical in
/data/vllm-gfx906-dsv4/tools/ and /data/vllm-gfx906-dsv4/vllm/tools/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mimo_convert_int4 as m  # noqa: E402


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=m.SRC_DIR)
    p.add_argument("--dst", default=m.DST_DIR)
    p.add_argument("--out", default=None,
                   help="output config.json path (default <dst>/config.json)")
    args = p.parse_args()
    m.torch.set_num_threads(min(32, m.os.cpu_count() or 8))
    m.cmd_drop_modules(args)


if __name__ == "__main__":
    main()
